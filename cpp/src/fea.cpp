#include "core.hpp"
#include "fea.hpp"
#include "multigrid/padding.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

// CBLAS for optimized matrix operations
#ifdef HAVE_CBLAS
  #ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

namespace top3d {

// In debug builds, verify 64-byte alignment when we claim it
#ifndef NDEBUG
static inline void assert_aligned_64(const void* ptr) {
	auto p = reinterpret_cast<std::uintptr_t>(ptr);
	assert((p % 64u) == 0 && "Ke is not 64-byte aligned!");
}
#endif

// Reusable view bundling invariant raw pointers for kernels
struct FEAKernelView {
	int numElements;
	const int* __restrict__ eNod;
	const double* __restrict__ Kptr;
};

static inline FEAKernelView make_fea_kernel_view(const Problem& pb) {
	const auto& mesh = pb.mesh;
	FEAKernelView v{
		mesh.numElements,
		mesh.eNodMat.data(),
		mesh.Ke.data()
	};
	return v;
}

// 64-bit Morton helpers: support up to 21 bits per coordinate (2,097,151)
static inline std::uint64_t part1by2_64(std::uint64_t x) {
	// Spread the lower 21 bits of x so that there are 2 zero bits between each
	x &= 0x1fffffULL;                          // keep only 21 bits
	x = (x | (x << 32)) & 0x1f00000000ffffULL;
	x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
	x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
	x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
	x = (x | (x << 2))  & 0x1249249249249249ULL;
	return x;
}

static inline std::uint64_t morton3D_64(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
	// Assumes x,y,z < 2^21
	std::uint64_t xx = part1by2_64(x);
	std::uint64_t yy = part1by2_64(y);
	std::uint64_t zz = part1by2_64(z);
	return (xx << 2) | (yy << 1) | zz;
}

// Inverse of nodeIndex(ix,iy,iz) = nnx*nny*iz + nnx*iy + ix
static inline void node_ijk(int n, int nnx, int nny, int& ix, int& iy, int& iz) {
	int slab = nnx * nny;
	iz = n / slab;
	int rem = n % slab;
	iy = rem / nnx;
	ix = rem % nnx;
}

// Given fullIdx = ny*nx*ez + ny*ex + (ny-1-ey)
static inline void elem_ijk_from_full(int fullIdx, int nx, int ny, int& ex, int& ey, int& ez) {
	int cellsPerLayer = nx * ny;
	ez = fullIdx / cellsPerLayer;
	int rem = fullIdx % cellsPerLayer;
	ex = rem / ny;
	int inv_ey = rem % ny; // inv_ey = ny-1-ey
	ey = (ny - 1) - inv_ey;
}

template<typename T>
static inline void permute_strided(std::vector<T>& a, const std::vector<int>& perm, int stride) {
	const int n = static_cast<int>(perm.size());
	std::vector<T> tmp(a.size());
	for (int newIdx = 0; newIdx < n; ++newIdx) {
		int oldIdx = perm[newIdx];
		int src = oldIdx * stride;
		int dst = newIdx * stride;
		for (int k = 0; k < stride; ++k) tmp[static_cast<size_t>(dst + k)] = a[static_cast<size_t>(src + k)];
	}
	a.swap(tmp);
}

template<typename T>
static inline void permute_unstrided(std::vector<T>& a, const std::vector<int>& perm) {
	const int n = static_cast<int>(perm.size());
	std::vector<T> tmp(static_cast<size_t>(n));
	for (int newIdx = 0; newIdx < n; ++newIdx) tmp[static_cast<size_t>(newIdx)] = a[static_cast<size_t>(perm[newIdx])];
	a.swap(tmp);
}

// Branchless maps from local DOF slot i in [0..23] to (local node a, component c)
static constexpr int node_of_i_[24] = {
	0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7
};
static constexpr int comp_of_i_[24] = {
	0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2
};

std::array<double,24*24> ComputeVoxelKe(double nu, double cellSize) {
	// Numerical 2x2x2 Gauss integration for an 8-node linear hex element
	double E = 1.0; // unit modulus; actual modulus scales at element level
	double lambda = E*nu/((1.0+nu)*(1.0-2.0*nu));
	double mu = E/(2.0*(1.0+nu));
	// Constitutive D (6x6) in Voigt order [xx,yy,zz,yz,xz,xy]
	double D[6][6] = {
		{lambda+2*mu, lambda,       lambda,       0,    0,    0},
		{lambda,       lambda+2*mu, lambda,       0,    0,    0},
		{lambda,       lambda,       lambda+2*mu, 0,    0,    0},
		{0,            0,            0,            mu,   0,    0},
		{0,            0,            0,            0,    mu,   0},
		{0,            0,            0,            0,    0,    mu}
	};
	std::array<double,24*24> Ke{};
	// Gauss points in [-1,1]
	const double gp[2] = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
	const double w[2] = {1.0, 1.0};
	// Mapping: natural -> physical with hx=hy=hz=cellSize; J = diag(h/2)
	double hx = cellSize, hy = cellSize, hz = cellSize;
	double detJ = (hx*hy*hz)/8.0;
	double sx = 2.0/hx, sy = 2.0/hy, sz = 2.0/hz; // d/dx = d/ds * sx, etc.
	for (int a=0;a<2;a++) for (int b=0;b<2;b++) for (int c=0;c<2;c++) {
		double s = gp[a], t = gp[b], p = gp[c];
		// Shape function derivatives in natural coords
		double dNds[8] = {
			-0.125*(1-t)*(1-p),  0.125*(1-t)*(1-p),  0.125*(1+t)*(1-p), -0.125*(1+t)*(1-p),
			-0.125*(1-t)*(1+p),  0.125*(1-t)*(1+p),  0.125*(1+t)*(1+p), -0.125*(1+t)*(1+p)
		};
		double dNdt[8] = {
			-0.125*(1-s)*(1-p), -0.125*(1+s)*(1-p),  0.125*(1+s)*(1-p),  0.125*(1-s)*(1-p),
			-0.125*(1-s)*(1+p), -0.125*(1+s)*(1+p),  0.125*(1+s)*(1+p),  0.125*(1-s)*(1+p)
		};
		double dNdp[8] = {
			-0.125*(1-s)*(1-t), -0.125*(1+s)*(1-t), -0.125*(1+s)*(1+t), -0.125*(1-s)*(1+t),
			 0.125*(1-s)*(1-t),  0.125*(1+s)*(1-t),  0.125*(1+s)*(1+t),  0.125*(1-s)*(1+t)
		};
		// Convert to physical derivatives
		double dNdx[8], dNdy[8], dNdz[8];
		for (int i=0;i<8;i++) { dNdx[i]=dNds[i]*sx; dNdy[i]=dNdt[i]*sy; dNdz[i]=dNdp[i]*sz; }
		// Build B (6x24)
		double B[6][24]; for (int r=0;r<6;r++) for (int c2=0;c2<24;c2++) B[r][c2]=0.0;
		for (int i=0;i<8;i++) {
			int c0 = 3*i;
			B[0][c0+0] = dNdx[i];
			B[1][c0+1] = dNdy[i];
			B[2][c0+2] = dNdz[i];
			B[3][c0+1] = dNdz[i]; B[3][c0+2] = dNdy[i]; // yz
			B[4][c0+0] = dNdz[i]; B[4][c0+2] = dNdx[i]; // xz
			B[5][c0+0] = dNdy[i]; B[5][c0+1] = dNdx[i]; // xy
		}
		// Ke += B' * D * B * w_s * w_t * w_p * detJ
		double DB[6][24]; for (int i=0;i<6;i++) for (int j=0;j<24;j++) { double sum=0.0; for (int k=0;k<6;k++) sum+=D[i][k]*B[k][j]; DB[i][j]=sum; }
		double weight = w[a]*w[b]*w[c]*detJ;
		for (int i=0;i<24;i++) {
			for (int j=0;j<24;j++) {
				double sum=0.0; for (int k=0;k<6;k++) sum += B[k][i]*DB[k][j];
				Ke[i*24+j] += sum * weight;
			}
		}
	}
	return Ke;
}

void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz) {
	CartesianMesh& mesh = pb.mesh;
	mesh.eleSize = {1.0f,1.0f,1.0f};
	// Record original resolutions (elements) before padding
	mesh.origResX = nelx;
	mesh.origResY = nely;
	mesh.origResZ = nelz;

    // Mirror MATLAB padding: compute levels and pad dims to multiples of 2^L
    const std::uint64_t numSolidVoxels = static_cast<std::uint64_t>(nely) * static_cast<std::uint64_t>(nelx) * static_cast<std::uint64_t>(nelz);
    const auto pad = top3d::compute_adjusted_dims(static_cast<std::uint64_t>(nelx),
                                                  static_cast<std::uint64_t>(nely),
                                                  static_cast<std::uint64_t>(nelz),
                                                  numSolidVoxels);
    mesh.resX = static_cast<int>(pad.adjustedNelx);
    mesh.resY = static_cast<int>(pad.adjustedNely);
    mesh.resZ = static_cast<int>(pad.adjustedNelz);

    const int nx = mesh.resX;
    const int ny = mesh.resY;
    const int nz = mesh.resZ;

    // Build padded voxel volume: original region = solid (1), padded region = void (0)
    std::vector<uint8_t> voxelized;
    voxelized.assign(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz), static_cast<uint8_t>(0));
    for (int z=0; z<nelz; ++z) {
        for (int x=0; x<nelx; ++x) {
            const int in_plane_out = ny * (x + nx * z);
            for (int y=0; y<nely; ++y) {
                // Our storage uses index with Y flipped: fullIdx = ny*nx*z + ny*x + (ny-1 - y)
                const int idx = in_plane_out + (ny - 1 - y);
                voxelized[idx] = static_cast<uint8_t>(1);
            }
        }
    }

	// Identify solid elements
	mesh.eleMapBack.clear();
	for (int z=0; z<nz; ++z) {
		for (int x=0; x<nx; ++x) {
			for (int y=0; y<ny; ++y) {
				int fullIdx = ny*nx*z + ny*x + (ny-1 - y);
				if (voxelized[fullIdx]) mesh.eleMapBack.push_back(fullIdx);
			}
		}
	}
	mesh.numElements = static_cast<int>(mesh.eleMapBack.size());
	mesh.eleMapForward.assign(nx*ny*nz, -1);
	for (int i=0;i<mesh.numElements;i++) 
		mesh.eleMapForward[ mesh.eleMapBack[i] ] = i;

	// Node numbering on full grid
	const int nnx = nx+1, nny = ny+1, nnz = nz+1;
	mesh.numNodes = nnx*nny*nnz;
	mesh.numDOFs = mesh.numNodes*3;

	// Lexicographic nodal order: identity maps
	mesh.nodMapForward.resize(mesh.numNodes);
	mesh.nodMapBack.resize(mesh.numNodes);
	for (int i = 0; i < mesh.numNodes; ++i) {
		mesh.nodMapForward[static_cast<size_t>(i)] = i;
		mesh.nodMapBack[static_cast<size_t>(i)] = i;
	}

	// eNodMat (compact by using compact node ids directly)
	mesh.eNodMat.resize(mesh.numElements*8);
	for (int ez=0; ez<nz; ++ez) {
		for (int ex=0; ex<nx; ++ex) {
			for (int ey=0; ey<ny; ++ey) {
				int fullIdx = ny*nx*ez + ny*ex + (ny-1 - ey);
				int comp = mesh.eleMapForward[fullIdx];
				if (comp < 0) continue;
				auto nodeIndex = [&](int ix,int iy,int iz){
					return (nnx*nny*iz + nnx*iy + ix);
				};
				// Local node order consistent with MATLAB generation ([1..8])
				int n1 = nodeIndex(ex,   ny-ey,   ez);
				int n2 = nodeIndex(ex+1, ny-ey,   ez);
				int n3 = nodeIndex(ex+1, ny-ey-1, ez);
				int n4 = nodeIndex(ex,   ny-ey-1, ez);
				int n5 = nodeIndex(ex,   ny-ey,   ez+1);
				int n6 = nodeIndex(ex+1, ny-ey,   ez+1);
				int n7 = nodeIndex(ex+1, ny-ey-1, ez+1);
				int n8 = nodeIndex(ex,   ny-ey-1, ez+1);
				int base = comp*8;
				mesh.eNodMat[base+0] = mesh.nodMapForward[n1];
				mesh.eNodMat[base+1] = mesh.nodMapForward[n2];
				mesh.eNodMat[base+2] = mesh.nodMapForward[n3];
				mesh.eNodMat[base+3] = mesh.nodMapForward[n4];
				mesh.eNodMat[base+4] = mesh.nodMapForward[n5];
				mesh.eNodMat[base+5] = mesh.nodMapForward[n6];
				mesh.eNodMat[base+6] = mesh.nodMapForward[n7];
				mesh.eNodMat[base+7] = mesh.nodMapForward[n8];
			}
		}
	}

	// in CreateVoxelFEAmodel_Cuboid, after eNodMat is built:
	mesh.eDofMat.resize(mesh.numElements * 24);
	for (int e=0; e<mesh.numElements; ++e) {
		for (int j=0; j<8; ++j) {
			int n = mesh.eNodMat[e*8 + j];
			mesh.eDofMat[e*24 + 3*j + 0] = 3*n + 0;
			mesh.eDofMat[e*24 + 3*j + 1] = 3*n + 1;
			mesh.eDofMat[e*24 + 3*j + 2] = 3*n + 2;
		}
	}

	// Reorder elements into 64-bit Morton order (based on structured (ex,ey,ez))
	{
		struct ElemMorton { std::uint64_t key; int oldIndex; int color; };
		std::vector<ElemMorton> elems(static_cast<size_t>(mesh.numElements));
		for (int comp = 0; comp < mesh.numElements; ++comp) {
			int fullIdx = mesh.eleMapBack[static_cast<size_t>(comp)];
			int ex, ey, ez;
			elem_ijk_from_full(fullIdx, nx, ny, ex, ey, ez);
			std::uint64_t code = morton3D_64(static_cast<std::uint32_t>(ex),
			                                 static_cast<std::uint32_t>(ey),
			                                 static_cast<std::uint32_t>(ez));
			int color = (ex & 1) | ((ey & 1) << 1) | ((ez & 1) << 2); // 0..7
			elems[static_cast<size_t>(comp)] = {code, comp, color};
		}
		std::sort(elems.begin(), elems.end(),
			[](const ElemMorton& a, const ElemMorton& b){ return a.key < b.key; });
		std::vector<int> elemPerm(static_cast<size_t>(mesh.numElements));
		std::vector<int> elemColorPerm(static_cast<size_t>(mesh.numElements));
		for (int newIdx = 0; newIdx < mesh.numElements; ++newIdx) {
			elemPerm[static_cast<size_t>(newIdx)]      = elems[static_cast<size_t>(newIdx)].oldIndex;
			elemColorPerm[static_cast<size_t>(newIdx)] = elems[static_cast<size_t>(newIdx)].color;
		}
		// Apply permutation to per-element data
		permute_strided(mesh.eNodMat, elemPerm, 8);
		permute_strided(mesh.eDofMat, elemPerm, 24);
		permute_unstrided(mesh.eleMapBack, elemPerm);
		// Recompute eleMapForward to stay consistent
		mesh.eleMapForward.assign(nx*ny*nz, -1);
		for (int i=0;i<mesh.numElements;i++) mesh.eleMapForward[ mesh.eleMapBack[static_cast<size_t>(i)] ] = i;
		// Build color buckets directly (8-color parity scheme)
		const int numColors = 8;
		mesh.coloring.numColors = numColors;
		mesh.coloring.elemColor = std::move(elemColorPerm);
		mesh.coloring.colorBuckets.assign(static_cast<size_t>(numColors), {});
		for (int e = 0; e < mesh.numElements; ++e) {
			const int c = mesh.coloring.elemColor[static_cast<size_t>(e)];
			mesh.coloring.colorBuckets[static_cast<size_t>(c)].push_back(e);
		}
	}

	// Boundary nodes/elements identification
	std::vector<int32_t> nodDegree(mesh.numNodes, 0);
	for (int e=0;e<mesh.numElements;e++) {
		for (int j=0;j<8;j++) nodDegree[ mesh.eNodMat[e*8+j] ] += 1;
	}
	mesh.nodesOnBoundary.clear();
	for (int i=0;i<mesh.numNodes;i++) if (nodDegree[i] < 8) mesh.nodesOnBoundary.push_back(i);
	std::vector<uint8_t> isBoundaryNode(mesh.numNodes, 0);
	for (int v: mesh.nodesOnBoundary) isBoundaryNode[v]=1;
	mesh.elementsOnBoundary.clear();
	for (int e=0;e<mesh.numElements;e++) {
		bool onB=false; for (int j=0;j<8;j++) if (isBoundaryNode[mesh.eNodMat[e*8+j]]) { onB=true; break; }
		if (onB) mesh.elementsOnBoundary.push_back(e);
	}

	// Element stiffness
	mesh.Ke = ComputeVoxelKe(0.3, 1.0);

	// Structured 8-coloring built during Morton reorder (mesh.coloring filled above)

	// Initialize design variables to V0 later
	pb.density.assign(mesh.numElements, 1.0f);

	// Allocate forces and DOF state
	pb.F.ux.assign(mesh.numNodes, 0.0);
	pb.F.uy.assign(mesh.numNodes, 0.0);
	pb.F.uz.assign(mesh.numNodes, 0.0);
	pb.isFreeDOF.assign(mesh.numDOFs, 1);
	pb.freeDofIndex.clear();
}

void ApplyBoundaryConditions(Problem& pb) {
    const int ny = pb.mesh.resY;
    const int nx = pb.mesh.resX;
    const int nz = pb.mesh.resZ;
    const int nnx = nx+1, nny=ny+1, nnz=nz+1;
    const int origNx = pb.mesh.origResX; // original element count in X

    // Reset DOF masks and loads
    std::fill(pb.isFreeDOF.begin(), pb.isFreeDOF.end(), 1);
	std::fill(pb.F.ux.begin(), pb.F.ux.end(), 0.0);
	std::fill(pb.F.uy.begin(), pb.F.uy.end(), 0.0);
	std::fill(pb.F.uz.begin(), pb.F.uz.end(), 0.0);

	// Built-in demo BCs as fallback
	std::vector<int32_t> fixedNodes;
	for (int iz=0; iz<nnz; ++iz) {
		for (int iy=0; iy<nny; ++iy) {
			int node = nnx*nny*iz + nnx*iy + 0;
			fixedNodes.push_back(pb.mesh.nodMapForward[node]);
		}
	}
	for (int n : fixedNodes) {
		pb.isFreeDOF[3*n + 0] = 0;
		pb.isFreeDOF[3*n + 1] = 0;
		pb.isFreeDOF[3*n + 2] = 0;
	}

	// Apply loads on the right face of the original (pre-padding) domain: ix = origNx
	{
		std::vector<int32_t> loadedNodes;
		int count=0;
		const int ixLoad = std::min(origNx, nnx-1); // guard
		for (int iz=0; iz<=std::max(1,nz/6); ++iz) {
			for (int iy=0; iy<nny; ++iy) {
				int node = nnx*nny*iz + nnx*iy + ixLoad;
				loadedNodes.push_back(pb.mesh.nodMapForward[node]);
				++count;
			}
		}
		if (count>0) {
			double fz = -1.0/static_cast<double>(count);
			for (int n : loadedNodes) pb.F.uz[n] += fz;
		}
	}

    // Fix DOFs on nodes not connected to any element (void nodes)
    {
        std::vector<uint8_t> usedNode(pb.mesh.numNodes, 0);
        for (int e=0;e<pb.mesh.numElements;e++) {
            int base = e*8;
            for (int j=0;j<8;j++) usedNode[pb.mesh.eNodMat[base+j]] = 1;
        }
        for (int n=0;n<pb.mesh.numNodes;n++) if (!usedNode[n]) {
            pb.isFreeDOF[3*n + 0] = 0;
            pb.isFreeDOF[3*n + 1] = 0;
            pb.isFreeDOF[3*n + 2] = 0;
        }
    }

    // Build free dof index list
    pb.freeDofIndex.clear();
    for (int i=0;i<pb.mesh.numDOFs;i++) if (pb.isFreeDOF[i]) pb.freeDofIndex.push_back(i);
	// Cache per-free-DOF mapping to node index and component for fast gathers/scatters
	pb.freeNodeIndex.resize(pb.freeDofIndex.size());
	pb.freeCompIndex.resize(pb.freeDofIndex.size());
	for (size_t i=0;i<pb.freeDofIndex.size();++i) {
		const int gi = pb.freeDofIndex[i];
		pb.freeNodeIndex[i] = gi / 3;
		pb.freeCompIndex[i] = gi % 3;
	}

	// Build eDofFreeMat: mapping from element dofs to free dof indices (or -1)
	pb.eDofFreeMat.assign(pb.mesh.numElements * 24, -1);
	std::vector<int> globalToFree(pb.mesh.numDOFs, -1);
	for (size_t i=0; i<pb.freeDofIndex.size(); ++i) {
		globalToFree[pb.freeDofIndex[i]] = static_cast<int>(i);
	}
	// Parallelize if large (though this is only done once)
	#if defined(_OPENMP)
	#pragma omp parallel for
	#endif
	for (int e=0; e<pb.mesh.numElements; ++e) {
		for (int j=0; j<24; ++j) {
			int globalDof = pb.mesh.eDofMat[e*24 + j];
			pb.eDofFreeMat[e*24 + j] = globalToFree[globalDof];
		}
	}
}

// Helper to process a batch of 8 elements using SIMD
// We assume BLOCK_SIZE is 8 to map well to AVX2 (4 doubles x 2) or AVX-512 (8 doubles)
template <int BLOCK_SIZE = 8>
static inline void ProcessBlock_8(
	const int* __restrict__ eIndices, // Indices of elements to process
	int count,                        // Valid elements in this batch (<= 8)
	const int* __restrict__ eNod,
	const double (* __restrict__ K2D)[24],
	const double* __restrict__ Ee,
	const double* __restrict__ ux,
	const double* __restrict__ uy,
	const double* __restrict__ uz,
	double* __restrict__ yx,
	double* __restrict__ yy,
	double* __restrict__ yz
) {
	alignas(64) double ue[24][BLOCK_SIZE];
	alignas(64) double fe[24][BLOCK_SIZE];
	double Ee_vals[BLOCK_SIZE];

	// 1. Gather Displacements & Moduli
	for (int k = 0; k < count; ++k) {
		int e = eIndices[k];
		Ee_vals[k] = Ee[e];
		const int* __restrict__ en = eNod + 8 * e;
		
		// Unrolled gather of 8 nodes * 3 DOFs
		#pragma GCC unroll 8
		for (int a = 0; a < 8; ++a) {
			int n = en[a];
			ue[3*a + 0][k] = ux[n];
			ue[3*a + 1][k] = uy[n];
			ue[3*a + 2][k] = uz[n];
		}
	}
	// Pad remainder of the block with dummy data to satisfy fixed loop bounds for SIMD
	for (int k = count; k < BLOCK_SIZE; ++k) {
		Ee_vals[k] = 0.0;
		for(int i=0; i<24; ++i) ue[i][k] = 0.0;
	}

	// 2. Matrix-Vector Product: fe = Ke * ue
	// The compiler will vectorize the inner loop over 'k' (SIMD across elements)
	// We unroll the outer loops (i, j) to remove branching overhead
	#pragma GCC unroll 24
	for (int i = 0; i < 24; ++i) {
		// Init accumulator
		#pragma omp simd
		for (int k = 0; k < BLOCK_SIZE; ++k) fe[i][k] = 0.0;

		#pragma GCC unroll 24
		for (int j = 0; j < 24; ++j) {
			double Kij = K2D[i][j];
			// Fused Multiply-Add across the batch
			#pragma omp simd
			for (int k = 0; k < BLOCK_SIZE; ++k) {
				fe[i][k] += Kij * ue[j][k];
			}
		}
	}

	// 3. Scatter Forces
	for (int k = 0; k < count; ++k) {
		double Eval = Ee_vals[k];
		if (Eval <= 1.0e-16) continue;

		int e = eIndices[k];
		const int* __restrict__ en = eNod + 8 * e;

		#pragma GCC unroll 8
		for (int a = 0; a < 8; ++a) {
			int n = en[a];
			// Accumulate forces
			yx[n] += fe[3*a + 0][k] * Eval;
			yy[n] += fe[3*a + 1][k] * Eval;
			yz[n] += fe[3*a + 2][k] * Eval;
		}
	}
}

// Inner kernel: raw-pointer only, to minimize aliasing paranoia
static inline void K_times_u_kernel(
	int numElements,
	const int* __restrict__ eNod,
	const double* __restrict__ Kptr_raw,
	const double* __restrict__ ux,
	const double* __restrict__ uy,
	const double* __restrict__ uz,
	double* __restrict__ yx,
	double* __restrict__ yy,
	double* __restrict__ yz,
	const double* __restrict__ Ee
){
	#if defined(__GNUC__) || defined(__clang__)
	const double* __restrict__ Kptr =
		static_cast<const double*>(__builtin_assume_aligned(Kptr_raw, 64));
	#else
	const double* __restrict__ Kptr = Kptr_raw;
	#endif
	#ifndef NDEBUG
	assert_aligned_64(Kptr);
	#endif
	const double (* __restrict__ K2D)[24] =
		reinterpret_cast<const double (* __restrict__)[24]>(Kptr);

	// Batching logic for serial/contiguous case
	constexpr int BS = 8;
	int e = 0;
	int eIdx[BS];
	
	// Main blocked loop
	for (; e <= numElements - BS; e += BS) {
		for(int k=0; k<BS; ++k) eIdx[k] = e + k;
		ProcessBlock_8<BS>(eIdx, BS, eNod, K2D, Ee, ux, uy, uz, yx, yy, yz);
	}
	// Remainder
	if (e < numElements) {
		int count = 0;
		for(; e < numElements; ++e) eIdx[count++] = e;
		ProcessBlock_8<BS>(eIdx, count, eNod, K2D, Ee, ux, uy, uz, yx, yy, yz);
	}
}

// Overload using reusable view
static inline void K_times_u_kernel(
	const FEAKernelView& view,
	const double* __restrict__ ux,
	const double* __restrict__ uy,
	const double* __restrict__ uz,
	double* __restrict__ yx,
	double* __restrict__ yy,
	double* __restrict__ yz,
	const double* __restrict__ Ee
){
	K_times_u_kernel(view.numElements, view.eNod, view.Kptr, ux, uy, uz, yx, yy, yz, Ee);
}

// Compute ce[e] = u_e' * Ke * u_e and return total compliance sum_e Ee[e]*ce[e]
static inline double compute_compliance_kernel(
	int numElements,
	const int* __restrict__ eNod,
	const double* __restrict__ Kptr_raw,
	const double* __restrict__ ux,
	const double* __restrict__ uy,
	const double* __restrict__ uz,
	const double* __restrict__ Ee,
	double* __restrict__ ceOut
){
	#if defined(__GNUC__) || defined(__clang__)
	const double* __restrict__ Kptr =
		static_cast<const double*>(__builtin_assume_aligned(Kptr_raw, 64));
	#else
	const double* __restrict__ Kptr = Kptr_raw;
	#endif
	#ifndef NDEBUG
	assert_aligned_64(Kptr);
	#endif
	const double (* __restrict__ K2D)[24] =
		reinterpret_cast<const double (* __restrict__)[24]>(Kptr);
	double totalC = 0.0;
	#if defined(_OPENMP)
	#pragma omp parallel
	{
		alignas(64) double ue[24];
		double localC = 0.0;
		#pragma omp for nowait schedule(static, 32)
		for (int e = 0; e < numElements; ++e) {
			const double Ee_e = Ee[e];
			if (Ee_e <= 1.0e-16) {
				ceOut[e] = 0.0;
				continue;
			}
			// Gather element displacements
			const int* __restrict__ en = eNod + 8*e;
			for (int a = 0; a < 8; ++a) {
				const int n = en[a];
				const int base = 3*a;
				ue[base + 0] = ux[n];
				ue[base + 1] = uy[n];
				ue[base + 2] = uz[n];
			}
			// ce = ue' * Ke * ue
			double ce = 0.0;
			for (int i = 0; i < 24; ++i) {
				const double* __restrict__ Ki = K2D[i];
				double rowDot = 0.0;
				#pragma omp simd reduction(+:rowDot)
				for (int j = 0; j < 24; ++j) rowDot += Ki[j] * ue[j];
				ce += ue[i] * rowDot;
			}
			ceOut[e] = ce;
			localC  += Ee_e * ce;
		}
		#pragma omp atomic
		totalC += localC;
	}
	#else
	{
		alignas(64) double ue[24];
		for (int e = 0; e < numElements; ++e) {
			const double Ee_e = Ee[e];
			if (Ee_e <= 1.0e-16) {
				ceOut[e] = 0.0;
				continue;
			}
			// Gather element displacements
			const int* __restrict__ en = eNod + 8*e;
			for (int a = 0; a < 8; ++a) {
				const int n = en[a];
				const int base = 3*a;
				ue[base + 0] = ux[n];
				ue[base + 1] = uy[n];
				ue[base + 2] = uz[n];
			}
			// ce = ue' * Ke * ue
			double ce = 0.0;
			for (int i = 0; i < 24; ++i) {
				const double* __restrict__ Ki = K2D[i];
				double rowDot = 0.0;
				for (int j = 0; j < 24; ++j) rowDot += Ki[j] * ue[j];
				ce += ue[i] * rowDot;
			}
			ceOut[e] = ce;
			totalC   += Ee_e * ce;
		}
	}
	#endif
	return totalC;
}

// Overload using reusable view
static inline double compute_compliance_kernel(
	const FEAKernelView& view,
	const double* __restrict__ ux,
	const double* __restrict__ uy,
	const double* __restrict__ uz,
	const double* __restrict__ Ee,
	double* __restrict__ ceOut
){
	return compute_compliance_kernel(view.numElements, view.eNod, view.Kptr, ux, uy, uz, Ee, ceOut);
}

// Helper to process a batch of 8 elements using SIMD directly on free DOFs
template <int BLOCK_SIZE = 8>
static inline void ProcessBlock_8_Free(
	const int* __restrict__ eIndices,
	int count,
	const int32_t* __restrict__ eDofFree,
	const double (* __restrict__ K2D)[24],
	const double* __restrict__ Ee,
	const double* __restrict__ uFree,
	double* __restrict__ yFree
) {
	alignas(64) double ue[24][BLOCK_SIZE];
	alignas(64) double fe[24][BLOCK_SIZE];
	double Ee_vals[BLOCK_SIZE];

	// 1. Gather Displacements & Moduli
	for (int k = 0; k < count; ++k) {
		int e = eIndices[k];
		Ee_vals[k] = Ee[e];
		const int32_t* __restrict__ ed = eDofFree + 24 * e;
		
		// Unrolled gather of 24 DOFs
		#pragma GCC unroll 24
		for (int a = 0; a < 24; ++a) {
			int idx = ed[a];
			if (idx >= 0) ue[a][k] = uFree[idx];
            else ue[a][k] = 0.0;
		}
	}
	// Pad remainder
	for (int k = count; k < BLOCK_SIZE; ++k) {
		Ee_vals[k] = 0.0;
		for(int i=0; i<24; ++i) ue[i][k] = 0.0;
	}

	// 2. Matrix-Vector Product: fe = Ke * ue
	#pragma GCC unroll 24
	for (int i = 0; i < 24; ++i) {
		#pragma omp simd
		for (int k = 0; k < BLOCK_SIZE; ++k) fe[i][k] = 0.0;
		#pragma GCC unroll 24
		for (int j = 0; j < 24; ++j) {
			double Kij = K2D[i][j];
			#pragma omp simd
			for (int k = 0; k < BLOCK_SIZE; ++k) {
				fe[i][k] += Kij * ue[j][k];
			}
		}
	}

	// 3. Scatter Forces
	for (int k = 0; k < count; ++k) {
		double Eval = Ee_vals[k];
		if (Eval <= 1.0e-16) continue;

		int e = eIndices[k];
		const int32_t* __restrict__ ed = eDofFree + 24 * e;

		#pragma GCC unroll 24
		for (int a = 0; a < 24; ++a) {
			int idx = ed[a];
			if (idx >= 0) {
				yFree[idx] += fe[a][k] * Eval;
			}
		}
	}
}

// MATLAB-style batched K*u implementation
// Step 1: Gather all element displacements into uMat (numElements x 24)
// Step 2: Batch matrix multiply: fMat = uMat * Ke^T, then scale by Ee
// Step 3: Scatter using precomputed sorted indices (no atomics)
void K_times_u_finest(const Problem& pb,
                      const std::vector<double>& eleModulus,
                      const std::vector<double>& uFree,
                      std::vector<double>& yFree)
{
	const auto& mesh = pb.mesh;
	const double* __restrict__ Kptr = mesh.Ke.data();
	const int numElements = mesh.numElements;
	const size_t numFreeDofs = uFree.size();

	// Zero output
	if (yFree.size() != numFreeDofs) yFree.assign(numFreeDofs, 0.0);
	else std::fill(yFree.begin(), yFree.end(), 0.0);

	const int32_t* __restrict__ eDofFree = pb.eDofFreeMat.data();
	const double* __restrict__ uF = uFree.data();
	double* __restrict__ yF_out = yFree.data();

	std::vector<double> uMat(numElements * 24);
	std::vector<double> fMat(numElements * 24);
	
	// Serial implementation with single BLAS call
	for (int e = 0; e < numElements; ++e) {
		const int32_t* ed = eDofFree + 24 * e;
		double* uRow = uMat.data() + e * 24;
		for (int a = 0; a < 24; ++a) {
			int idx = ed[a];
			uRow[a] = (idx >= 0) ? uF[idx] : 0.0;
		}
	}
#ifdef HAVE_CBLAS
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
	            numElements, 24, 24,
	            1.0, uMat.data(), 24, Kptr, 24, 0.0, fMat.data(), 24);
#else
	for (int e = 0; e < numElements; ++e) {
		const double* uRow = uMat.data() + e * 24;
		double* fRow = fMat.data() + e * 24;
		for (int i = 0; i < 24; ++i) {
			const double* KeRow = Kptr + i * 24;
			double sum = 0.0;
			for (int j = 0; j < 24; ++j) {
				sum += uRow[j] * KeRow[j];
			}
			fRow[i] = sum;
		}
	}
#endif
	for (int e = 0; e < numElements; ++e) {
		const double Eval = eleModulus[e];
		if (Eval > 1.0e-16) {
			for (int a = 0; a < 24; ++a) {
				int idx = eDofFree[e*24+a];
				if (idx >= 0) {
					yF_out[idx] += fMat[e*24+a] * Eval;
				}
			}
		}
	}
}

double ComputeCompliance(const Problem& pb,
						   const std::vector<double>& eleModulus,
						   const DOFData& U,
						   std::vector<double>& ceList) {
	const auto& mesh = pb.mesh;
	const int numElements = mesh.numElements;
	ceList.assign(numElements, 0.0);
	const double* __restrict__ ux   = U.ux.data();
	const double* __restrict__ uy   = U.uy.data();
	const double* __restrict__ uz   = U.uz.data();
	const double* __restrict__ Ee   = eleModulus.data();
	const FEAKernelView view = make_fea_kernel_view(pb);
	return compute_compliance_kernel(view, ux, uy, uz, Ee, ceList.data());
}

} // namespace top3d
