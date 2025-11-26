#include "core.hpp"
#include "multigrid/padding.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cassert>

namespace top3d {

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

	// Build Morton permutation (64-bit codes) and nodal maps
	{
		// Sanity for 64-bit Morton: max coordinate per axis < 2^21
#ifndef NDEBUG
		assert((nnx-1) < (1<<21));
		assert((nny-1) < (1<<21));
		assert((nnz-1) < (1<<21));
#endif
		struct MortonNode { std::uint64_t morton; int oldIndex; };
		std::vector<MortonNode> nodes;
		nodes.reserve(static_cast<size_t>(mesh.numNodes));
		for (int old = 0; old < mesh.numNodes; ++old) {
			int ix, iy, iz;
			node_ijk(old, nnx, nny, ix, iy, iz);
			std::uint64_t code = morton3D_64(static_cast<std::uint32_t>(ix),
			                                 static_cast<std::uint32_t>(iy),
			                                 static_cast<std::uint32_t>(iz));
			nodes.push_back({code, old});
		}
		std::sort(nodes.begin(), nodes.end(),
			[](const MortonNode& a, const MortonNode& b){ return a.morton < b.morton; });
		mesh.nodMapForward.resize(mesh.numNodes);
		mesh.nodMapBack.resize(mesh.numNodes);
		for (int newIdx = 0; newIdx < mesh.numNodes; ++newIdx) {
			int oldIdx = nodes[static_cast<size_t>(newIdx)].oldIndex;
			mesh.nodMapForward[static_cast<size_t>(oldIdx)] = newIdx;
			mesh.nodMapBack[static_cast<size_t>(newIdx)] = oldIdx;
		}
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
		struct ElemMorton { std::uint64_t key; int oldIndex; };
		std::vector<ElemMorton> elems(static_cast<size_t>(mesh.numElements));
		for (int comp = 0; comp < mesh.numElements; ++comp) {
			int fullIdx = mesh.eleMapBack[static_cast<size_t>(comp)];
			int ex, ey, ez;
			elem_ijk_from_full(fullIdx, nx, ny, ex, ey, ez);
			std::uint64_t code = morton3D_64(static_cast<std::uint32_t>(ex),
			                                 static_cast<std::uint32_t>(ey),
			                                 static_cast<std::uint32_t>(ez));
			elems[static_cast<size_t>(comp)] = {code, comp};
		}
		std::sort(elems.begin(), elems.end(),
			[](const ElemMorton& a, const ElemMorton& b){ return a.key < b.key; });
		std::vector<int> elemPerm(static_cast<size_t>(mesh.numElements));
		for (int newIdx = 0; newIdx < mesh.numElements; ++newIdx) elemPerm[static_cast<size_t>(newIdx)] = elems[static_cast<size_t>(newIdx)].oldIndex;
		// Apply permutation to per-element data
		permute_strided(mesh.eNodMat, elemPerm, 8);
		permute_strided(mesh.eDofMat, elemPerm, 24);
		permute_unstrided(mesh.eleMapBack, elemPerm);
		// Recompute eleMapForward to stay consistent
		mesh.eleMapForward.assign(nx*ny*nz, -1);
		for (int i=0;i<mesh.numElements;i++) mesh.eleMapForward[ mesh.eleMapBack[static_cast<size_t>(i)] ] = i;
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
}

void K_times_u_finest(const Problem& pb,
                      const std::vector<double>& eleModulus,
                      const DOFData& U,
                      DOFData& Y) {
    const auto& mesh = pb.mesh; 
    const auto& Ke   = mesh.Ke; 

    if ((int)Y.ux.size() != mesh.numNodes) Y.ux.assign(mesh.numNodes, 0.0); else std::fill(Y.ux.begin(), Y.ux.end(), 0.0);
    if ((int)Y.uy.size() != mesh.numNodes) Y.uy.assign(mesh.numNodes, 0.0); else std::fill(Y.uy.begin(), Y.uy.end(), 0.0);
    if ((int)Y.uz.size() != mesh.numNodes) Y.uz.assign(mesh.numNodes, 0.0); else std::fill(Y.uz.begin(), Y.uz.end(), 0.0);

    alignas(64) std::array<double,24> ue{};
	alignas(64) std::array<double,24> fe{};

    for (int e=0; e<mesh.numElements; ++e) {
        const double Ee = eleModulus[e];
        if (Ee <= 1.0e-16) continue;

		// Gather ue branchlessly using lookup tables
		for (int i=0;i<24;i++) {
			const int a = node_of_i_[i];
			const int c = comp_of_i_[i];
			const int n = mesh.eNodMat[e*8 + a];
			ue[i] = (c==0 ? U.ux[n] : (c==1 ? U.uy[n] : U.uz[n]));
		}

        // fe = Ee * Ke * ue
        for (int i=0;i<24;i++) {
            const double* __restrict__ Ki = &Ke[i*24];
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int j=0;j<24;j++) sum += Ki[j]*ue[j];
			fe[i] = Ee * sum;
        }
		// Scatter once per local DOF slot, branchlessly
		for (int i=0;i<24;i++) {
			const int a = node_of_i_[i];
			const int c = comp_of_i_[i];
			const int n = mesh.eNodMat[e*8 + a];
			const double val = fe[i];
			if (c==0) Y.ux[n] += val;
			else if (c==1) Y.uy[n] += val;
			else Y.uz[n] += val;
		}
    }
}

double ComputeCompliance(const Problem& pb,
						   const std::vector<double>& eleModulus,
						   const DOFData& U,
						   std::vector<double>& ceList) {
	const auto& mesh = pb.mesh;
	const auto& Ke = mesh.Ke;
	ceList.assign(mesh.numElements, 0.0);
	std::array<double,24> ue{};
	for (int e=0;e<mesh.numElements;e++) {
		// gather element DOFs directly
		for (int a=0;a<8;a++) {
			int n = mesh.eNodMat[e*8 + a];
			ue[3*a+0] = U.ux[n];
			ue[3*a+1] = U.uy[n];
			ue[3*a+2] = U.uz[n];
		}
		double tmp[24];
		for (int i=0;i<24;i++) {
			double sum=0.0; const double* Ki = &Ke[i*24];
			#pragma omp simd reduction(+:sum)
			for (int j=0;j<24;j++) sum += Ki[j]*ue[j];
			tmp[i] = sum;
		}
		double val=0.0; for (int i=0;i<24;i++) val += ue[i]*tmp[i];
		ceList[e] = val;
	}
	// Total compliance = sum(Ee * ce)
	double C=0.0; for (int e=0;e<mesh.numElements;e++) C += eleModulus[e]*ceList[e];
	return C;
}

} // namespace top3d
