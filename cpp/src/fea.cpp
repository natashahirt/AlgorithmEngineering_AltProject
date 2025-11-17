#include "core.hpp"
#include "multigrid/padding.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>

namespace top3d {

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
		double DB[6][24]; for (int i=0;i<6;i++) for (int j=0;j<24;j++) { double sum=0; for (int k=0;k<6;k++) sum+=D[i][k]*B[k][j]; DB[i][j]=sum; }
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
	mesh.eleSize = {1.0,1.0,1.0};
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

	mesh.nodMapBack.resize(mesh.numNodes);
	std::iota(mesh.nodMapBack.begin(), mesh.nodMapBack.end(), 0);
	mesh.nodMapForward.resize(mesh.numNodes);
	for (int i=0;i<mesh.numNodes;i++) mesh.nodMapForward[i] = i;

	// eNodMat (compact by using compact node ids directly)
	mesh.eNodMat.resize(mesh.numElements*8);
	int eIdx = 0;
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
				mesh.eNodMat[base+0] = n1;
				mesh.eNodMat[base+1] = n2;
				mesh.eNodMat[base+2] = n3;
				mesh.eNodMat[base+3] = n4;
				mesh.eNodMat[base+4] = n5;
				mesh.eNodMat[base+5] = n6;
				mesh.eNodMat[base+6] = n7;
				mesh.eNodMat[base+7] = n8;
				++eIdx;
			}
		}
	}

	// eDofMat (flattened DOF indices per element: 24 per element)
	mesh.eDofMat.resize(mesh.numElements * 24);
	for (int e=0; e<mesh.numElements; ++e) {
		int nb = e*8;
		int db = e*24;
		for (int j=0;j<8;j++) {
			int n = mesh.eNodMat[nb + j];
			int d = 3*n;
			mesh.eDofMat[db + 3*j + 0] = d + 0;
			mesh.eDofMat[db + 3*j + 1] = d + 1;
			mesh.eDofMat[db + 3*j + 2] = d + 2;
		}
	}

	// Boundary nodes/elements identification
	std::vector<int> nodDegree(mesh.numNodes, 0);
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
	pb.density.assign(mesh.numElements, 1.0);

	// Allocate forces and DOF state
	pb.F.assign(mesh.numDOFs, 0.0);
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
    std::fill(pb.F.begin(), pb.F.end(), 0.0);

	// Built-in demo BCs as fallback
	std::vector<int> fixedNodes;
	for (int iz=0; iz<nnz; ++iz) {
		for (int iy=0; iy<nny; ++iy) {
			int node = nnx*nny*iz + nnx*iy + 0;
			fixedNodes.push_back(node);
		}
	}
	for (int n : fixedNodes) {
		pb.isFreeDOF[3*n + 0] = 0;
		pb.isFreeDOF[3*n + 1] = 0;
		pb.isFreeDOF[3*n + 2] = 0;
	}

	// Apply loads on the right face of the original (pre-padding) domain: ix = origNx
	{
		std::vector<int> loadedNodes;
		int count=0;
		const int ixLoad = std::min(origNx, nnx-1); // guard
		for (int iz=0; iz<=std::max(1,nz/6); ++iz) {
			for (int iy=0; iy<nny; ++iy) {
				int node = nnx*nny*iz + nnx*iy + ixLoad;
				loadedNodes.push_back(node);
				++count;
			}
		}
		if (count>0) {
			double fz = -1.0/static_cast<double>(count);
			for (int n : loadedNodes) pb.F[3*n + 2] += fz;
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
}

void K_times_u_finest(const Problem& pb, const std::vector<double>& eleModulus,
					   const std::vector<double>& uFull, std::vector<double>& yFull) {
	// get constants from the problem statement
	const auto& mesh = pb.mesh; 
	const auto& Ke = mesh.Ke; 
	// prep yFull vector (output of K * u)
	if ((int)yFull.size() == mesh.numDOFs) {
		std::fill(yFull.begin(), yFull.end(), 0.0);
	} else {
		yFull.assign(mesh.numDOFs, 0.0);
	}
	// element-local displacement buffer
	alignas(64) std::array<double,24> ue{};

	// generate pointers for u, y, and elements
	const double* __restrict__ uPtr = uFull.data();
	double* __restrict__ yPtr = yFull.data();
	const int32_t* __restrict__ eDof = mesh.eDofMat.data();

	for (int e=0; e<mesh.numElements; ++e) {
		const double Ee = eleModulus[e];
		if (Ee <= 1.0e-16) continue;
		// gather element DOFs (8 nodes x 3 comps) from uFull global displacement vector
		{
			const int32_t* __restrict__ dptr = &eDof[e*24];
			for (int j=0;j<8;j++) {
				ue[3*j+0] = uPtr[dptr[3*j+0]];
				ue[3*j+1] = uPtr[dptr[3*j+1]];
				ue[3*j+2] = uPtr[dptr[3*j+2]];
			}
		}
		// direct scatter: y[dof_i] += Ee * (Ke[i,:] · ue)
		{
			const int32_t* __restrict__ dptr = &eDof[e*24];
			for (int i=0;i<24;i++) {
				const double* __restrict__ Ki = &Ke[i*24];
				double sum = 0.0;
				#pragma omp simd reduction(+:sum)
				for (int j=0;j<24;j++) sum += Ki[j]*ue[j];
				yPtr[dptr[i]] += Ee * sum;
			}
		}
	}
	// No Dirichlet row zeroing (callers on free subspace don't require)
}

double ComputeCompliance(const Problem& pb,
						   const std::vector<double>& eleModulus,
						   const std::vector<double>& uFull,
						   std::vector<double>& ceList) {
	const auto& mesh = pb.mesh;
	const auto& Ke = mesh.Ke;
	ceList.assign(mesh.numElements, 0.0);
	std::array<double,24> ue{};
	for (int e=0;e<mesh.numElements;e++) {
		{
			const int32_t* __restrict__ dptr = &mesh.eDofMat[e*24];
			for (int j=0;j<8;j++) {
				ue[3*j+0] = uFull[dptr[3*j+0]];
				ue[3*j+1] = uFull[dptr[3*j+1]];
				ue[3*j+2] = uFull[dptr[3*j+2]];
			}
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
