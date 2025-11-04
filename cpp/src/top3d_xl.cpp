#include "top3d_xl.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <string>
#include "io_export.hpp"
#include "voxel_surface.hpp"

namespace top3d {

static inline int idx_dof(int nodeIdx, int comp) { return 3*nodeIdx + comp; }

// forward declarations for exports
static void export_volume_nifti(const Problem& pb, const std::vector<double>& xPhys, const std::string& path);
static void export_surface_stl(const Problem& pb, const std::vector<double>& xPhys, const std::string& path, float iso);

void InitialSettings(GlobalParams& out) {
	out = GlobalParams{};
}

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

static void build_cuboid_voxel(bool value, int nely, int nelx, int nelz, std::vector<uint8_t>& vox) {
	vox.assign(nelx*nely*nelz, static_cast<uint8_t>(value?1:0));
}

void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz) {
	CartesianMesh& mesh = pb.mesh;
	mesh.resX = nelx;
	mesh.resY = nely;
	mesh.resZ = nelz;
	mesh.eleSize = {1.0,1.0,1.0};

	std::vector<uint8_t> voxelized;
	build_cuboid_voxel(true, nely, nelx, nelz, voxelized);

	const int nx = nelx;
	const int ny = nely;
	const int nz = nelz;

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
	mesh.eleMapForward.assign(nx*ny*nz, 0);
	for (int i=0;i<mesh.numElements;i++) mesh.eleMapForward[ mesh.eleMapBack[i] ] = i+1;

	// Node numbering on full grid
	const int nnx = nx+1, nny = ny+1, nnz = nz+1;
	mesh.numNodes = nnx*nny*nnz;
	mesh.numDOFs = mesh.numNodes*3;

	mesh.nodMapBack.resize(mesh.numNodes);
	std::iota(mesh.nodMapBack.begin(), mesh.nodMapBack.end(), 1);
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
				if (comp==0) continue;
				int eComp = comp-1;
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
				int base = eComp*8;
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

	// Fix x=0 face nodes in all three directions
	std::vector<int> fixedNodes;
	for (int iz=0; iz<nnz; ++iz) {
		for (int iy=0; iy<nny; ++iy) {
			int node = nnx*nny*iz + nnx*iy + 0;
			fixedNodes.push_back(node);
		}
	}
	for (int n : fixedNodes) {
		pb.isFreeDOF[idx_dof(n,0)] = 0;
		pb.isFreeDOF[idx_dof(n,1)] = 0;
		pb.isFreeDOF[idx_dof(n,2)] = 0;
	}

	// Apply a distributed -Z load on x=nx face, lower half of z as in demo
	std::vector<int> loadedNodes;
	int count=0;
	for (int iz=0; iz<=std::max(1,nz/6); ++iz) {
		for (int iy=0; iy<nny; ++iy) {
			int node = nnx*nny*iz + nnx*iy + nx;
			loadedNodes.push_back(node);
			++count;
		}
	}
	if (count>0) {
		double fz = -1.0/static_cast<double>(count);
		for (int n : loadedNodes) pb.F[idx_dof(n,2)] += fz;
	}

	// Build free dof index list
	pb.freeDofIndex.clear();
	for (int i=0;i<pb.mesh.numDOFs;i++) if (pb.isFreeDOF[i]) pb.freeDofIndex.push_back(i);
}

void K_times_u_finest(const Problem& pb, const std::vector<double>& eleModulus,
					   const std::vector<double>& uFull, std::vector<double>& yFull) {
	const auto& mesh = pb.mesh;
	yFull.assign(mesh.numDOFs, 0.0);
	const auto& Ke = mesh.Ke;

	std::array<double,24> ue{};
	std::array<double,24> fe{};
	for (int e=0; e<mesh.numElements; ++e) {
		// Gather element DOFs (8 nodes x 3 comps)
		for (int j=0;j<8;j++) {
			int n = mesh.eNodMat[e*8 + j];
			ue[3*j+0] = uFull[idx_dof(n,0)];
			ue[3*j+1] = uFull[idx_dof(n,1)];
			ue[3*j+2] = uFull[idx_dof(n,2)];
		}
		// fe = (Ee*Ke) * ue
		std::fill(fe.begin(), fe.end(), 0.0);
		double Ee = eleModulus[e];
		for (int i=0;i<24;i++) {
			double sum=0.0;
			const double* Ki = &Ke[i*24];
			for (int j=0;j<24;j++) sum += Ki[j]*ue[j];
			fe[i] = Ee * sum;
		}
		// Scatter to global yFull
		for (int j=0;j<8;j++) {
			int n = mesh.eNodMat[e*8 + j];
			yFull[idx_dof(n,0)] += fe[3*j+0];
			yFull[idx_dof(n,1)] += fe[3*j+1];
			yFull[idx_dof(n,2)] += fe[3*j+2];
		}
	}
	// Impose Dirichlet by zeroing rows (simple but adequate for operator on free subspace wrapper)
	for (int i=0;i<mesh.numDOFs;i++) if (!pb.isFreeDOF[i]) yFull[i] = 0.0;
}

static void restrict_to_free(const Problem& pb, const std::vector<double>& full, std::vector<double>& freev) {
	freev.resize(pb.freeDofIndex.size());
	for (size_t i=0;i<pb.freeDofIndex.size();++i) freev[i] = full[pb.freeDofIndex[i]];
}

static void scatter_from_free(const Problem& pb, const std::vector<double>& freev, std::vector<double>& full) {
	for (size_t i=0;i<pb.freeDofIndex.size();++i) full[pb.freeDofIndex[i]] = freev[i];
}

int PCG_free(const Problem& pb,
			  const std::vector<double>& eleModulus,
			  const std::vector<double>& bFree,
			  std::vector<double>& xFree,
			  double tol, int maxIt) {
	// Preconditioner: simple Jacobi (identity here, upgraded later with MG)
	std::vector<double> r = bFree;
	std::vector<double> z(r.size(), 0.0), p(r.size(), 0.0), Ap(r.size(), 0.0);
	// Initial xFree maintained
	// r = b - A*x
	if (!xFree.empty()) {
		std::vector<double> xfull(pb.mesh.numDOFs, 0.0), yfull;
		scatter_from_free(pb, xFree, xfull);
		K_times_u_finest(pb, eleModulus, xfull, yfull);
		std::vector<double> yfree; restrict_to_free(pb, yfull, yfree);
		for (size_t i=0;i<r.size();++i) r[i] -= yfree[i];
	}
	// Jacobi as 1.0 (scaling) for now
	for (size_t i=0;i<r.size();++i) z[i] = r[i];
	double rz_old = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
	if (xFree.empty()) xFree.assign(r.size(), 0.0);
	int it=0;
	const double normb = std::sqrt(std::inner_product(bFree.begin(), bFree.end(), bFree.begin(), 0.0));
	while (it < maxIt) {
		if (it==0) p = z; else {
			double beta = rz_old / std::max(1e-30, std::inner_product(r.begin(), r.end(), z.begin(), 0.0));
			for (size_t i=0;i<p.size();++i) p[i] = z[i] + beta * p[i];
		}
		// Ap = A*p
		std::vector<double> pfull(pb.mesh.numDOFs, 0.0), Apfull;
		scatter_from_free(pb, p, pfull);
		K_times_u_finest(pb, eleModulus, pfull, Apfull);
		std::vector<double> Apfree; restrict_to_free(pb, Apfull, Apfree);
		Ap.swap(Apfree);
		double alpha = rz_old / std::max(1e-30, std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0));
		for (size_t i=0;i<xFree.size();++i) xFree[i] += alpha * p[i];
		for (size_t i=0;i<r.size();++i) r[i] -= alpha * Ap[i];
		double res = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0)) / std::max(1e-30, normb);
		if (res < tol) return it+1;
		// z = M^{-1} r, Jacobi ~ identity
		for (size_t i=0;i<z.size();++i) z[i] = r[i];
		double rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
		rz_old = rz_new;
		++it;
	}
	return it;
}

// Minimal hook to later enable MG (no-op here to keep builds simple)
void EnableMultigridPreconditioner(const Problem& /*pb*/, const MGPrecondConfig& /*cfg*/) {}

double ComputeCompliance(const Problem& pb,
						   const std::vector<double>& eleModulus,
						   const std::vector<double>& uFull,
						   std::vector<double>& ceList) {
	const auto& mesh = pb.mesh;
	const auto& Ke = mesh.Ke;
	ceList.assign(mesh.numElements, 0.0);
	std::array<double,24> ue{};
	for (int e=0;e<mesh.numElements;e++) {
		for (int j=0;j<8;j++) {
			int n = mesh.eNodMat[e*8 + j];
			ue[3*j+0] = uFull[idx_dof(n,0)];
			ue[3*j+1] = uFull[idx_dof(n,1)];
			ue[3*j+2] = uFull[idx_dof(n,2)];
		}
		double tmp[24];
		for (int i=0;i<24;i++) {
			double sum=0.0; const double* Ki = &Ke[i*24];
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

// ===== PDE Filter =====
PDEFilter SetupPDEFilter(const Problem& pb, double filterRadius) {
	PDEFilter pf;
	// 2x2x2 Gaussian points in natural coords and shape derivatives
	const double s[8] = {-1,1,1,-1,-1,1,1,-1};
	const double t[8] = {-1,-1,1,1,-1,-1,1,1};
	const double p[8] = {-1,-1,-1,-1,1,1,1,1};
	const double w[8] = {1,1,1,1,1,1,1,1};
	// Trilinear shape N (8) and dN/ds, dN/dt, dN/dp
	double N[8][8];
	double dS[3*8][8];
	for (int gp=0; gp<8; ++gp) {
		double sg = s[gp]/std::sqrt(3.0), tg = t[gp]/std::sqrt(3.0), pg = p[gp]/std::sqrt(3.0);
		double Ns[8];
		Ns[0]=0.125*(1-sg)*(1-tg)*(1-pg); Ns[1]=0.125*(1+sg)*(1-tg)*(1-pg);
		Ns[2]=0.125*(1+sg)*(1+tg)*(1-pg); Ns[3]=0.125*(1-sg)*(1+tg)*(1-pg);
		Ns[4]=0.125*(1-sg)*(1-tg)*(1+pg); Ns[5]=0.125*(1+sg)*(1-tg)*(1+pg);
		Ns[6]=0.125*(1+sg)*(1+tg)*(1+pg); Ns[7]=0.125*(1-sg)*(1+tg)*(1+pg);
		for (int a=0;a<8;a++) N[gp][a]=Ns[a];
		double dNds[8] = {
			-0.125*(1-tg)*(1-pg), 0.125*(1-tg)*(1-pg), 0.125*(1+tg)*(1-pg), -0.125*(1+tg)*(1-pg),
			-0.125*(1-tg)*(1+pg), 0.125*(1-tg)*(1+pg), 0.125*(1+tg)*(1+pg), -0.125*(1+tg)*(1+pg)
		};
		double dNdt[8] = {
			-0.125*(1-sg)*(1-pg), -0.125*(1+sg)*(1-pg), 0.125*(1+sg)*(1-pg), 0.125*(1-sg)*(1-pg),
			-0.125*(1-sg)*(1+pg), -0.125*(1+sg)*(1+pg), 0.125*(1+sg)*(1+pg), 0.125*(1-sg)*(1+pg)
		};
		double dNdp[8] = {
			-0.125*(1-sg)*(1-tg), -0.125*(1+sg)*(1-tg), -0.125*(1+sg)*(1+tg), -0.125*(1-sg)*(1+tg),
			 0.125*(1-sg)*(1-tg),  0.125*(1+sg)*(1-tg),  0.125*(1+sg)*(1+tg),  0.125*(1-sg)*(1+tg)
		};
		for (int a=0;a<8;a++) { dS[3*gp+0][a]=dNds[a]; dS[3*gp+1][a]=dNdt[a]; dS[3*gp+2][a]=dNdp[a]; }
	}
	// Cell size assumed 1, detJ=1/8, wgt = 1/8 per gp
	double KEF0[8*8]={0}; // dShape' * dShape
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) {
		double sum=0.0; for (int k=0;k<3*8;k++) sum += dS[k][i]*dS[k][j];
		KEF0[i*8+j] = sum;
	}
	double KEF1[8*8]={0}; // N' * diag(wgt) * N; wgt=1/8
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) {
		double sum=0.0; for (int gp=0;gp<8;gp++) sum += N[gp][i]*(1.0/8.0)*N[gp][j];
		KEF1[i*8+j] = sum;
	}
	// iRmin in MATLAB: (filterRadius * eleSize(1))/2/sqrt(3)
	double iRmin = (filterRadius * pb.mesh.eleSize[0]) / (2.0*std::sqrt(3.0));
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) pf.kernel[i*8+j] = iRmin*iRmin*KEF0[i*8+j] + KEF1[i*8+j];
	// Diagonal preconditioner by accumulating kernel contributions to nodes
	pf.diagPrecondNode.assign(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base = e*8;
		for (int a=0;a<8;a++) {
			int na = pb.mesh.eNodMat[base+a];
			double sum=0.0; for (int b=0;b<8;b++) sum += pf.kernel[a*8+b];
			pf.diagPrecondNode[na] += sum;
		}
	}
	for (double& v : pf.diagPrecondNode) v = v>0 ? 1.0/v : 1.0;
	return pf;
}

static void MatTimesVec_PDE(const Problem& pb, const PDEFilter& pf, const std::vector<double>& xNode, std::vector<double>& yNode) {
	yNode.assign(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base = e*8;
		double u[8]; for (int a=0;a<8;a++) u[a] = xNode[pb.mesh.eNodMat[base+a]];
		double f[8]={0};
		for (int i=0;i<8;i++) {
			double sum=0.0; for (int j=0;j<8;j++) sum += pf.kernel[i*8+j]*u[j];
			f[i]=sum;
		}
		for (int a=0;a<8;a++) yNode[pb.mesh.eNodMat[base+a]] += f[a];
	}
}

void ApplyPDEFilter(const Problem& pb, const PDEFilter& pf, const std::vector<double>& srcEle, std::vector<double>& dstEle) {
	// Ele -> Node (sum/8)
	std::vector<double> rhs(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		double val = srcEle[e] * (1.0/8.0);
		int base = e*8; for (int a=0;a<8;a++) rhs[pb.mesh.eNodMat[base+a]] += val;
	}
	// Solve (PDE kernel) * x = rhs with PCG and Jacobi precond
	std::vector<double> x(pb.mesh.numNodes, 0.0);
	std::vector<double> r = rhs, z(rhs.size()), pvec(rhs.size()), Ap(rhs.size());
	for (size_t i=0;i<r.size();++i) z[i] = pf.diagPrecondNode[i]*r[i];
	double rz = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
	if (rz==0) rz=1.0; pvec = z;
	const double tol = 1e-6;
	const int maxIt = 400;
	for (int it=0; it<maxIt; ++it) {
		MatTimesVec_PDE(pb, pf, pvec, Ap);
		double denom = std::inner_product(pvec.begin(), pvec.end(), Ap.begin(), 0.0);
		double alpha = rz / std::max(1e-30, denom);
		for (size_t i=0;i<x.size();++i) x[i] += alpha * pvec[i];
		for (size_t i=0;i<r.size();++i) r[i] -= alpha * Ap[i];
		double rn = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0));
		if (rn < tol) break;
		for (size_t i=0;i<z.size();++i) z[i] = pf.diagPrecondNode[i]*r[i];
		double rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
		double beta = rz_new / std::max(1e-30, rz);
		for (size_t i=0;i<pvec.size();++i) pvec[i] = z[i] + beta * pvec[i];
		rz = rz_new;
	}
	// Node -> Ele (sum/8)
	dstEle.assign(pb.mesh.numElements, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base=e*8; double sum=0.0; for (int a=0;a<8;a++) sum += x[pb.mesh.eNodMat[base+a]];
		dstEle[e] = sum*(1.0/8.0);
	}
}

void TOP3D_XL_GLOBAL(int nely, int nelx, int nelz, double V0, int nLoop, double rMin) {
	Problem pb;
	InitialSettings(pb.params);
	CreateVoxelFEAmodel_Cuboid(pb, nely, nelx, nelz);
	ApplyBoundaryConditions(pb);
	PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
	// Initialize design
	for (double& x: pb.density) x = V0;

	const int ne = pb.mesh.numElements;
	std::vector<double> x = pb.density;
	std::vector<double> xPhys = x;
	std::vector<double> ce(ne, 0.0);
	std::vector<double> eleMod(ne, pb.params.youngsModulus);

	// Solve fully solid for reference
	{
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		std::vector<double> uFree; uFree.assign(bFree.size(), 0.0);
		PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt);
		scatter_from_free(pb, uFree, U);
		double Csolid = ComputeCompliance(pb, eleMod, U, ce);
		std::cout << "Compliance of Fully Solid Domain: " << Csolid << "\n";
	}

	int loop=0;
	double change=1.0;
	while (loop < nLoop && change > 1e-4) {
		// Update modulus via SIMP
		for (int e=0;e<ne;e++) {
			double rho = std::clamp(xPhys[e], 0.0, 1.0);
			eleMod[e] = pb.params.youngsModulusMin + std::pow(rho, pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		}
		// Solve KU=F
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		std::vector<double> uFree(bFree.size(), 0.0);
		PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt);
		scatter_from_free(pb, uFree, U);
		// Compliance and sensitivities
		double C = ComputeCompliance(pb, eleMod, U, ce);
		std::vector<double> dc(ne, 0.0);
		for (int e=0;e<ne;e++) {
			double rho = std::clamp(xPhys[e], 0.0, 1.0);
			double dEdrho = pb.params.simpPenalty * std::pow(rho, pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
			dc[e] = - dEdrho * ce[e];
		}
		// PDE filter on dc (ft=1): filter(x.*dc)./max(1e-3,x)
		{
			std::vector<double> xdc(ne);
			for (int e=0;e<ne;e++) xdc[e] = x[e]*dc[e];
			std::vector<double> dc_filt; ApplyPDEFilter(pb, pfFilter, xdc, dc_filt);
			for (int e=0;e<ne;e++) dc[e] = dc_filt[e] / std::max(1e-3, x[e]);
		}
		// OC update with move limits
		double l1=0.0, l2=1e9;
		double move=0.2;
		std::vector<double> xnew(ne, 0.0);
		while ((l2-l1)/(l1+l2) > 1e-6) {
			double lmid = 0.5*(l1+l2);
			for (int e=0;e<ne;e++) {
				double val = std::sqrt(std::max(1e-30, -dc[e]/lmid));
				double xe = std::clamp(x[e]*val, x[e]-move, x[e]+move);
				xnew[e] = std::clamp(xe, 0.0, 1.0);
			}
			double vol = std::accumulate(xnew.begin(), xnew.end(), 0.0) / static_cast<double>(ne);
			if (vol - V0 > 0) l1 = lmid; else l2 = lmid;
		}
		change = 0.0; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
		x.swap(xnew);
		xPhys = x; // no filter in this minimal port
		++loop;
		std::cout << " It.:" << loop << " Obj.:" << C << " Vol.:" << (std::accumulate(xPhys.begin(), xPhys.end(), 0.0)/ne)
				  << " Ch.:" << change << "\n";
	}
	// Exports
	export_volume_nifti(pb, xPhys, "DesignVolume.nii");
	export_surface_stl(pb, xPhys, "DesignVolume.stl", 0.3f);
	std::cout << "Done.\n";
}

// ===== LOCAL (PIO) =====
void TOP3D_XL_LOCAL(int nely, int nelx, int nelz, double Ve0, int nLoop, double rMin, double rHat) {
	Problem pb; InitialSettings(pb.params);
	CreateVoxelFEAmodel_Cuboid(pb, nely, nelx, nelz);
	ApplyBoundaryConditions(pb);
	const int ne = pb.mesh.numElements;
	std::vector<double> x(ne, Ve0), xTilde(ne, Ve0), xPhys(ne, Ve0);
	std::vector<double> eleMod(ne, pb.params.youngsModulus);
	PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
	PDEFilter pfLVF = SetupPDEFilter(pb, rHat);
	double beta = 1.0, eta = 0.5; int p = 16, pMax = 128; int loopbeta=0;
	for (int it=0; it<nLoop; ++it) {
		loopbeta++;
		// SIMP modulus
		for (int e=0;e<ne;e++) {
			eleMod[e] = pb.params.youngsModulusMin + std::pow(xPhys[e], pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		}
		// Solve KU=F
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		std::vector<double> uFree(bFree.size(), 0.0);
		PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt);
		scatter_from_free(pb, uFree, U);
		// Compliance sensitivities dc = -dE/drho * ce_norm (we use ce raw here as solid ref omitted)
		std::vector<double> ce; double C = ComputeCompliance(pb, eleMod, U, ce);
		std::vector<double> dc(ne);
		for (int e=0;e<ne;e++) {
			double dE = pb.params.simpPenalty * std::pow(std::max(1e-9,xPhys[e]), pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
			dc[e] = - dE * ce[e];
		}
		// Local Volume Fraction via PDE filter
		std::vector<double> xHat; ApplyPDEFilter(pb, pfLVF, xPhys, xHat);
		double accum=0.0; for (int e=0;e<ne;e++) accum += std::pow(xHat[e], p);
		double f = std::pow(accum / ne, 1.0/p) - 1.0; // <= 0
		// df/dx via chain: df/dxHat * dxHat/dxPhys (apply filters twice like MATLAB); approximate df/dxPhys by filtering gradient once
		std::vector<double> dfdx_hat(ne);
		double coeff = std::pow(accum/ne, 1.0/p - 1.0) / ne;
		for (int e=0;e<ne;e++) dfdx_hat[e] = coeff * p * std::pow(std::max(1e-9,xHat[e]), p-1);
		std::vector<double> dfdx_phys; ApplyPDEFilter(pb, pfLVF, dfdx_hat, dfdx_phys);
		// Heaviside projection and its derivative wrt xTilde
		auto H = [&](double v){ return (std::tanh(beta*eta) + std::tanh(beta*(v-eta))) / (std::tanh(beta*eta) + std::tanh(beta*(1-eta))); };
		auto dH = [&](double v){ double th = std::tanh(beta*(v-eta)); double den = (std::tanh(beta*eta) + std::tanh(beta*(1-eta))); return beta*(1-th*th)/den; };
		std::vector<double> dx(ne); for (int e=0;e<ne;e++) dx[e] = dH(xTilde[e]);
		// Filter dc and dfdx to design space via PDE filter and chain with dH
		std::vector<double> dc_filt; ApplyPDEFilter(pb, pfFilter, std::vector<double>(dc.begin(), dc.end()), dc_filt);
		for (int e=0;e<ne;e++) dc_filt[e] *= dx[e];
		std::vector<double> dfdx_filt; ApplyPDEFilter(pb, pfFilter, dfdx_phys, dfdx_filt);
		for (int e=0;e<ne;e++) dfdx_filt[e] *= dx[e];
		// MMA-like OC update with one constraint: minimize c s.t. g=f<=0 and 0<=x<=1
		double l1=0.0, l2=1e9; double move=0.1; std::vector<double> xnew(ne);
		while ((l2-l1)/(l1+l2) > 1e-6) {
			double lm = 0.5*(l1+l2);
			double volg=0.0;
			for (int e=0;e<ne;e++) {
				double step = -dc_filt[e] / std::max(1e-16, lm*dfdx_filt[e]);
				double xe = std::clamp(x[e]*std::sqrt(std::max(0.1, step)), x[e]-move, x[e]+move);
				xnew[e] = std::clamp(xe, 0.0, 1.0);
				volg += dfdx_filt[e] * (xnew[e]-x[e]);
			}
			if (f + volg > 0) l1 = lm; else l2 = lm;
		}
		double change=0.0; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
		x.swap(xnew);
		// Update xTilde and xPhys
		ApplyPDEFilter(pb, pfFilter, x, xTilde);
		for (int e=0;e<ne;e++) xPhys[e] = H(xTilde[e]);
		if (beta < pMax && loopbeta >= 40) { beta *= 2.0; loopbeta = 0; }
		std::cout << " It.:" << (it+1) << " Obj.:" << C << " Cons.:" << f << "\n";
		if (change < 1e-4) break;
	}
}

// ===== Export helpers =====
static void export_volume_nifti(const Problem& pb, const std::vector<double>& xPhys, const std::string& path) {
	int ny = pb.mesh.resY, nx = pb.mesh.resX, nz = pb.mesh.resZ;
	std::vector<float> vol(ny*nx*nz, 0.0f);
	for (int e=0; e<pb.mesh.numElements; ++e) vol[pb.mesh.eleMapBack[e]] = static_cast<float>(xPhys[e]);
	ioexp::write_nifti_float32(path, nx, ny, nz, (float)pb.mesh.eleSize[0], (float)pb.mesh.eleSize[1], (float)pb.mesh.eleSize[2], vol);
}

static void export_surface_stl(const Problem& pb, const std::vector<double>& xPhys, const std::string& path, float iso=0.5f) {
	int ny = pb.mesh.resY, nx = pb.mesh.resX, nz = pb.mesh.resZ;
	std::vector<float> vol(ny*nx*nz, 0.0f);
	for (int e=0; e<pb.mesh.numElements; ++e) vol[pb.mesh.eleMapBack[e]] = static_cast<float>(xPhys[e]);
	std::vector<std::array<float,3>> verts; std::vector<std::array<uint32_t,3>> faces;
	voxsurf::extract_faces(vol, ny, nx, nz, iso, verts, faces);
	ioexp::write_stl_binary(path, verts, faces);
}

} // namespace top3d


