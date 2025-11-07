#include "top3d_xl.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <filesystem>
#include <functional>
#include <string>
#include <fstream>
#include <stdexcept>
#include "io_export.hpp"
#include "voxel_surface.hpp"
#include "opt/MMAseq.hpp"
#include "multigrid_padding.hpp"

namespace top3d {

static inline int idx_dof(int nodeIdx, int comp) { return 3*nodeIdx + comp; }

// forward declarations for exports
static void export_surface_stl(const Problem& pb, const std::vector<double>& xPhys, const std::string& path, float iso);

// output directory helpers
static inline void ensure_out_dir(const std::string& outDir) {
	std::error_code ec; (void)std::filesystem::create_directories(outDir, ec);
}

static inline std::string out_dir_for_cwd() {
	namespace fs = std::filesystem;
	std::error_code ec;
	fs::path cwd = fs::current_path(ec);
	if (ec) return std::string("./out/");
	fs::path tryLocal = cwd / "out";
	if (fs::exists(tryLocal, ec)) return tryLocal.string() + "/";
	fs::path tryParent = cwd.parent_path() / "out";
	if (fs::exists(tryParent, ec)) return tryParent.string() + "/";
	return (cwd / "out").string() + "/";
}

static inline std::string out_stl_dir_for_cwd() {
	return out_dir_for_cwd() + "stl/";
}

static inline std::string generate_unique_filename(const std::string& mode) {
	// Generate datetime string matching bash format: YYYYMMDD_HHMMSS
	auto now = std::chrono::system_clock::now();
	auto time_t = std::chrono::system_clock::to_time_t(now);
	std::tm* tm_buf = std::localtime(&time_t);
	
	std::ostringstream oss;
	oss << std::setfill('0') << std::setw(4) << (tm_buf->tm_year + 1900)
		<< std::setw(2) << (tm_buf->tm_mon + 1)
		<< std::setw(2) << tm_buf->tm_mday << "_"
		<< std::setw(2) << tm_buf->tm_hour
		<< std::setw(2) << tm_buf->tm_min
		<< std::setw(2) << tm_buf->tm_sec;
	std::string datetime = oss.str();
	
	// Get SLURM_JOB_ID from environment if available
	const char* job_id = std::getenv("SLURM_JOB_ID");
	std::string job_id_str = job_id ? std::string(job_id) : "";
	
	// Format: MODE_DATETIME_JOBID.stl (or MODE_DATETIME.stl if no job ID)
	if (!job_id_str.empty()) {
		return mode + "_" + datetime + "_" + job_id_str + ".stl";
	} else {
		return mode + "_" + datetime + ".stl";
	}
}

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
	mesh.eleSize = {1.0,1.0,1.0};

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

// moved to topvoxel.cpp

void ApplyBoundaryConditions(Problem& pb) {
    const int ny = pb.mesh.resY;
    const int nx = pb.mesh.resX;
    const int nz = pb.mesh.resZ;
    const int nnx = nx+1, nny=ny+1, nnz=nz+1;

    // Reset DOF masks and loads
    std::fill(pb.isFreeDOF.begin(), pb.isFreeDOF.end(), 1);
    std::fill(pb.F.begin(), pb.F.end(), 0.0);

    // Prefer external BC/loads if present
    bool usedExternal = false;
    if (pb.extBC || pb.extLoads) {
        usedExternal = true;
        if (pb.extBC) {
            for (auto f : pb.extBC->fixations) {
                int full = f[0]-1; if (full < 0) continue;
                if (full >= (int)pb.mesh.nodMapForward.size()) continue;
                int n = pb.mesh.nodMapForward[full]; if (n < 0 || n >= pb.mesh.numNodes) continue;
                if (f[1]) pb.isFreeDOF[idx_dof(n,0)] = 0;
                if (f[2]) pb.isFreeDOF[idx_dof(n,1)] = 0;
                if (f[3]) pb.isFreeDOF[idx_dof(n,2)] = 0;
            }
        }
        if (pb.extLoads && !pb.extLoads->cases.empty()) {
            const auto& lc = pb.extLoads->cases.front();
            for (auto rec : lc) {
                int full = int(rec[0]) - 1; if (full < 0) continue;
                if (full >= (int)pb.mesh.nodMapForward.size()) continue;
                int n = pb.mesh.nodMapForward[full]; if (n < 0 || n>=pb.mesh.numNodes) continue;
                pb.F[idx_dof(n,0)] += rec[1];
                pb.F[idx_dof(n,1)] += rec[2];
                pb.F[idx_dof(n,2)] += rec[3];
            }
        }
    }

    if (!usedExternal) {
        // Built-in demo BCs as fallback
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
    }

    // Fix DOFs on nodes not connected to any element (void nodes)
    {
        std::vector<uint8_t> usedNode(pb.mesh.numNodes, 0);
        for (int e=0;e<pb.mesh.numElements;e++) {
            int base = e*8;
            for (int j=0;j<8;j++) usedNode[pb.mesh.eNodMat[base+j]] = 1;
        }
        for (int n=0;n<pb.mesh.numNodes;n++) if (!usedNode[n]) {
            pb.isFreeDOF[idx_dof(n,0)] = 0;
            pb.isFreeDOF[idx_dof(n,1)] = 0;
            pb.isFreeDOF[idx_dof(n,2)] = 0;
        }
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
			  double tol, int maxIt,
			  Preconditioner M) {
    std::vector<double> r = bFree;
    std::vector<double> z(r.size(), 0.0), p(r.size(), 0.0), Ap(r.size(), 0.0);
    // r = b - A*x (if nonzero initial guess)
    if (!xFree.empty()) {
        std::vector<double> xfull(pb.mesh.numDOFs, 0.0), yfull;
        scatter_from_free(pb, xFree, xfull);
        K_times_u_finest(pb, eleModulus, xfull, yfull);
        std::vector<double> yfree; restrict_to_free(pb, yfull, yfree);
        for (size_t i=0;i<r.size();++i) r[i] -= yfree[i];
    }
    if (xFree.empty()) xFree.assign(r.size(), 0.0);
    // z0 and p0
    if (M) M(r, z); else z = r;
    double rz_old = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
    p = z;

    const double normb = std::sqrt(std::inner_product(bFree.begin(), bFree.end(), bFree.begin(), 0.0));
    for (int it=0; it<maxIt; ++it) {
        // Ap = A*p
        std::vector<double> pfull(pb.mesh.numDOFs, 0.0), Apfull;
        scatter_from_free(pb, p, pfull);
        K_times_u_finest(pb, eleModulus, pfull, Apfull);
        std::vector<double> Apfree; restrict_to_free(pb, Apfull, Apfree);
        Ap.swap(Apfree);

        double denom = std::max(1e-30, std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0));
        double alpha = rz_old / denom;

        for (size_t i=0;i<xFree.size();++i) xFree[i] += alpha * p[i];
        for (size_t i=0;i<r.size();++i)     r[i]     -= alpha * Ap[i];

        double res = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0)) / std::max(1e-30, normb);
        if (res < tol) return it+1;

        // z_{k+1}
        if (M) M(r, z); else z = r;
        double rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);

        double beta = rz_new / std::max(1e-30, rz_old);
        for (size_t i=0;i<p.size();++i) p[i] = z[i] + beta * p[i];
        rz_old = rz_new;
    }
    return maxIt;
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
	// initialize warm-start buffers
	pf.lastXNode.assign(pb.mesh.numNodes, 0.0);
	pf.lastRhsNode.assign(pb.mesh.numNodes, 0.0);
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

void ApplyPDEFilter(const Problem& pb, PDEFilter& pf, const std::vector<double>& srcEle, std::vector<double>& dstEle) {
	// Ele -> Node (sum/8)
	std::vector<double> rhs(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		double val = srcEle[e] * (1.0/8.0);
		int base = e*8; for (int a=0;a<8;a++) rhs[pb.mesh.eNodMat[base+a]] += val;
	}
	// Solve (PDE kernel) * x = rhs with PCG and Jacobi precond
	// Conditional warm-start: reuse lastX if rhs hasn't changed much
	const double relThresh = 0.1; // relative RHS change threshold for warm-start
	if ((int)pf.lastXNode.size() != pb.mesh.numNodes) pf.lastXNode.assign(pb.mesh.numNodes, 0.0);
	if ((int)pf.lastRhsNode.size() != pb.mesh.numNodes) pf.lastRhsNode.assign(pb.mesh.numNodes, 0.0);
	double normRhs=0.0, diff=0.0;
	for (size_t i=0;i<rhs.size();++i) { normRhs += rhs[i]*rhs[i]; double d = rhs[i]-pf.lastRhsNode[i]; diff += d*d; }
	normRhs = std::sqrt(normRhs); diff = std::sqrt(diff);
	bool useWarm = (normRhs > 0.0) && (diff / normRhs < relThresh);
	std::vector<double> x = useWarm ? pf.lastXNode : std::vector<double>(pb.mesh.numNodes, 0.0);

	std::vector<double> r(rhs.size()), z(rhs.size()), pvec(rhs.size()), Ap(rhs.size());
	if (useWarm) {
		std::vector<double> y0; MatTimesVec_PDE(pb, pf, x, y0);
		for (size_t i=0;i<r.size();++i) r[i] = rhs[i] - y0[i];
	} else {
		r = rhs;
	}
	for (size_t i=0;i<r.size();++i) z[i] = pf.diagPrecondNode[i]*r[i];
	double rz = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
	if (rz==0) rz=1.0; pvec = z;
	const double tol = 1e-6;
	const int maxIt = 400;
	// Early exit if warm-start already good
	{
		double rn = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0));
		if (rn < tol) {
			// Node -> Ele (sum/8)
			dstEle.assign(pb.mesh.numElements, 0.0);
			for (int e=0;e<pb.mesh.numElements;e++) {
				int base=e*8; double sum=0.0; for (int a=0;a<8;a++) sum += x[pb.mesh.eNodMat[base+a]];
				dstEle[e] = sum*(1.0/8.0);
			}
			pf.lastXNode = x;
			pf.lastRhsNode = rhs;
			return;
		}
	}
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
	// Save warm-start buffers
	pf.lastXNode = x;
	pf.lastRhsNode = rhs;
}

// ===== MG scaffolding: trilinear weights and hierarchy (Step 2) =====

// Trilinear shape for 8-node hex at natural coords (xi, eta, zeta) in [-1,1]
static inline void shape_trilinear8(double xi, double eta, double zeta, double N[8]) {
	N[0] = 0.125*(1-xi)*(1-eta)*(1-zeta);
	N[1] = 0.125*(1+xi)*(1-eta)*(1-zeta);
	N[2] = 0.125*(1+xi)*(1+eta)*(1-zeta);
	N[3] = 0.125*(1-xi)*(1+eta)*(1-zeta);
	N[4] = 0.125*(1-xi)*(1-eta)*(1+zeta);
	N[5] = 0.125*(1+xi)*(1-eta)*(1+zeta);
	N[6] = 0.125*(1+xi)*(1+eta)*(1+zeta);
	N[7] = 0.125*(1-xi)*(1+eta)*(1+zeta);
}

// Build per-element nodal weights table for a given span (2 or 4)
// Mapping: coarse element's 8 vertices -> (span+1)^3 embedded fine-grid vertices
static std::vector<double> make_trilinear_weights_table(int span) {
	const int grid = span + 1;
	std::vector<double> W(grid*grid*grid*8, 0.0);
	for (int iz=0; iz<=span; ++iz) {
		for (int iy=0; iy<=span; ++iy) {
			for (int ix=0; ix<=span; ++ix) {
				double xi   = -1.0 + 2.0 * (double(ix) / double(span));
				double eta  = -1.0 + 2.0 * (double(iy) / double(span));
				double zeta = -1.0 + 2.0 * (double(iz) / double(span));
				double N[8]; shape_trilinear8(xi, eta, zeta, N);
				const int v = (iz*grid + iy)*grid + ix;
				for (int a=0; a<8; ++a) W[v*8 + a] = N[a];
			}
		}
	}
	return W;
}

// Build a structured MG level of resolution (nx,ny,nz)
static MGLevel build_structured_level(int nx, int ny, int nz, int span) {
	MGLevel L;
	L.resX = nx; L.resY = ny; L.resZ = nz;
	L.spanWidth = span;
	L.numElements = nx*ny*nz;
	const int nnx = nx+1, nny = ny+1, nnz = nz+1;
	L.numNodes = nnx*nny*nnz;
	L.numDOFs = L.numNodes * 3;

	L.nodMapBack.resize(L.numNodes);
	std::iota(L.nodMapBack.begin(), L.nodMapBack.end(), 0);
	L.nodMapForward = L.nodMapBack;

	L.eNodMat.resize(L.numElements*8);
	auto nodeIndex = [&](int ix,int iy,int iz){ return (nnx*nny*iz + nnx*iy + ix); };
	int eComp=0;
	for (int ez=0; ez<nz; ++ez) {
		for (int ex=0; ex<nx; ++ex) {
			for (int ey=0; ey<ny; ++ey) {
				int n1 = nodeIndex(ex,   ny-ey,   ez);
				int n2 = nodeIndex(ex+1, ny-ey,   ez);
				int n3 = nodeIndex(ex+1, ny-ey-1, ez);
				int n4 = nodeIndex(ex,   ny-ey-1, ez);
				int n5 = nodeIndex(ex,   ny-ey,   ez+1);
				int n6 = nodeIndex(ex+1, ny-ey,   ez+1);
				int n7 = nodeIndex(ex+1, ny-ey-1, ez+1);
				int n8 = nodeIndex(ex,   ny-ey-1, ez+1);
				int base = eComp*8;
				L.eNodMat[base+0]=n1; L.eNodMat[base+1]=n2; L.eNodMat[base+2]=n3; L.eNodMat[base+3]=n4;
				L.eNodMat[base+4]=n5; L.eNodMat[base+5]=n6; L.eNodMat[base+6]=n7; L.eNodMat[base+7]=n8;
				++eComp;
			}
		}
	}
	L.weightsNode = make_trilinear_weights_table(span);
	return L;
}

// Heuristic: adapt MG level count to keep coarsest DOFs <= NlimitDofs and stop when tiny
static int ComputeAdaptiveMaxLevels(const Problem& pb, bool nonDyadic, int cap, int NlimitDofs) {
	int nx = pb.mesh.resX;
	int ny = pb.mesh.resY;
	int nz = pb.mesh.resZ;
	int levels = 1; // include finest

	for (int li=1; li<cap; ++li) {
		int span = (li==1 && nonDyadic ? 4 : 2);
		// Require exact divisibility (to match structured transfers)
		if (nx % span != 0 || ny % span != 0 || nz % span != 0) break;

		int nnx = nx / span, nny = ny / span, nnz = nz / span;
		if (nnx < 1 || nny < 1 || nnz < 1) break;

		long long nodes = 1LL*(nnx+1)*(nny+1)*(nnz+1);
		long long dofs  = 3LL * nodes;
		// Stop before exceeding coarsest dense factorization limit
		if (dofs > NlimitDofs) break;

		// Accept this level
		nx = nnx; ny = nny; nz = nnz;
		levels++;

		// Mirror MATLAB's guard: stop when very small
		if (std::min({nx,ny,nz}) <= 2) break;
	}
	return levels;
}

void BuildMGHierarchy(const Problem& pb, bool nonDyadic, MGHierarchy& H, int maxLevels) {
	H.levels.clear();
	H.nonDyadic = nonDyadic;

	// Finest level: same resolution; span not used here
	{
		MGLevel L0 = build_structured_level(pb.mesh.resX, pb.mesh.resY, pb.mesh.resZ, /*span=*/1);
		L0.weightsNode.clear();
		H.levels.push_back(std::move(L0));
	}

	int nx = pb.mesh.resX;
	int ny = pb.mesh.resY;
	int nz = pb.mesh.resZ;

    for (int li=1; li<maxLevels; ++li) {
        int span = (li==1 && nonDyadic ? 4 : 2);
        // Enforce divisibility to avoid misalignment; stop if indivisible
        if (nx % span != 0 || ny % span != 0 || nz % span != 0) break;
        if (nx/span < 1 || ny/span < 1 || nz/span < 1) break;
        nx = nx / span; ny = ny / span; nz = nz / span;
        H.levels.push_back(build_structured_level(nx, ny, nz, span));
        if (std::min({nx,ny,nz}) <= 2) break;
    }
}

// ===== MG diagonal-only V-cycle (Adapted) =====

// Forward declarations for transfer operators used below
static void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
                                  const std::vector<double>& xc, std::vector<double>& xf);
static void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
                                   const std::vector<double>& rf, std::vector<double>& rc);

// Build per-level fixed-DOF masks by restricting the finest mask
static void MG_BuildFixedMasks(const Problem& pb, const MGHierarchy& H,
							   std::vector<std::vector<uint8_t>>& fixedMasks) {
	fixedMasks.resize(H.levels.size());
	// Finest-level mask from pb.isFreeDOF (3 DOFs per node)
	{
		const int n0 = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		fixedMasks[0].assign(3*n0, 0);
		for (int n=0;n<pb.mesh.numNodes;n++) {
			for (int c=0;c<3;c++) fixedMasks[0][3*n+c] = pb.isFreeDOF[3*n+c] ? 0 : 1;
		}
	}
	// Restrict masks to coarser levels (component-wise), threshold 0.5
	for (size_t l=0; l+1<H.levels.size(); ++l) {
		const int fnn = H.levels[l].numNodes;
		const int cnn = H.levels[l+1].numNodes;
		fixedMasks[l+1].assign(3*cnn, 0);
		for (int c=0;c<3;c++) {
			std::vector<double> rf(fnn), rc;
			for (int n=0;n<fnn;n++) rf[n] = fixedMasks[l][3*n+c] ? 1.0 : 0.0;
			MG_Restrict_nodes(H.levels[l+1], H.levels[l], rf, rc);
			for (int n=0;n<cnn;n++) fixedMasks[l+1][3*n+c] = (rc[n] > 0.5) ? 1 : 0;
		}
	}
}

static void ComputeJacobiDiagonalFull(const Problem& pb,
									  const std::vector<double>& eleModulus,
									  std::vector<double>& diagFull) {
	const auto& mesh = pb.mesh;
	diagFull.assign(mesh.numDOFs, 0.0);
	const auto& Ke = mesh.Ke;
	for (int e=0; e<mesh.numElements; ++e) {
		double Ee = eleModulus[e];
		for (int a=0; a<8; ++a) {
			int n = mesh.eNodMat[e*8 + a];
			for (int c=0; c<3; ++c) {
				int local = 3*a + c;
				diagFull[3*n + c] += Ke[local*24 + local] * Ee;
			}
		}
	}
	for (double& v : diagFull) if (!(v > 0.0)) v = 1.0;
}

static void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
								  const std::vector<double>& xc, std::vector<double>& xf) {
	const int fnnx = Lf.resX+1, fnny = Lf.resY+1, fnnz = Lf.resZ+1;
	const int cnnx = Lc.resX+1, cnny = Lc.resY+1, cnnz = Lc.resZ+1;
const int span = Lc.spanWidth;
	const int grid = span+1;

	xf.assign(fnnx*fnny*fnnz, 0.0);
	std::vector<double> wsum(xf.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return fnnx*fnny*iz + fnnx*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return cnnx*cnny*iz + cnnx*iy + ix; };

	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
				int c1 = idxC(cex,   Lc.resY-cey,   cez);
				int c2 = idxC(cex+1, Lc.resY-cey,   cez);
				int c3 = idxC(cex+1, Lc.resY-cey-1, cez);
				int c4 = idxC(cex,   Lc.resY-cey-1, cez);
				int c5 = idxC(cex,   Lc.resY-cey,   cez+1);
				int c6 = idxC(cex+1, Lc.resY-cey,   cez+1);
				int c7 = idxC(cex+1, Lc.resY-cey-1, cez+1);
				int c8 = idxC(cex,   Lc.resY-cey-1, cez+1);
				double cv[8] = {xc[c1],xc[c2],xc[c3],xc[c4],xc[c5],xc[c6],xc[c7],xc[c8]};

                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
				for (int iz=0; iz<=span; ++iz) {
					for (int iy=0; iy<=span; ++iy) {
						for (int ix=0; ix<=span; ++ix) {
							int fxi = fx0 + ix;
							int fyi = fy0 - iy;
							int fzi = fz0 + iz;
							int fidx = idxF(fxi, fyi, fzi);
const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
							double sum = 0.0; for (int a=0;a<8;a++) sum += W[a]*cv[a];
							xf[fidx] += sum; wsum[fidx] += 1.0;
						}
					}
				}
			}
		}
	}
	for (size_t i=0;i<xf.size();++i) if (wsum[i]>0) xf[i] /= wsum[i];
}

static void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
								 const std::vector<double>& rf, std::vector<double>& rc) {
	const int fnnx = Lf.resX+1, fnny = Lf.resY+1, fnnz = Lf.resZ+1;
	const int cnnx = Lc.resX+1, cnny = Lc.resY+1, cnnz = Lc.resZ+1;
const int span = Lc.spanWidth;
	const int grid = span+1;

	rc.assign(cnnx*cnny*cnnz, 0.0);
	std::vector<double> wsum(rc.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return fnnx*fnny*iz + fnnx*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return cnnx*cnny*iz + cnnx*iy + ix; };

	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
				int cidx[8] = {
					idxC(cex,   Lc.resY-cey,   cez),
					idxC(cex+1, Lc.resY-cey,   cez),
					idxC(cex+1, Lc.resY-cey-1, cez),
					idxC(cex,   Lc.resY-cey-1, cez),
					idxC(cex,   Lc.resY-cey,   cez+1),
					idxC(cex+1, Lc.resY-cey,   cez+1),
					idxC(cex+1, Lc.resY-cey-1, cez+1),
					idxC(cex,   Lc.resY-cey-1, cez+1)
				};

                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
				for (int iz=0; iz<=span; ++iz) {
					for (int iy=0; iy<=span; ++iy) {
						for (int ix=0; ix<=span; ++ix) {
							int fxi = fx0 + ix;
							int fyi = fy0 - iy;
							int fzi = fz0 + iz;
							int fidx = idxF(fxi, fyi, fzi);
const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
							double val = rf[fidx];
							for (int a=0;a<8;a++) { rc[cidx[a]] += W[a]*val; wsum[cidx[a]] += W[a]; }
						}
					}
				}
			}
		}
	}
	for (size_t i=0;i<rc.size();++i) if (wsum[i]>0) rc[i] /= wsum[i];
}

static void MG_CoarsenDiagonal(const MGLevel& Lc, const MGLevel& Lf,
								  const std::vector<double>& diagFine,
								  const std::vector<uint8_t>& fineFixedMask,
								  const std::vector<uint8_t>& coarseFixedMask,
								  std::vector<double>& diagCoarse) {
    const int span = Lc.spanWidth;
    const int grid = span+1;

	diagCoarse.assign((Lc.resX+1)*(Lc.resY+1)*(Lc.resZ+1)*3, 0.0);
	std::vector<double> wsum(diagCoarse.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return (Lf.resX+1)*(Lf.resY+1)*iz + (Lf.resX+1)*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return (Lc.resX+1)*(Lc.resY+1)*iz + (Lc.resX+1)*iy + ix; };

	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
				for (int iz=0; iz<=span; ++iz) {
					for (int iy=0; iy<=span; ++iy) {
						for (int ix=0; ix<=span; ++ix) {
							int fxi = fx0 + ix; int fyi = fy0 - iy; int fzi = fz0 + iz;
							int fn = idxF(fxi, fyi, fzi);
                            const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
							int cidx[8] = {
								idxC(cex,   Lc.resY-cey,   cez),
								idxC(cex+1, Lc.resY-cey,   cez),
								idxC(cex+1, Lc.resY-cey-1, cez),
								idxC(cex,   Lc.resY-cey-1, cez),
								idxC(cex,   Lc.resY-cey,   cez+1),
								idxC(cex+1, Lc.resY-cey,   cez+1),
								idxC(cex+1, Lc.resY-cey-1, cez+1),
								idxC(cex,   Lc.resY-cey-1, cez+1)
							};
							for (int a=0;a<8;a++) {
								double w2 = W[a]*W[a];
								for (int c=0;c<3;c++) {
									const int fineD = 3*fn + c;
									const int coarseD = 3*cidx[a] + c;
									if (coarseFixedMask[coarseD]) {
										// We'll set fixed coarse diagonals later; skip accumulation
										continue;
									}
									if (fineFixedMask[fineD]) {
										// Skip fixed fine contributions to avoid inflating neighboring coarse diagonals
										continue;
									}
									diagCoarse[coarseD] += w2 * diagFine[fineD];
									wsum[coarseD] += w2;
								}
							}
						}
					}
				}
			}
		}
	}
	for (size_t i=0;i<diagCoarse.size();++i) {
		if (coarseFixedMask[i]) {
			diagCoarse[i] = 1.0;
		} else if (wsum[i]>0) {
			diagCoarse[i] /= wsum[i];
			if (!(diagCoarse[i] > 0.0)) diagCoarse[i] = 1.0;
		} else {
			diagCoarse[i] = 1.0;
		}
	}
}

static void MG_BuildDiagonals(const Problem& pb, const MGHierarchy& H,
							   const std::vector<std::vector<uint8_t>>& fixedMasks,
							   const std::vector<double>& eleModulus,
							   std::vector<std::vector<double>>& diagLevels) {
	diagLevels.resize(H.levels.size());
	ComputeJacobiDiagonalFull(pb, eleModulus, diagLevels[0]);
	// Impose BCs at finest level on the diagonal
	{
		const auto& mask0 = fixedMasks[0];
		for (size_t d=0; d<diagLevels[0].size() && d<mask0.size(); ++d) {
			if (mask0[d]) diagLevels[0][d] = 1.0;
		}
	}
	for (size_t l=0; l+1<H.levels.size(); ++l) {
		MG_CoarsenDiagonal(H.levels[l+1], H.levels[l],
			diagLevels[l],
			fixedMasks[l],
			fixedMasks[l+1],
			diagLevels[l+1]);
	}
}

// ===== Coarsest-level assembly and Cholesky (to mirror MATLAB direct solve) =====

static bool chol_spd_inplace(std::vector<double>& A, int N) {
	for (int i=0;i<N;i++) {
		for (int j=0;j<=i;j++) {
			double sum = A[i*N + j];
			for (int k=0;k<j;k++) sum -= A[i*N + k]*A[j*N + k];
			if (i==j) {
				if (sum <= 0.0) return false;
				A[i*N + j] = std::sqrt(sum);
			} else {
				A[i*N + j] = sum / A[j*N + j];
			}
		}
		for (int j=i+1;j<N;j++) A[i*N + j] = 0.0;
	}
	return true;
}

static void chol_solve_lower(const std::vector<double>& L,
							  const std::vector<double>& b,
							  std::vector<double>& x, int N) {
	std::vector<double> y(N, 0.0);
	for (int i=0;i<N;i++) {
		double sum = b[i];
		for (int k=0;k<i;k++) sum -= L[i*N+k]*y[k];
		y[i] = sum / L[i*N+i];
	}
	x.assign(N, 0.0);
	for (int i=N-1;i>=0;i--) {
		double sum = y[i];
		for (int k=i+1;k<N;k++) sum -= L[k*N+i]*x[k];
		x[i] = sum / L[i*N+i];
	}
}

// ===== Static MG context and Galerkin coarsest assembly =====
// Build coarsest-level K via P^T * K_fine * P with fine-level BC masking

// Build H and fixed masks once, reuse later
static void MG_BuildStaticOnce(const Problem& pb, const MGPrecondConfig& cfg,
								  MGHierarchy& H, std::vector<std::vector<uint8_t>>& fixedMasks) {
	H.levels.clear();
	const int NlimitDofs = 200000;
	int adaptiveMax = ComputeAdaptiveMaxLevels(pb, cfg.nonDyadic, cfg.maxLevels, NlimitDofs);
	BuildMGHierarchy(pb, cfg.nonDyadic, H, adaptiveMax);
	// One-line debug print (enable by setting env TOP3D_MG_DEBUG to any value)
	if (std::getenv("TOP3D_MG_DEBUG")) {
		const auto& Lc = H.levels.back();
		std::cout << "[MG] levels=" << H.levels.size()
				  << " coarsest=" << Lc.resX << "x" << Lc.resY << "x" << Lc.resZ
				  << " dofs=" << (3 * Lc.numNodes) << "\n";
	}
	MG_BuildFixedMasks(pb, H, fixedMasks);
}

// Expand compact element moduli to full structured fine grid order (level 0)
static void EleMod_CompactToFull_Finest(const Problem& pb,
										const std::vector<double>& eleModCompact,
										std::vector<double>& eleModFull) {
	int nx = pb.mesh.resX, ny = pb.mesh.resY, nz = pb.mesh.resZ;
	eleModFull.assign(nx*ny*nz, 0.0);
	for (int e=0; e<pb.mesh.numElements; ++e) {
		int full = pb.mesh.eleMapBack[e];
		if (full >= 0 && full < (int)eleModFull.size()) eleModFull[full] = eleModCompact[e];
	}
}

// Assemble coarsest dense K using Galerkin triple-products with BC at fine level
static void MG_AssembleCoarsestDenseK_Galerkin(const Problem& pb,
											   const MGHierarchy& H,
											   const std::vector<double>& eleModFineFull,
											   const std::vector<uint8_t>& fineFixedDofMask,
											   std::vector<double>& Kc) {
	const MGLevel& Lf = H.levels.front();
	const MGLevel& Lc = H.levels.back();
	const int N = 3*Lc.numNodes;
	Kc.assign(N*N, 0.0);
	const int s = Lc.spanWidth;
	const int grid = s + 1;

	auto idxElemF = [&](int ex,int ey,int ez)->int {
		return (Lf.resY*Lf.resX)*ez + (Lf.resY)*ex + (Lf.resY - 1 - ey);
	};

	// Loop coarse elements
	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
				// coarse element global dofs
				int c_dof[24];
				for (int a=0;a<8;a++) {
					int n = Lc.eNodMat[(cez*Lc.resX*Lc.resY + cex*Lc.resY + cey)*8 + a];
					c_dof[3*a+0] = 3*n+0;
					c_dof[3*a+1] = 3*n+1;
					c_dof[3*a+2] = 3*n+2;
				}
				double Kce[24*24]; for (int i=0;i<24*24;i++) Kce[i]=0.0;

				int fx0 = cex*s;
				int fy0 = (Lc.resY - cey)*s;
				int fz0 = cez*s;
				// iterate s^3 sub-elements
				for (int iz=0; iz<s; ++iz) {
					for (int iy=0; iy<s; ++iy) {
						for (int ix=0; ix<s; ++ix) {
							int fex = fx0 + ix;
							int fey = fy0 - iy - 1;
							int fez = fz0 + iz;
							int ef = idxElemF(fex, fey, fez);
							double Ee = std::max(pb.params.youngsModulusMin, eleModFineFull[ef]);

							// Build Kf = Ee * Ke (24x24)
							double Kf[24*24];
							for (int i=0;i<24;i++) {
								for (int j=0;j<24;j++) {
									Kf[i*24+j] = Ee * pb.mesh.Ke[i*24+j];
								}
							}
							// Apply fine-level BC mask on this fine element's dofs
							for (int v=0; v<8; ++v) {
								int n_f = Lf.eNodMat[ef*8 + v];
								for (int c=0;c<3;c++) {
									if (fineFixedDofMask[3*n_f + c]) {
										int d = 3*v + c;
										for (int j=0;j<24;j++) { Kf[d*24+j] = 0.0; Kf[j*24+d] = 0.0; }
										Kf[d*24 + d] = 1.0;
									}
								}
							}
							// Build T (24x24): maps coarse local dofs to fine local dofs at the 8 vertices
							double T[24*24]; for (int i=0;i<24*24;i++) T[i]=0.0;
							for (int v=0; v<8; ++v) {
								int vx = (v==0||v==3||v==4||v==7) ? 0 : 1;
								int vy = (v==0||v==1||v==4||v==5) ? 0 : 1;
								int vz = (v<=3) ? 0 : 1;
								const double* W = &Lc.weightsNode[(((iz+vz)*grid + (iy+vy))*grid + (ix+vx))*8];
								for (int a=0; a<8; ++a) {
									for (int c=0;c<3;c++) {
										int row = 3*v + c;
										int col = 3*a + c;
										T[row*24 + col] = W[a];
									}
								}
							}
							// Kce += T^T * Kf * T
							double M[24*24];
							for (int i=0;i<24;i++) {
								for (int j=0;j<24;j++) {
									double ssum=0.0;
									for (int k=0;k<24;k++) ssum += Kf[i*24+k]*T[k*24+j];
									M[i*24+j] = ssum;
								}
							}
							for (int i=0;i<24;i++) {
								for (int j=0;j<24;j++) {
									double ssum=0.0;
									for (int k=0;k<24;k++) ssum += T[k*24+i]*M[k*24+j];
									Kce[i*24 + j] += ssum;
								}
							}
						}
					}
				}
				// Scatter Kce to global
				for (int i=0;i<24;i++) {
					int gi = c_dof[i];
					for (int j=0;j<24;j++) {
						int gj = c_dof[j];
						Kc[gi*N + gj] += Kce[i*24 + j];
					}
				}
			}
		}
	}
}

// Reuse H/fixedMasks; per-iteration, rebuild diagonals and assemble SIMP-modulated coarsest K
Preconditioner MakeMGDiagonalPreconditionerFromStatic(const Problem& pb,
														  const MGHierarchy& H,
														  const std::vector<std::vector<uint8_t>>& fixedMasks,
														  const std::vector<double>& eleModulus,
														  const MGPrecondConfig& cfg) {
	// 1) Build per-level diagonals
	std::vector<std::vector<double>> diag;
	MG_BuildDiagonals(pb, H, fixedMasks, eleModulus, diag);

	// 2) Build aggregated Ee at coarsest level and factorize
	std::vector<double> Lcoarse; int Ncoarse = 0;
	{
		const auto& Lc = H.levels.back();
		Ncoarse = 3*Lc.numNodes;
		const int NlimitDofs = 200000;
		if (H.levels.size() == 1 || Ncoarse > NlimitDofs) {
			Ncoarse = 0; // diagonal fallback
		} else {
			// 1) Finest-grid modulus in structured order
			std::vector<double> emFineFull;
			EleMod_CompactToFull_Finest(pb, eleModulus, emFineFull);

			// 2) Coarsest dense K via Galerkin triple products with fine-level BC mask
			std::vector<double> Kc;
			MG_AssembleCoarsestDenseK_Galerkin(pb, H, emFineFull, fixedMasks.front(), Kc);

			// 3) Safety: impose coarsest-level BCs too
			for (int n=0;n<Lc.numNodes;n++) {
				for (int c=0;c<3;c++) {
					if (fixedMasks.back()[3*n+c]) {
						int d = 3*n+c;
						for (int j=0;j<Ncoarse;j++) { Kc[d*Ncoarse + j] = 0.0; Kc[j*Ncoarse + d] = 0.0; }
						Kc[d*Ncoarse + d] = 1.0;
					}
				}
			}

			// 4) Factorize
			if (chol_spd_inplace(Kc, Ncoarse)) Lcoarse.swap(Kc);
			else { Lcoarse.clear(); Ncoarse = 0; }
		}
	}

	// 3) Return preconditioner closure (same as MG diagonal-only path)
	return [H, diag, cfg, &pb, fixedMasks, Lcoarse, Ncoarse](const std::vector<double>& rFree, std::vector<double>& zFree) {
		const int n0_nodes = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		const int n0_dofs  = 3*n0_nodes;
		std::vector<double> r0(n0_dofs, 0.0);
		for (size_t i=0;i<pb.freeDofIndex.size();++i) r0[pb.freeDofIndex[i]] = rFree[i];

		std::vector<std::vector<double>> rLv(H.levels.size());
		std::vector<std::vector<double>> xLv(H.levels.size());
		rLv[0] = r0; xLv[0].assign(n0_dofs, 0.0);

		for (int i=0;i<n0_nodes;i++) {
			for (int c=0;c<3;c++) if (fixedMasks[0][3*i+c]) rLv[0][3*i+c] = 0.0;
		}

		for (size_t l=0; l+1<H.levels.size(); ++l) {
			const int fn_nodes = H.levels[l].numNodes;
			const int cn_nodes = H.levels[l+1].numNodes;
			if ((int)xLv[l].size() != 3*fn_nodes) xLv[l].assign(3*fn_nodes, 0.0);
			const auto& D = diag[l];
			for (int i=0;i<fn_nodes;i++) {
				for (int c=0;c<3;c++) {
					const int d = 3*i+c; if (fixedMasks[l][d]) continue;
					xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1e-30, D[d]);
				}
			}
			rLv[l+1].assign(3*cn_nodes, 0.0);
			for (int c=0;c<3;c++) {
				std::vector<double> rf(fn_nodes), rc;
				for (int i=0;i<fn_nodes;i++) rf[i] = rLv[l][3*i+c];
				MG_Restrict_nodes(H.levels[l+1], H.levels[l], rf, rc);
				for (int i=0;i<cn_nodes;i++) rLv[l+1][3*i+c] = rc[i];
			}
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l+1][3*i+c]) rLv[l+1][3*i+c] = 0.0;
			xLv[l+1].assign(3*cn_nodes, 0.0);
		}

		const size_t Lidx = H.levels.size()-1;
		if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
			chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
			const int cn_nodes = H.levels[Lidx].numNodes;
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[Lidx][3*i+c]) xLv[Lidx][3*i+c] = 0.0;
		} else {
			const auto& D = diag[Lidx];
			const int cn_nodes = H.levels[Lidx].numNodes;
			xLv[Lidx].assign(3*cn_nodes, 0.0);
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) {
				const int d = 3*i+c; if (fixedMasks[Lidx][d]) { xLv[Lidx][d] = 0.0; continue; }
				xLv[Lidx][d] = rLv[Lidx][d] / std::max(1e-30, D[d]);
			}
		}

		for (int l=(int)H.levels.size()-2; l>=0; --l) {
			const int fn_nodes = H.levels[l].numNodes;
			std::vector<double> add(3*fn_nodes, 0.0);
			for (int c=0;c<3;c++) {
				std::vector<double> xc(H.levels[l+1].numNodes), xf;
				for (int i=0;i<H.levels[l+1].numNodes;i++) xc[i] = xLv[l+1][3*i+c];
				MG_Prolongate_nodes(H.levels[l+1], H.levels[l], xc, xf);
				for (int i=0;i<fn_nodes;i++) add[3*i+c] = xf[i];
			}
			if ((int)xLv[l].size() != 3*fn_nodes) xLv[l].assign(3*fn_nodes, 0.0);
			for (int i=0;i<3*fn_nodes;i++) xLv[l][i] += add[i];
			const auto& D = diag[l];
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l][3*i+c]) xLv[l][3*i+c] = 0.0;
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) {
				int d = 3*i+c; if (fixedMasks[l][d]) continue;
				xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1e-30, D[d]);
			}
		}
		zFree.resize(rFree.size());
		for (size_t i=0;i<pb.freeDofIndex.size();++i) zFree[i] = xLv[0][pb.freeDofIndex[i]];
	};
}

void TOP3D_XL_GLOBAL(int nely, int nelx, int nelz, double V0, int nLoop, double rMin) {
	auto tStartTotal = std::chrono::steady_clock::now();
	
	Problem pb;
	InitialSettings(pb.params);
	double CsolidRef = 0.0;
	
	std::cout << "\n==========================Displaying Inputs==========================\n";
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "..............................................Volume Fraction: " << std::setw(6) << V0 << "\n";
	std::cout << "..........................................Filter Radius: " << std::setw(6) << rMin << " Cells\n";
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "................................................Cell Size: " << std::setw(10) << pb.params.cellSize << "\n";
	std::cout << std::fixed;
	std::cout << "...............................................#CG Iterations: " << std::setw(4) << pb.params.cgMaxIt << "\n";
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "...........................................Youngs Modulus: " << std::setw(10) << pb.params.youngsModulus << "\n";
	std::cout << "....................................Youngs Modulus (Min.): " << std::setw(10) << pb.params.youngsModulusMin << "\n";
	std::cout << "...........................................Poissons Ratio: " << std::setw(10) << pb.params.poissonRatio << "\n";
	std::cout << std::fixed << std::setprecision(6);
	
	auto tStart = std::chrono::steady_clock::now();
	CreateVoxelFEAmodel_Cuboid(pb, nely, nelx, nelz);
	ApplyBoundaryConditions(pb);
	PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
	// Build MG hierarchy and fixed masks once; reuse across solves
	MGPrecondConfig mgcfgStatic_tv; mgcfgStatic_tv.nonDyadic = true; mgcfgStatic_tv.maxLevels = 5; mgcfgStatic_tv.weight = 0.6;
	MGHierarchy H; std::vector<std::vector<uint8_t>> fixedMasks; MG_BuildStaticOnce(pb, mgcfgStatic_tv, H, fixedMasks);
    
	auto tModelTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
	std::cout << "Preparing Voxel-based FEA Model Costs " << std::setw(10) << std::setprecision(1) << tModelTime << "s\n";
	
	// Initialize design
	for (double& x: pb.density) x = V0;

	const int ne = pb.mesh.numElements;
	std::vector<double> x = pb.density;
	std::vector<double> xPhys = x;
	std::vector<double> ce(ne, 0.0);
	std::vector<double> eleMod(ne, pb.params.youngsModulus);
	std::vector<double> uFreeWarm; // warm-start buffer for PCG
	// MMA old vectors (mirror MATLAB usage)
	std::vector<double> xold1 = x;
	std::vector<double> xold2 = x;

	// Solve fully solid for reference
	{
		tStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		std::vector<double> uFree; uFree.assign(bFree.size(), 0.0);
		// Preconditioner: reuse static MG context, per-iter diagonals and SIMP-modulated coarsest
		MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgcfg);
        int pcgIters = PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt, MG);
		scatter_from_free(pb, uFree, U);
		double Csolid = ComputeCompliance(pb, eleMod, U, ce);
		CsolidRef = Csolid;
		// Seed warm start for first optimization iteration
		uFreeWarm = uFree;
		auto tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
		std::cout << std::scientific << std::setprecision(6);
		std::cout << "Compliance of Fully Solid Domain: " << std::setw(16) << Csolid << "\n";
		std::cout << std::fixed;
		std::cout << " It.: " << std::setw(4) << 0 << " Solver Time: " << std::setw(4) << std::setprecision(0) << tSolveTime << "s.\n\n";
		std::cout << std::setprecision(6);
	}

	int loop=0;
	double change=1.0;
	double sharpness = 1.0;
	
	while (loop < nLoop && change > 1e-4 && sharpness > 0.01) {
		auto tPerIter = std::chrono::steady_clock::now();
		++loop;
		
		// Update modulus via SIMP
		for (int e=0;e<ne;e++) {
			double rho = std::clamp(xPhys[e], 0.0, 1.0);
			eleMod[e] = pb.params.youngsModulusMin + std::pow(rho, pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		}
		
		// Solve KU=F
		auto tSolveStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		// Ensure warm-start vector matches current system size
		if (uFreeWarm.size() != bFree.size()) uFreeWarm.assign(bFree.size(), 0.0);
		// Reuse static MG context for current SIMP-modified modulus
		MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgcfg);
        int pcgIters = PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, MG);
		scatter_from_free(pb, uFreeWarm, U);
		auto tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSolveStart).count();
		
		// Compliance and sensitivities
		auto tOptStart = std::chrono::steady_clock::now();
        double C = ComputeCompliance(pb, eleMod, U, ce);
        // Normalized reporting to mirror MATLAB: ceNorm = ce / CsolidRef; cObj = sum(Ee*ceNorm); Cdisp = cObj*CsolidRef
        double cObjNorm = 0.0;
        if (CsolidRef > 0) {
            for (int e=0;e<ne;e++) cObjNorm += eleMod[e] * (ce[e] / CsolidRef);
        } else {
            for (int e=0;e<ne;e++) cObjNorm += eleMod[e] * ce[e];
        }
        double Cdisp = (CsolidRef > 0 ? cObjNorm * CsolidRef : C);
        std::vector<double> dc(ne, 0.0);
        for (int e=0;e<ne;e++) {
            double rho = std::clamp(xPhys[e], 0.0, 1.0);
            double dEdrho = pb.params.simpPenalty * std::pow(rho, pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
            double ceNorm = (CsolidRef > 0 ? ce[e] / CsolidRef : ce[e]);
            dc[e] = - dEdrho * ceNorm;
        }
		
		// PDE filter on dc (ft=1): filter(x.*dc)./max(1e-3,x)
		auto tFilterStart = std::chrono::steady_clock::now();
		{
			std::vector<double> xdc(ne);
			for (int e=0;e<ne;e++) xdc[e] = x[e]*dc[e];
			std::vector<double> dc_filt; ApplyPDEFilter(pb, pfFilter, xdc, dc_filt);
			for (int e=0;e<ne;e++) dc[e] = dc_filt[e] / std::max(1e-3, x[e]);
		}
		auto tFilterTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tFilterStart).count();
		
		// OC update with move limits o
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
			// Enforce passive elements (if provided externally)
			if (pb.extBC && !pb.extBC->passiveCompact.empty()) {
				for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xnew[pe] = 1.0;
			}
			double vol = std::accumulate(xnew.begin(), xnew.end(), 0.0) / static_cast<double>(ne);
			if (vol - V0 > 0) l1 = lmid; else l2 = lmid;
		}
		change = 0.0; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
		x.swap(xnew);
		xPhys = x; // no filter in this minimal port
		if (pb.extBC && !pb.extBC->passiveCompact.empty()) {
			for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xPhys[pe] = 1.0;
		}
		sharpness = 4.0 * std::accumulate(xPhys.begin(), xPhys.end(), 0.0, 
			[](double sum, double val) { return sum + val * (1.0 - val); }) / static_cast<double>(ne);
		auto tOptTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tOptStart).count();
		auto tTotalTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tPerIter).count();
		
		double volFrac = std::accumulate(xPhys.begin(), xPhys.end(), 0.0) / static_cast<double>(ne);
		double fval = volFrac - V0;
		
		// Print iteration results (matching MATLAB format)
		std::cout << std::scientific << std::setprecision(8);
		std::cout << " It.:" << std::setw(4) << loop 
				  << " Obj.:" << std::setw(16) << Cdisp 
				  << " Vol.:" << std::setw(6) << std::setprecision(4) << volFrac
				  << " Sharp.:" << std::setw(6) << sharpness
				  << " Cons.:" << std::setw(4) << std::setprecision(2) << fval
				  << " Ch.:" << std::setw(4) << change << "\n";
		std::cout << std::fixed << std::setprecision(2);
		std::cout << " It.: " << std::setw(4) << loop << " (Time)... Total per-It.: " << std::setw(8) << std::scientific << tTotalTime << "s;"
				  << " CG: " << std::setw(8) << tSolveTime << "s;"
				  << " Opti.: " << std::setw(8) << tOptTime << "s;"
				  << " Filtering: " << std::setw(8) << tFilterTime << "s.\n";
		std::cout << std::fixed;
	}
	// Exports
	auto tTotalTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStartTotal).count();
	std::cout << "\n..........Performing Topology Optimization Costs (in total): " 
			  << std::scientific << std::setprecision(4) << tTotalTime << "s.\n";
	std::cout << std::fixed;
	
	const std::string stlDir = out_stl_dir_for_cwd();
	ensure_out_dir(stlDir);
	// NIfTI export disabled
	// export_volume_nifti(pb, xPhys, outDir + "DesignVolume.nii");
	std::string stlFilename = generate_unique_filename("GLOBAL");
	export_surface_stl(pb, xPhys, stlDir + stlFilename, 0.3f);
	std::cout << "STL file saved to: " << stlDir << stlFilename << "\n";
	std::cout << "\nDone.\n";
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
	// Per-element capacity (fixed) for LVF normalization (matches MATLAB PIO)
	std::vector<double> volMaxList(ne, Ve0);
	// MMA state (active elements exclude passives)
	std::vector<int> activeIdx;
	activeIdx.reserve(ne);
	{
		std::vector<uint8_t> isPassive(ne, 0);
		if (pb.extBC && !pb.extBC->passiveCompact.empty()) {
			for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) isPassive[pe] = 1;
		}
		for (int i=0;i<ne;i++) if (!isPassive[i]) activeIdx.push_back(i);
	}
	std::vector<double> xold1(activeIdx.size(), Ve0), xold2(activeIdx.size(), Ve0);
	// Build MG hierarchy and fixed masks once; reuse across solves
	MGPrecondConfig mgcfgStatic_ltv; mgcfgStatic_ltv.nonDyadic = true; mgcfgStatic_ltv.maxLevels = 5; mgcfgStatic_ltv.weight = 0.6;
	MGHierarchy H; std::vector<std::vector<uint8_t>> fixedMasks; MG_BuildStaticOnce(pb, mgcfgStatic_ltv, H, fixedMasks);
    
    double beta = 1.0, eta = 0.5; int p = 16, pMax = 128; int loopbeta=0;
	std::vector<double> uFreeWarm; // warm-start buffer for PCG
    // Fully solid reference compliance (for ce normalization like MATLAB PIO)
    double CsolidRef = 0.0;
    {
        std::vector<double> eleSolid(ne, pb.params.youngsModulus);
        std::vector<double> U(pb.mesh.numDOFs, 0.0);
        std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
        std::vector<double> uFree; uFree.assign(bFree.size(), 0.0);
        MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
        auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleSolid, mgcfg);
        PCG_free(pb, eleSolid, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt, MG);
        scatter_from_free(pb, uFree, U);
        std::vector<double> ceRef; CsolidRef = ComputeCompliance(pb, eleSolid, U, ceRef);
    }
	for (int it=0; it<nLoop; ++it) {
		loopbeta++;
		// SIMP modulus
		for (int e=0;e<ne;e++) {
			eleMod[e] = pb.params.youngsModulusMin + std::pow(xPhys[e], pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		}
		// Solve KU=F
		auto tSolveStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		if (uFreeWarm.size() != bFree.size()) uFreeWarm.assign(bFree.size(), 0.0);
		// Use MG with static hierarchy; per-iter diagonals and SIMP-modulated coarsest
		MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgcfg);
        PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, MG);
        if (!std::all_of(uFreeWarm.begin(), uFreeWarm.end(), [](double v){ return std::isfinite(v); })) {
            MGPrecondConfig mgdiag = mgcfg; mgdiag.maxLevels = 1;
            auto MG2 = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgdiag);
            std::fill(uFreeWarm.begin(), uFreeWarm.end(), 0.0);
            PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, MG2);
        }
		scatter_from_free(pb, uFreeWarm, U);
		double tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSolveStart).count();
        // Compliance sensitivities dc = -dE/drho * ceNorm (normalize by CsolidRef like MATLAB)
        std::vector<double> ce; double C = ComputeCompliance(pb, eleMod, U, ce);
		std::vector<double> dc(ne);
        for (int e=0;e<ne;e++) {
            double dE = pb.params.simpPenalty * std::pow(std::max(1e-9,xPhys[e]), pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
            double ceNorm = (CsolidRef > 0.0 ? ce[e] / CsolidRef : ce[e]);
            dc[e] = - dE * ceNorm;
        }
		// Local Volume Fraction via PDE filter
		std::vector<double> xHat; ApplyPDEFilter(pb, pfLVF, xPhys, xHat);
		double accum=0.0; for (int e=0;e<ne;e++) {
			double ratio = xHat[e] / std::max(1e-12, volMaxList[e]);
			accum += std::pow(ratio, p);
		}
		double f = std::pow(accum / ne, 1.0/p) - 1.0; // <= 0
		// df/dx via chain: df/dxHat * dxHat/dxPhys; approximate df/dxPhys by filtering gradient once
		std::vector<double> dfdx_hat(ne);
		double coeff = std::pow(accum/ne, 1.0/p - 1.0) / ne;
		for (int e=0;e<ne;e++) {
			double vcap = std::max(1e-12, volMaxList[e]);
			double ratio = xHat[e] / vcap;
			dfdx_hat[e] = coeff * p * std::pow(std::max(1e-12, ratio), p-1) * (1.0 / vcap);
		}
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
		// MMA update (m=1 constraint over active elements)
		const int mcons = 1;
		const int nvars = (int)activeIdx.size();
		std::vector<double> xvalM(nvars), xminM(nvars), xmaxM(nvars), df0dxM(nvars), dg_colMajor(nvars);
		for (int i=0;i<nvars;i++) {
			int e = activeIdx[i];
			xvalM[i] = x[e];
			df0dxM[i] = dc_filt[e];
			dg_colMajor[i] = dfdx_filt[e];
		}
		const double move = 0.1;
		for (int i=0;i<nvars;i++) {
			double lo = std::max(0.0, xvalM[i] - move);
			double hi = std::min(1.0, xvalM[i] + move);
			xminM[i] = lo; xmaxM[i] = hi;
		}
		std::vector<double> gx_col(1, f);
		std::vector<double> xmma;
		top3d::mma::MMAseq(mcons, nvars, xvalM, xminM, xmaxM, xold1, xold2, df0dxM, gx_col, dg_colMajor, xmma);
		double change=0.0;
		for (int i=0;i<nvars;i++) {
			int e = activeIdx[i];
			change = std::max(change, std::abs(xmma[i] - x[e]));
			x[e] = std::clamp(xmma[i], 0.0, 1.0);
		}
		// Enforce passive elements
		if (pb.extBC && !pb.extBC->passiveCompact.empty()) {
			for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) x[pe] = 1.0;
		}
		// Update xTilde and xPhys
		ApplyPDEFilter(pb, pfFilter, x, xTilde);
		for (int e=0;e<ne;e++) xPhys[e] = H(xTilde[e]);
		if (pb.extBC && !pb.extBC->passiveCompact.empty()) {
			for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xPhys[pe] = 1.0;
		}
		if (beta < pMax && loopbeta >= 40) { beta *= 2.0; loopbeta = 0; }
		std::cout << " It.:" << (it+1) << " Obj.:" << C << " Cons.:" << f << " CG:" << tSolveTime << "s\n";
		if (change < 1e-4) break;
	}
	
	// Exports
	const std::string stlDir = out_stl_dir_for_cwd();
	ensure_out_dir(stlDir);
	// Adapt iso based on density distribution to avoid empty STL when densities are uniformly low
	double minVal = 1.0, maxVal = 0.0, sumVal = 0.0; int cntGE03 = 0;
	for (double v : xPhys) { minVal = std::min(minVal, v); maxVal = std::max(maxVal, v); sumVal += v; if (v >= 0.3) cntGE03++; }
	double meanVal = sumVal / std::max(1, (int)xPhys.size());
	double fracGE03 = (double)cntGE03 / std::max(1, (int)xPhys.size());
	float iso = 0.3f;
	if (maxVal < 0.3) iso = std::max(0.05f, 0.5f * (float)maxVal);
	else if (fracGE03 < 0.01) iso = 0.2f;
	std::cout << "Density stats -> min:" << minVal << " max:" << maxVal << " mean:" << meanVal << " frac>=0.3:" << fracGE03 << " iso:" << iso << "\n";
	std::string stlFilename = generate_unique_filename("LOCAL");
	export_surface_stl(pb, xPhys, stlDir + stlFilename, iso);
	std::cout << "STL file saved to: " << stlDir << stlFilename << "\n";
	std::cout << "\nDone.\n";
}

void TOP3D_XL_GLOBAL_FromTopVoxel(const std::string& file, double V0, int nLoop, double rMin) {
	auto tStartTotal = std::chrono::steady_clock::now();
	Problem pb; InitialSettings(pb.params);
	std::cout << "\n==========================Displaying Inputs==========================\n";
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "..............................................Volume Fraction: " << std::setw(6) << V0 << "\n";
	std::cout << "..........................................Filter Radius: " << std::setw(6) << rMin << " Cells\n";
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "................................................Cell Size: " << std::setw(10) << pb.params.cellSize << "\n";
	std::cout << std::fixed;
	std::cout << "...............................................#CG Iterations: " << std::setw(4) << pb.params.cgMaxIt << "\n";
	std::cout << std::scientific << std::setprecision(4);
	std::cout << "...........................................Youngs Modulus: " << std::setw(10) << pb.params.youngsModulus << "\n";
	std::cout << "....................................Youngs Modulus (Min.): " << std::setw(10) << pb.params.youngsModulusMin << "\n";
	std::cout << "...........................................Poissons Ratio: " << std::setw(10) << pb.params.poissonRatio << "\n";
	std::cout << std::fixed << std::setprecision(6);
	
	auto tStart = std::chrono::steady_clock::now();
	CreateVoxelFEAmodel_TopVoxel(pb, file);
	ApplyBoundaryConditions(pb);
	PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
    // Build MG hierarchy and fixed masks once; reuse across solves
    MGPrecondConfig mgcfgStatic_tv; mgcfgStatic_tv.nonDyadic = true; mgcfgStatic_tv.maxLevels = 5; mgcfgStatic_tv.weight = 0.6;
    MGHierarchy H; std::vector<std::vector<uint8_t>> fixedMasks; MG_BuildStaticOnce(pb, mgcfgStatic_tv, H, fixedMasks);
	auto tModelTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
	std::cout << "Preparing Voxel-based FEA Model Costs " << std::setw(10) << std::setprecision(1) << tModelTime << "s\n";

	for (double& x: pb.density) x = V0;
	const int ne = pb.mesh.numElements;
	std::vector<double> x = pb.density;
	std::vector<double> xPhys = x;
	std::vector<double> ce(ne, 0.0);
	std::vector<double> eleMod(ne, pb.params.youngsModulus);
	double CsolidRef = 0.0;
	std::vector<double> uFreeWarm; // warm-start buffer for PCG

	{
		tStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		std::vector<double> uFree; uFree.assign(bFree.size(), 0.0);
		MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgcfg);
		int pcgIters = PCG_free(pb, eleMod, bFree, uFree, pb.params.cgTol, pb.params.cgMaxIt, MG);
		scatter_from_free(pb, uFree, U);
		double Csolid = ComputeCompliance(pb, eleMod, U, ce);
		CsolidRef = Csolid;
		// Seed warm start for first optimization iteration
		uFreeWarm = uFree;
		auto tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
		std::cout << std::scientific << std::setprecision(6);
		std::cout << "Compliance of Fully Solid Domain: " << std::setw(16) << Csolid << "\n";
		std::cout << std::fixed;
		std::cout << " It.: " << std::setw(4) << 0 << " Solver Time: " << std::setw(4) << std::setprecision(0) << tSolveTime << "s.\n\n";
		std::cout << std::setprecision(6);
	}

	int loop=0; double change=1.0; double sharpness=1.0;
	while (loop < nLoop && change > 1e-4 && sharpness > 0.01) {
		auto tPerIter = std::chrono::steady_clock::now();
		++loop;
		for (int e=0;e<ne;e++) {
			double rho = std::clamp(xPhys[e], 0.0, 1.0);
			eleMod[e] = pb.params.youngsModulusMin + std::pow(rho, pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		}
		auto tSolveStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		if (uFreeWarm.size() != bFree.size()) uFreeWarm.assign(bFree.size(), 0.0);
		MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgcfg);
		int pcgIters = PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, MG);
		scatter_from_free(pb, uFreeWarm, U);
		auto tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSolveStart).count();
		auto tOptStart = std::chrono::steady_clock::now();
		double C = ComputeCompliance(pb, eleMod, U, ce);
		double cObjNorm = 0.0;
		if (CsolidRef > 0) { for (int e=0;e<ne;e++) cObjNorm += eleMod[e] * (ce[e] / CsolidRef); }
		else { for (int e=0;e<ne;e++) cObjNorm += eleMod[e] * ce[e]; }
		double Cdisp = (CsolidRef > 0 ? cObjNorm * CsolidRef : C);
		std::vector<double> dc(ne, 0.0);
		for (int e=0;e<ne;e++) {
			double rho = std::clamp(xPhys[e], 0.0, 1.0);
			double dEdrho = pb.params.simpPenalty * std::pow(rho, pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
			dc[e] = - dEdrho * ce[e];
		}
		auto tFilterStart = std::chrono::steady_clock::now();
		{
			std::vector<double> xdc(ne);
			for (int e=0;e<ne;e++) xdc[e] = x[e]*dc[e];
			std::vector<double> dc_filt; ApplyPDEFilter(pb, pfFilter, xdc, dc_filt);
			for (int e=0;e<ne;e++) dc[e] = dc_filt[e] / std::max(1e-3, x[e]);
		}
		auto tFilterTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tFilterStart).count();
		double l1=0.0, l2=1e9; double move=0.2; std::vector<double> xnew(ne, 0.0);
		while ((l2-l1)/(l1+l2) > 1e-6) {
			double lmid = 0.5*(l1+l2);
			for (int e=0;e<ne;e++) {
				double val = std::sqrt(std::max(1e-30, -dc[e]/lmid));
				double xe = std::clamp(x[e]*val, x[e]-move, x[e]+move);
				xnew[e] = std::clamp(xe, 0.0, 1.0);
			}
			if (pb.extBC && !pb.extBC->passiveCompact.empty()) { for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xnew[pe] = 1.0; }
			double vol = std::accumulate(xnew.begin(), xnew.end(), 0.0) / static_cast<double>(ne);
			if (vol - V0 > 0) l1 = lmid; else l2 = lmid;
		}
		change = 0.0; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
		x.swap(xnew);
		xPhys = x;
		if (pb.extBC && !pb.extBC->passiveCompact.empty()) { for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xPhys[pe] = 1.0; }
		sharpness = 4.0 * std::accumulate(xPhys.begin(), xPhys.end(), 0.0, [](double sum, double val){ return sum + val*(1.0-val); }) / static_cast<double>(ne);
		auto tOptTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tOptStart).count();
		auto tTotalTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tPerIter).count();
		double volFrac = std::accumulate(xPhys.begin(), xPhys.end(), 0.0) / static_cast<double>(ne);
		double fval = volFrac - V0;
		std::cout << std::scientific << std::setprecision(8);
		std::cout << " It.:" << std::setw(4) << loop
				  << " Obj.:" << std::setw(16) << Cdisp
				  << " Vol.:" << std::setw(6) << std::setprecision(4) << volFrac
				  << " Sharp.:" << std::setw(6) << sharpness
				  << " Cons.:" << std::setw(4) << std::setprecision(2) << fval
				  << " Ch.:" << std::setw(4) << change << "\n";
		std::cout << std::fixed << std::setprecision(2);
		std::cout << " It.: " << std::setw(4) << loop << " (Time)... Total per-It.: " << std::setw(8) << std::scientific << tTotalTime << "s;"
			  << " CG: " << std::setw(8) << tSolveTime << "s;"
			  << " Opti.: " << std::setw(8) << tOptTime << "s;"
			  << " Filtering: " << std::setw(8) << tFilterTime << "s.\n";
		std::cout << std::fixed;
	}
	auto tTotalTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStartTotal).count();
	std::cout << "\n..........Performing Topology Optimization Costs (in total): "
			  << std::scientific << std::setprecision(4) << tTotalTime << "s.\n";
	std::cout << std::fixed;
	const std::string stlDir = out_stl_dir_for_cwd(); ensure_out_dir(stlDir);
	std::string stlFilename = generate_unique_filename("GLOBAL");
	export_surface_stl(pb, xPhys, stlDir + stlFilename, 0.3f);
	std::cout << "STL file saved to: " << stlDir << stlFilename << "\n";
	std::cout << "\nDone.\n";
}

void TOP3D_XL_LOCAL_FromTopVoxel(const std::string& file, double Ve0, int nLoop, double rMin, double rHat) {
	Problem pb; InitialSettings(pb.params);
	CreateVoxelFEAmodel_TopVoxel(pb, file);
	ApplyBoundaryConditions(pb);
	const int ne = pb.mesh.numElements;
	std::vector<double> x(ne, Ve0), xTilde(ne, Ve0), xPhys(ne, Ve0);
	std::vector<double> eleMod(ne, pb.params.youngsModulus);
	PDEFilter pfFilter = SetupPDEFilter(pb, rMin);
	PDEFilter pfLVF = SetupPDEFilter(pb, rHat);
    // Build MG hierarchy and fixed masks once; reuse across solves
    MGPrecondConfig mgcfgStatic_tv; mgcfgStatic_tv.nonDyadic = true; mgcfgStatic_tv.maxLevels = 5; mgcfgStatic_tv.weight = 0.6;
    MGHierarchy H; std::vector<std::vector<uint8_t>> fixedMasks; MG_BuildStaticOnce(pb, mgcfgStatic_tv, H, fixedMasks);
	double beta = 1.0, eta = 0.5; int p = 16, pMax = 128; int loopbeta=0;
	std::vector<double> uFreeWarm; // warm-start buffer for PCG
	for (int it=0; it<nLoop; ++it) {
		loopbeta++;
		for (int e=0;e<ne;e++) eleMod[e] = pb.params.youngsModulusMin + std::pow(xPhys[e], pb.params.simpPenalty) * (pb.params.youngsModulus - pb.params.youngsModulusMin);
		auto tSolveStart = std::chrono::steady_clock::now();
		std::vector<double> U(pb.mesh.numDOFs, 0.0);
		std::vector<double> bFree; restrict_to_free(pb, pb.F, bFree);
		if (uFreeWarm.size() != bFree.size()) uFreeWarm.assign(bFree.size(), 0.0);
		MGPrecondConfig mgcfg; mgcfg.nonDyadic = true; mgcfg.maxLevels = 5; mgcfg.weight = 0.6;
		auto MG = MakeMGDiagonalPreconditionerFromStatic(pb, H, fixedMasks, eleMod, mgcfg);
		PCG_free(pb, eleMod, bFree, uFreeWarm, pb.params.cgTol, pb.params.cgMaxIt, MG);
		scatter_from_free(pb, uFreeWarm, U);
		double tSolveTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - tSolveStart).count();
		std::vector<double> ce; double C = ComputeCompliance(pb, eleMod, U, ce);
		std::vector<double> dc(ne);
		for (int e=0;e<ne;e++) { double dE = pb.params.simpPenalty * std::pow(std::max(1e-9,xPhys[e]), pb.params.simpPenalty-1.0) * (pb.params.youngsModulus - pb.params.youngsModulusMin); dc[e] = - dE * ce[e]; }
		std::vector<double> xHat; ApplyPDEFilter(pb, pfLVF, xPhys, xHat);
		double accum=0.0; for (int e=0;e<ne;e++) accum += std::pow(xHat[e], p);
		double f = std::pow(accum / ne, 1.0/p) - 1.0;
		std::vector<double> dfdx_hat(ne); double coeff = std::pow(accum/ne, 1.0/p - 1.0) / ne; for (int e=0;e<ne;e++) dfdx_hat[e] = coeff * p * std::pow(std::max(1e-9,xHat[e]), p-1);
		std::vector<double> dfdx_phys; ApplyPDEFilter(pb, pfLVF, dfdx_hat, dfdx_phys);
		auto H = [&](double v){ return (std::tanh(beta*eta) + std::tanh(beta*(v-eta))) / (std::tanh(beta*eta) + std::tanh(beta*(1-eta))); };
		auto dH = [&](double v){ double th = std::tanh(beta*(v-eta)); double den = (std::tanh(beta*eta) + std::tanh(beta*(1-eta))); return beta*(1-th*th)/den; };
		std::vector<double> dx(ne); for (int e=0;e<ne;e++) dx[e] = dH(xTilde[e]);
		std::vector<double> dc_filt; ApplyPDEFilter(pb, pfFilter, std::vector<double>(dc.begin(), dc.end()), dc_filt); for (int e=0;e<ne;e++) dc_filt[e] *= dx[e];
		std::vector<double> dfdx_filt; ApplyPDEFilter(pb, pfFilter, dfdx_phys, dfdx_filt); for (int e=0;e<ne;e++) dfdx_filt[e] *= dx[e];
		double l1=0.0, l2=1e9; double move=0.1; std::vector<double> xnew(ne);
		while ((l2-l1)/(l1+l2) > 1e-6) {
			double lm = 0.5*(l1+l2);
			double volg=0.0;
			for (int e=0;e<ne;e++) { double step = -dc_filt[e] / std::max(1e-16, lm*dfdx_filt[e]); double xe = std::clamp(x[e]*std::sqrt(std::max(0.1, step)), x[e]-move, x[e]+move); xnew[e] = std::clamp(xe, 0.0, 1.0); volg += dfdx_filt[e] * (xnew[e]-x[e]); }
			if (pb.extBC && !pb.extBC->passiveCompact.empty()) { for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xnew[pe] = 1.0; }
			if (f + volg > 0) l1 = lm; else l2 = lm;
		}
		double change=0.0; for (int e=0;e<ne;e++) change = std::max(change, std::abs(xnew[e]-x[e]));
		x.swap(xnew);
		ApplyPDEFilter(pb, pfFilter, x, xTilde);
		for (int e=0;e<ne;e++) xPhys[e] = H(xTilde[e]);
		if (pb.extBC && !pb.extBC->passiveCompact.empty()) { for (int pe : pb.extBC->passiveCompact) if (pe>=0 && pe<ne) xPhys[pe] = 1.0; }
		if (beta < pMax && loopbeta >= 40) { beta *= 2.0; loopbeta = 0; }
		std::cout << " It.:" << (it+1) << " Obj.:" << C << " Cons.:" << f << " CG:" << tSolveTime << "s\n";
		if (change < 1e-4) break;
	}
	const std::string stlDir = out_stl_dir_for_cwd(); ensure_out_dir(stlDir);
	std::string stlFilename = generate_unique_filename("LOCAL");
	export_surface_stl(pb, xPhys, stlDir + stlFilename, 0.3f);
	std::cout << "STL file saved to: " << stlDir << stlFilename << "\n";
	std::cout << "\nDone.\n";
}
// ===== Export helpers =====
static void export_surface_stl(const Problem& pb, const std::vector<double>& xPhys, const std::string& path, float iso=0.5f) {
	int ny = pb.mesh.resY, nx = pb.mesh.resX, nz = pb.mesh.resZ;
	std::vector<float> vol(ny*nx*nz, 0.0f);
	for (int e=0; e<pb.mesh.numElements; ++e) vol[pb.mesh.eleMapBack[e]] = static_cast<float>(xPhys[e]);
	std::vector<std::array<float,3>> verts; std::vector<std::array<uint32_t,3>> faces;
	voxsurf::extract_faces(vol, ny, nx, nz, iso, verts, faces);
	ioexp::write_stl_binary(path, verts, faces);
}

} // namespace top3d
