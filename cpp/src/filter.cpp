#include "core.hpp"
#include "filter.hpp"
#include <numeric>
#include <cmath>

namespace top3d {

// ===== PDE Filter =====
PDEFilter SetupPDEFilter(const Problem& pb, float filterRadius) {
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
	double iRmin = (static_cast<double>(filterRadius) * static_cast<double>(pb.mesh.eleSize[0])) / (2.0*std::sqrt(3.0));
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) pf.kernel[i*8+j] = iRmin*iRmin*KEF0[i*8+j] + KEF1[i*8+j];
	// Diagonal preconditioner by accumulating kernel contributions to nodes
	double rowSum[8];
	for (int a=0;a<8;a++) {
		double s = 0.0;
		for (int b=0;b<8;b++) s += pf.kernel[a*8 + b];
		rowSum[a] = s;
	}
	pf.diagPrecondNode.assign(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base = e*8;
		for (int a=0;a<8;a++) {
			int na = pb.mesh.eNodMat[base+a];
			pf.diagPrecondNode[na] += rowSum[a];
		}
	}
	for (double& v : pf.diagPrecondNode) v = v>0 ? 1.0/v : 1.0;
	// initialize warm-start buffers
	pf.lastXNode.assign(pb.mesh.numNodes, 0.0);
	pf.lastRhsNode.assign(pb.mesh.numNodes, 0.0);
	return pf;
}

static void MatTimesVec_PDE(const Problem& pb, const PDEFilter& pf, const std::vector<double>& xNode, std::vector<double>& yNode) {
	const auto& mesh = pb.mesh;
	const int numNodes    = mesh.numNodes;
	const int numElements = mesh.numElements;
	
    // Resize if needed, but delay zeroing to parallel region
    if (yNode.size() != numNodes) yNode.resize(numNodes);

	const int* __restrict__ eNod = mesh.eNodMat.data();
	// 2D view over the 8x8 kernel to aid autovectorization
	const double (* __restrict__ K)[8] = reinterpret_cast<const double (*)[8]>(pf.kernel.data());
#if defined(_OPENMP)
	#pragma omp parallel
	{
        // Parallel Zeroing (Fixes cache coherence storm)
        #pragma omp for schedule(static)
        for (int i=0; i<numNodes; ++i) yNode[i] = 0.0;
        
		alignas(64) double u[8];
		alignas(64) double f[8];

        if (mesh.coloring.numColors > 0) {
            // COLORED PATH: No atomics needed
            const auto& buckets = mesh.coloring.colorBuckets;
            for (int c=0; c < mesh.coloring.numColors; ++c) {
                const auto& elems = buckets[c];
                int nElems = static_cast<int>(elems.size());
                
                #pragma omp for schedule(static) 
                for (int i=0; i<nElems; ++i) {
                    int e = elems[i];
                    const int base = e*8;
                    
                    // gather
                    for (int a=0; a<8; ++a) u[a] = xNode[eNod[base + a]];
                    
                    // f = K * u
                    for (int k=0; k<8; ++k) {
                        const double* __restrict__ Kk = K[k];
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (int j=0; j<8; ++j) sum += Kk[j]*u[j];
                        f[k] = sum;
                    }
                    
                    // scatter WITHOUT atomics (safe due to coloring)
                    for (int a=0; a<8; ++a) {
                        yNode[eNod[base + a]] += f[a];
                    }
                }
            }
        } else {
            // FALLBACK: Atomic updates
            #pragma omp for schedule(static)
            for (int e=0; e<numElements; ++e) {
                const int base = e*8;
                // gather
                for (int a=0; a<8; ++a) {
                    const int n = eNod[base + a];
                    u[a] = xNode[n];
                }
                // f = K * u
                for (int i=0; i<8; ++i) {
                    const double* __restrict__ Ki = K[i];
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (int j=0; j<8; ++j) sum += Ki[j]*u[j];
                    f[i] = sum;
                }
                // scatter with atomics
                for (int a=0; a<8; ++a) {
                    const int n = eNod[base + a];
                    #pragma omp atomic update
                    yNode[n] += f[a];
                }
            }
        }
	}
#else
	// Sequential fallback
    std::fill(yNode.begin(), yNode.end(), 0.0);
	for (int e=0; e<numElements; ++e) {
		const int base = e*8;
		alignas(64) double u[8];
		for (int a=0; a<8; ++a) u[a] = xNode[eNod[base + a]];
		alignas(64) double f[8];
		for (int i=0; i<8; ++i) {
			const double* __restrict__ Ki = K[i];
			double sum = 0.0;
			#pragma omp simd reduction(+:sum)
			for (int j=0; j<8; ++j) sum += Ki[j]*u[j];
			f[i] = sum;
		}
		for (int a=0; a<8; ++a) yNode[eNod[base + a]] += f[a];
	}
#endif
}

void ApplyPDEFilter(const Problem& pb, PDEFilter& pf, const std::vector<float>& srcEle, std::vector<float>& dstEle, PDEFilterWorkspace& ws) {
	// Ele -> Node (sum/8) into ws.rhs
	const int nNodes = pb.mesh.numNodes;
	ws.rhs.assign(nNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		double val = static_cast<double>(srcEle[e]) * (1.0/8.0);
		int base = e*8; for (int a=0;a<8;a++) ws.rhs[pb.mesh.eNodMat[base+a]] += val;
	}
	// Solve (PDE kernel) * x = rhs with PCG and Jacobi precond
	// Conditional warm-start: reuse lastX if rhs hasn't changed much
	const double relThresh = 1000.0; // Aggressive warm-start
	if ((int)pf.lastXNode.size() != nNodes) pf.lastXNode.assign(nNodes, 0.0);
	if ((int)pf.lastRhsNode.size() != nNodes) pf.lastRhsNode.assign(nNodes, 0.0);
	double normRhs=0.0, diff=0.0;
	for (int i=0;i<nNodes;++i) {
		const double rhs_i = ws.rhs[i];
		const double d = rhs_i - pf.lastRhsNode[i];
		normRhs += rhs_i * rhs_i;
		diff += d * d;
	}
	normRhs = std::sqrt(normRhs); diff = std::sqrt(diff);
	bool useWarm = (normRhs > 0.0) && (diff / std::max(1.0e-30, normRhs) < relThresh);
	// Prepare workspace vectors
	ws.x.resize(nNodes);
	if (useWarm) ws.x = pf.lastXNode; else std::fill(ws.x.begin(), ws.x.end(), 0.0);
	ws.r.resize(nNodes);
	ws.z.resize(nNodes);
	ws.p.resize(nNodes);
	ws.Ap.resize(nNodes);
	// r = rhs - A*x (or rhs if x=0)
	if (useWarm) {
		MatTimesVec_PDE(pb, pf, ws.x, ws.Ap);
		#pragma omp parallel for
		for (int i=0;i<nNodes;++i) ws.r[i] = ws.rhs[i] - ws.Ap[i];
	} else {
		#pragma omp parallel for
		for (int i=0;i<nNodes;++i) ws.r[i] = ws.rhs[i];
	}
	double rz = 0.0;
	#pragma omp parallel for reduction(+:rz)
	for (int i=0;i<nNodes;++i) {
		ws.z[i] = pf.diagPrecondNode[i]*ws.r[i];
		rz += ws.r[i]*ws.z[i];
	}
	if (rz==0) rz=1.0; ws.p = ws.z;
	const double tol = 1e-3;
	const int maxIt = 80;
	// Early exit if warm-start already good
	{
		double rn2 = 0.0;
		#pragma omp parallel for reduction(+:rn2)
		for (int i=0;i<nNodes;++i) rn2 += ws.r[i]*ws.r[i];
		double rn = std::sqrt(rn2);
		if (rn < 1.0e-2) {
			// Node -> Ele (sum/8)
			dstEle.assign(pb.mesh.numElements, 0.0f);
			for (int e=0;e<pb.mesh.numElements;e++) {
				int base=e*8; double sum=0.0; for (int a=0;a<8;a++) sum += ws.x[pb.mesh.eNodMat[base+a]];
				dstEle[e] = static_cast<float>(sum*(1.0/8.0));
			}
			pf.lastXNode = ws.x;
			pf.lastRhsNode = ws.rhs;
			return;
		}
	}
	for (int it=0; it<maxIt; ++it) {
		MatTimesVec_PDE(pb, pf, ws.p, ws.Ap);
		double denom = 0.0;
		#pragma omp parallel for reduction(+:denom)
		for (int i=0;i<nNodes;++i) denom += ws.p[i]*ws.Ap[i];
		double alpha = rz / std::max(1.0e-30, denom);
		#pragma omp parallel for
		for (int i=0;i<nNodes;++i) ws.x[i] += alpha * ws.p[i];
		#pragma omp parallel for
		for (int i=0;i<nNodes;++i) ws.r[i] -= alpha * ws.Ap[i];
		double rn2 = 0.0;
		#pragma omp parallel for reduction(+:rn2)
		for (int i=0;i<nNodes;++i) rn2 += ws.r[i]*ws.r[i];
		double rn = std::sqrt(rn2);
		if (rn < tol) break;
		double rz_new = 0.0;
		#pragma omp parallel for reduction(+:rz_new)
		for (int i=0;i<nNodes;++i) {
			ws.z[i] = pf.diagPrecondNode[i]*ws.r[i];
			rz_new += ws.r[i]*ws.z[i];
		}
		double beta = rz_new / std::max(1.0e-30, rz);
		#pragma omp parallel for
		for (int i=0;i<nNodes;++i) ws.p[i] = ws.z[i] + beta * ws.p[i];
		rz = rz_new;
	}
	// Node -> Ele (sum/8)
	dstEle.assign(pb.mesh.numElements, 0.0f);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base=e*8; double sum=0.0; for (int a=0;a<8;a++) sum += ws.x[pb.mesh.eNodMat[base+a]];
		dstEle[e] = static_cast<float>(sum*(1.0/8.0));
	}
	// Save warm-start buffers
	pf.lastXNode = ws.x;
	pf.lastRhsNode = ws.rhs;
}

} // namespace top3d
