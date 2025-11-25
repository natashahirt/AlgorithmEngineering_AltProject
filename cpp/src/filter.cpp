#include "core.hpp"
#include "filter.hpp"
#include <numeric>
#include <cmath>

namespace top3d {

// ===== PDE Filter =====
PDEFilter SetupPDEFilter(const Problem& pb, float filterRadius) {
	PDEFilter pf;
	// 2x2x2 Gaussian points in natural coords and shape derivatives
	const float s[8] = {-1,1,1,-1,-1,1,1,-1};
	const float t[8] = {-1,-1,1,1,-1,-1,1,1};
	const float p[8] = {-1,-1,-1,-1,1,1,1,1};
	const float w[8] = {1,1,1,1,1,1,1,1};
	// Trilinear shape N (8) and dN/ds, dN/dt, dN/dp
	float N[8][8];
	float dS[3*8][8];
	for (int gp=0; gp<8; ++gp) {
		float sg = s[gp]/std::sqrt(3.0), tg = t[gp]/std::sqrt(3.0), pg = p[gp]/std::sqrt(3.0);
		float Ns[8];
		Ns[0]=0.125*(1-sg)*(1-tg)*(1-pg); Ns[1]=0.125*(1+sg)*(1-tg)*(1-pg);
		Ns[2]=0.125*(1+sg)*(1+tg)*(1-pg); Ns[3]=0.125*(1-sg)*(1+tg)*(1-pg);
		Ns[4]=0.125*(1-sg)*(1-tg)*(1+pg); Ns[5]=0.125*(1+sg)*(1-tg)*(1+pg);
		Ns[6]=0.125*(1+sg)*(1+tg)*(1+pg); Ns[7]=0.125*(1-sg)*(1+tg)*(1+pg);
		for (int a=0;a<8;a++) N[gp][a]=Ns[a];
		float dNds[8] = {
			-0.125f*(1-tg)*(1-pg), 0.125f*(1-tg)*(1-pg), 0.125f*(1+tg)*(1-pg), -0.125f*(1+tg)*(1-pg),
			-0.125f*(1-tg)*(1+pg), 0.125f*(1-tg)*(1+pg), 0.125f*(1+tg)*(1+pg), -0.125f*(1+tg)*(1+pg)
		};
		float dNdt[8] = {
			-0.125f*(1-sg)*(1-pg), -0.125f*(1+sg)*(1-pg), 0.125f*(1+sg)*(1-pg), 0.125f*(1-sg)*(1-pg),
			-0.125f*(1-sg)*(1+pg), -0.125f*(1+sg)*(1+pg), 0.125f*(1+sg)*(1+pg), 0.125f*(1-sg)*(1+pg)
		};
		float dNdp[8] = {
			-0.125f*(1-sg)*(1-tg), -0.125f*(1+sg)*(1-tg), -0.125f*(1+sg)*(1+tg), -0.125f*(1-sg)*(1+tg),
			 0.125f*(1-sg)*(1-tg),  0.125f*(1+sg)*(1-tg),  0.125f*(1+sg)*(1+tg),  0.125f*(1-sg)*(1+tg)
		};
		for (int a=0;a<8;a++) { dS[3*gp+0][a]=dNds[a]; dS[3*gp+1][a]=dNdt[a]; dS[3*gp+2][a]=dNdp[a]; }
	}
	// Cell size assumed 1, detJ=1/8, wgt = 1/8 per gp
	float KEF0[8*8]={0}; // dShape' * dShape
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) {
		float sum=0.0; for (int k=0;k<3*8;k++) sum += dS[k][i]*dS[k][j];
		KEF0[i*8+j] = sum;
	}
	float KEF1[8*8]={0}; // N' * diag(wgt) * N; wgt=1/8
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) {
		float sum=0.0; for (int gp=0;gp<8;gp++) sum += N[gp][i]*(1.0/8.0)*N[gp][j];
		KEF1[i*8+j] = sum;
	}
	// iRmin in MATLAB: (filterRadius * eleSize(1))/2/sqrt(3)
	float iRmin = (filterRadius * pb.mesh.eleSize[0]) / (2.0*std::sqrt(3.0));
	for (int i=0;i<8;i++) for (int j=0;j<8;j++) pf.kernel[i*8+j] = iRmin*iRmin*KEF0[i*8+j] + KEF1[i*8+j];
	// Diagonal preconditioner by accumulating kernel contributions to nodes
	pf.diagPrecondNode.assign(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base = e*8;
		for (int a=0;a<8;a++) {
			int na = pb.mesh.eNodMat[base+a];
			float sum=0.0; for (int b=0;b<8;b++) sum += pf.kernel[a*8+b];
			pf.diagPrecondNode[na] += sum;
		}
	}
	for (float& v : pf.diagPrecondNode) v = v>0 ? 1.0/v : 1.0;
	// initialize warm-start buffers
	pf.lastXNode.assign(pb.mesh.numNodes, 0.0);
	pf.lastRhsNode.assign(pb.mesh.numNodes, 0.0);
	return pf;
}

static void MatTimesVec_PDE(const Problem& pb, const PDEFilter& pf, const std::vector<float>& xNode, std::vector<float>& yNode) {
	yNode.assign(pb.mesh.numNodes, 0.0);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base = e*8;
		float u[8]; for (int a=0;a<8;a++) u[a] = xNode[pb.mesh.eNodMat[base+a]];
		float f[8]={0};
		for (int i=0;i<8;i++) {
			float sum=0.0; for (int j=0;j<8;j++) sum += pf.kernel[i*8+j]*u[j];
			f[i]=sum;
		}
		for (int a=0;a<8;a++) yNode[pb.mesh.eNodMat[base+a]] += f[a];
	}
}

void ApplyPDEFilter(const Problem& pb, PDEFilter& pf, const std::vector<float>& srcEle, std::vector<float>& dstEle, PDEFilterWorkspace& ws) {
	// Ele -> Node (sum/8) into ws.rhs
	const int nNodes = pb.mesh.numNodes;
	ws.rhs.assign(nNodes, 0.0f);
	for (int e=0;e<pb.mesh.numElements;e++) {
		float val = srcEle[e] * (1.0/8.0);
		int base = e*8; for (int a=0;a<8;a++) ws.rhs[pb.mesh.eNodMat[base+a]] += val;
	}
	// Solve (PDE kernel) * x = rhs with PCG and Jacobi precond
	// Conditional warm-start: reuse lastX if rhs hasn't changed much
	const float relThresh = 0.1f; // relative RHS change threshold for warm-start
	if ((int)pf.lastXNode.size() != nNodes) pf.lastXNode.assign(nNodes, 0.0f);
	if ((int)pf.lastRhsNode.size() != nNodes) pf.lastRhsNode.assign(nNodes, 0.0f);
	float normRhs=0.0f, diff=0.0f;
	for (int i=0;i<nNodes;++i) { normRhs += ws.rhs[i]*ws.rhs[i]; float d = ws.rhs[i]-pf.lastRhsNode[i]; diff += d*d; }
	normRhs = std::sqrt(normRhs); diff = std::sqrt(diff);
	bool useWarm = (normRhs > 0.0f) && (diff / std::max(1e-30f, normRhs) < relThresh);
	// Prepare workspace vectors
	ws.x.resize(nNodes);
	if (useWarm) ws.x = pf.lastXNode; else std::fill(ws.x.begin(), ws.x.end(), 0.0f);
	ws.r.resize(nNodes);
	ws.z.resize(nNodes);
	ws.p.resize(nNodes);
	ws.Ap.resize(nNodes);
	// r = rhs - A*x (or rhs if x=0)
	if (useWarm) {
		MatTimesVec_PDE(pb, pf, ws.x, ws.Ap);
		for (int i=0;i<nNodes;++i) ws.r[i] = ws.rhs[i] - ws.Ap[i];
	} else {
		for (int i=0;i<nNodes;++i) ws.r[i] = ws.rhs[i];
	}
	for (int i=0;i<nNodes;++i) ws.z[i] = pf.diagPrecondNode[i]*ws.r[i];
	float rz = std::inner_product(ws.r.begin(), ws.r.end(), ws.z.begin(), 0.0f);
	if (rz==0) rz=1.0f; ws.p = ws.z;
	const float tol = 1e-6f;
	const int maxIt = 400;
	// Early exit if warm-start already good
	{
		float rn = std::sqrt(std::inner_product(ws.r.begin(), ws.r.end(), ws.r.begin(), 0.0f));
		if (rn < tol) {
			// Node -> Ele (sum/8)
			dstEle.assign(pb.mesh.numElements, 0.0f);
			for (int e=0;e<pb.mesh.numElements;e++) {
				int base=e*8; float sum=0.0f; for (int a=0;a<8;a++) sum += ws.x[pb.mesh.eNodMat[base+a]];
				dstEle[e] = sum*(1.0f/8.0f);
			}
			pf.lastXNode = ws.x;
			pf.lastRhsNode = ws.rhs;
			return;
		}
	}
	for (int it=0; it<maxIt; ++it) {
		MatTimesVec_PDE(pb, pf, ws.p, ws.Ap);
		float denom = std::inner_product(ws.p.begin(), ws.p.end(), ws.Ap.begin(), 0.0f);
		float alpha = rz / std::max(1e-30f, denom);
		for (int i=0;i<nNodes;++i) ws.x[i] += alpha * ws.p[i];
		for (int i=0;i<nNodes;++i) ws.r[i] -= alpha * ws.Ap[i];
		float rn = std::sqrt(std::inner_product(ws.r.begin(), ws.r.end(), ws.r.begin(), 0.0f));
		if (rn < tol) break;
		for (int i=0;i<nNodes;++i) ws.z[i] = pf.diagPrecondNode[i]*ws.r[i];
		float rz_new = std::inner_product(ws.r.begin(), ws.r.end(), ws.z.begin(), 0.0f);
		float beta = rz_new / std::max(1e-30f, rz);
		for (int i=0;i<nNodes;++i) ws.p[i] = ws.z[i] + beta * ws.p[i];
		rz = rz_new;
	}
	// Node -> Ele (sum/8)
	dstEle.assign(pb.mesh.numElements, 0.0f);
	for (int e=0;e<pb.mesh.numElements;e++) {
		int base=e*8; float sum=0.0f; for (int a=0;a<8;a++) sum += ws.x[pb.mesh.eNodMat[base+a]];
		dstEle[e] = sum*(1.0f/8.0f);
	}
	// Save warm-start buffers
	pf.lastXNode = ws.x;
	pf.lastRhsNode = ws.rhs;
}

} // namespace top3d
