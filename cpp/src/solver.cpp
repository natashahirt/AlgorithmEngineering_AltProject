#include "core.hpp"
#include "fea.hpp"
#include "solver.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace top3d {

void restrict_to_free(const Problem& pb, const DOFData& full, std::vector<double>& freev) {
	freev.resize(pb.freeDofIndex.size());
	if (pb.freeNodeIndex.size() == pb.freeDofIndex.size() &&
		pb.freeCompIndex.size() == pb.freeDofIndex.size()) {
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
			const int n = pb.freeNodeIndex[i];
			const int c = pb.freeCompIndex[i];
			freev[i] = (c==0 ? full.ux[n] : (c==1 ? full.uy[n] : full.uz[n]));
		}
	} else {
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
			const int gi = pb.freeDofIndex[i];
			const int n = gi / 3;
			const int c = gi % 3;
			freev[i] = (c==0 ? full.ux[n] : (c==1 ? full.uy[n] : full.uz[n]));
		}
	}
}

void scatter_from_free(const Problem& pb, const std::vector<double>& freev, DOFData& full) {
	if (pb.freeNodeIndex.size() == pb.freeDofIndex.size() &&
		pb.freeCompIndex.size() == pb.freeDofIndex.size()) {
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
			const int n = pb.freeNodeIndex[i];
			const int c = pb.freeCompIndex[i];
			const double v = freev[i];
			if (c==0) full.ux[n] = v; else if (c==1) full.uy[n] = v; else full.uz[n] = v;
		}
	} else {
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
			const int gi = pb.freeDofIndex[i];
			const int n = gi / 3;
			const int c = gi % 3;
			const double v = freev[i];
			if (c==0) full.ux[n] = v; else if (c==1) full.uy[n] = v; else full.uz[n] = v;
		}
	}
}

Preconditioner make_jacobi_preconditioner(const Problem& pb, const std::vector<double>& eleModulus) {
	// Build full-space diagonal then restrict to free dofs
	std::vector<double> diagFull(pb.mesh.numDOFs, 0.0);
	const auto& Ke = pb.mesh.Ke; // 24x24 row-major
	for (int e=0; e<pb.mesh.numElements; ++e) {
		const double Ee = eleModulus[e];
		if (Ee <= 1.0e-16) continue;
		const int baseD = e*24;
		for (int i=0;i<24;i++) {
			const int gi = pb.mesh.eDofMat[baseD + i];
			diagFull[gi] += Ee * Ke[i*24 + i];
		}
	}
	// Extract diagonal on free dofs
	std::vector<double> diagFree(pb.freeDofIndex.size(), 1.0);
	for (size_t i=0;i<pb.freeDofIndex.size();++i) {
		double d = diagFull[pb.freeDofIndex[i]];
		diagFree[i] = (d > 1.0e-30) ? d : 1.0;
	}
	// Return functor that applies z = inv(D) * r
	return [diag = std::move(diagFree)](const std::vector<double>& r, std::vector<double>& z) {
		if (z.size() != r.size()) z.resize(r.size());
		for (size_t i=0;i<r.size();++i) z[i] = r[i] / diag[i];
	};
}

int PCG_free(const Problem& pb,
			  const std::vector<double>& eleModulus,
			  const std::vector<double>& bFree,
			  std::vector<double>& xFree,
			  double tol, int maxIt,
			  Preconditioner M,
			  PCGFreeWorkspace& ws) {
	std::vector<double> r = bFree;
	std::vector<double> z(r.size(), 0.0), p(r.size(), 0.0), Ap(r.size(), 0.0);

	// Initialize workspace buffers
	const int nNodes = pb.mesh.numNodes;
	ws.xfull.ux.assign(nNodes, 0.0);
	ws.xfull.uy.assign(nNodes, 0.0);
	ws.xfull.uz.assign(nNodes, 0.0);
	ws.yfull.ux.assign(nNodes, 0.0);
	ws.yfull.uy.assign(nNodes, 0.0);
	ws.yfull.uz.assign(nNodes, 0.0);
	ws.tmpFree.resize(static_cast<int>(bFree.size()));
	auto& X = ws.xfull;
	auto& Y = ws.yfull;
	auto& F = ws.tmpFree;

	// r = b - A*x (if nonzero initial guess)
	if (!xFree.empty()) {
		std::fill(X.ux.begin(), X.ux.end(), 0.0);
		std::fill(X.uy.begin(), X.uy.end(), 0.0);
		std::fill(X.uz.begin(), X.uz.end(), 0.0);
		scatter_from_free(pb, xFree, X);
		K_times_u_finest(pb, eleModulus, X, Y);
		restrict_to_free(pb, Y, F);
		for (size_t i=0;i<r.size();++i) r[i] -= F[i];
	}
	if (xFree.empty()) xFree.assign(r.size(), 0.0);

	if (M) M(r, z); else z = r;
	double rz_old = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
	p = z;

	const double normb = std::sqrt(std::inner_product(bFree.begin(), bFree.end(), bFree.begin(), 0.0));
	for (int it=0; it<maxIt; ++it) {
		// Ap = A * p, using workspace buffers
		std::fill(X.ux.begin(), X.ux.end(), 0.0);
		std::fill(X.uy.begin(), X.uy.end(), 0.0);
		std::fill(X.uz.begin(), X.uz.end(), 0.0);
		scatter_from_free(pb, p, X);
		K_times_u_finest(pb, eleModulus, X, Y);

		// Fused restrict (A -> Ap) and denom = p·Ap
		double denom = 0.0;
		Ap.resize(p.size());
		for (size_t i=0;i<p.size();++i) {
			int gi = pb.freeDofIndex[i];
			int n = gi / 3;
			int c = gi % 3;
			double api = (c==0 ? Y.ux[n] : (c==1 ? Y.uy[n] : Y.uz[n]));
			Ap[i] = api;
			denom += p[i] * api;
		}
		denom = std::max(1.0e-30, denom);
		double alpha = rz_old / denom;

		// Fused updates of x and r and residual norm
		double rnorm2 = 0.0;
		for (size_t i=0;i<p.size();++i) {
			xFree[i] += alpha * p[i];
			r[i]     -= alpha * Ap[i];
			rnorm2   += r[i] * r[i];
		}
		double res = std::sqrt(rnorm2) / std::max(1.0e-30, normb);
		if (res < tol) return it+1;

		if (M) M(r, z); else z = r;
		double rz_new;
		if (M) {
			rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
		} else {
			// z == r
			rz_new = rnorm2;
		}
		double beta = rz_new / std::max(1.0e-30, rz_old);
		for (size_t i=0;i<p.size();++i) p[i] = z[i] + beta * p[i];
		rz_old = rz_new;
	}
	return maxIt;
}

} // namespace top3d
