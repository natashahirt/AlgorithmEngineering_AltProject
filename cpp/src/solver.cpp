#include "core.hpp"
#include "fea.hpp"
#include "solver.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace top3d {

void restrict_to_free(const Problem& pb, const std::vector<double>& full, std::vector<double>& freev) {
	freev.resize(pb.freeDofIndex.size());
	for (size_t i=0;i<pb.freeDofIndex.size();++i) freev[i] = full[pb.freeDofIndex[i]];
}

void scatter_from_free(const Problem& pb, const std::vector<double>& freev, std::vector<double>& full) {
	for (size_t i=0;i<pb.freeDofIndex.size();++i) full[pb.freeDofIndex[i]] = freev[i];
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
	const int nFull = pb.mesh.numDOFs;
	ws.xfull.assign(nFull, 0.0);
	ws.yfull.assign(nFull, 0.0);
	ws.tmpFree.resize(static_cast<int>(bFree.size()));
	auto& X = ws.xfull;
	auto& Y = ws.yfull;
	auto& F = ws.tmpFree;

	// r = b - A*x (if nonzero initial guess)
	if (!xFree.empty()) {
		std::fill(X.begin(), X.end(), 0.0);
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
		std::fill(X.begin(), X.end(), 0.0);
		scatter_from_free(pb, p, X);
		K_times_u_finest(pb, eleModulus, X, Y);

		// Fused restrict (A -> Ap) and denom = p·Ap
		double denom = 0.0;
		Ap.resize(p.size());
		for (size_t i=0;i<p.size();++i) {
			int gi = pb.freeDofIndex[i];
			double api = Y[gi];
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
