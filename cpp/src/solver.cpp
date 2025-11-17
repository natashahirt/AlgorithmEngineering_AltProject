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

int PCG_free(const Problem& pb,
			  const std::vector<double>& eleModulus,
			  const std::vector<double>& bFree,
			  std::vector<double>& xFree,
			  double tol, int maxIt,
			  Preconditioner M,
			  std::vector<double>* xfull,
			  std::vector<double>* yfull,
			  std::vector<double>* pfull,
			  std::vector<double>* Apfull,
			  std::vector<double>* freeTmp) {
	std::vector<double> r = bFree;
	std::vector<double> z(r.size(), 0.0), p(r.size(), 0.0), Ap(r.size(), 0.0);

	// Acquire buffers (use provided or local)
	std::vector<double> xfull_loc, yfull_loc, pfull_loc, Apfull_loc, freeTmp_loc;
	auto& X = xfull ? *xfull : (xfull_loc = std::vector<double>(pb.mesh.numDOFs, 0.0), xfull_loc);
	auto& Y = yfull ? *yfull : (yfull_loc = std::vector<double>(pb.mesh.numDOFs, 0.0), yfull_loc);
	auto& P = pfull ? *pfull : (pfull_loc = std::vector<double>(pb.mesh.numDOFs, 0.0), pfull_loc);
	auto& A = Apfull ? *Apfull : (Apfull_loc = std::vector<double>(pb.mesh.numDOFs, 0.0), Apfull_loc);
	auto& F = freeTmp ? *freeTmp : (freeTmp_loc = std::vector<double>(pb.freeDofIndex.size(), 0.0), freeTmp_loc);
	if ((int)X.size() != pb.mesh.numDOFs) X.assign(pb.mesh.numDOFs, 0.0);
	if ((int)Y.size() != pb.mesh.numDOFs) Y.assign(pb.mesh.numDOFs, 0.0);
	if ((int)P.size() != pb.mesh.numDOFs) P.assign(pb.mesh.numDOFs, 0.0);
	if ((int)A.size() != pb.mesh.numDOFs) A.assign(pb.mesh.numDOFs, 0.0);
	if ((int)F.size() != (int)pb.freeDofIndex.size()) F.assign(pb.freeDofIndex.size(), 0.0);

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
		std::fill(P.begin(), P.end(), 0.0);
		scatter_from_free(pb, p, P);
		K_times_u_finest(pb, eleModulus, P, A);

		// Fused restrict (A -> Ap) and denom = p·Ap
		double denom = 0.0;
		Ap.resize(p.size());
		for (size_t i=0;i<p.size();++i) {
			int gi = pb.freeDofIndex[i];
			double api = A[gi];
			Ap[i] = api;
			denom += p[i] * api;
		}
		denom = std::max(1e-30, denom);
		double alpha = rz_old / denom;

		// Fused updates of x and r and residual norm
		double rnorm2 = 0.0;
		for (size_t i=0;i<p.size();++i) {
			xFree[i] += alpha * p[i];
			r[i]     -= alpha * Ap[i];
			rnorm2   += r[i] * r[i];
		}
		double res = std::sqrt(rnorm2) / std::max(1e-30, normb);
		if (res < tol) return it+1;

		if (M) M(r, z); else z = r;
		double rz_new;
		if (M) {
			rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
		} else {
			// z == r
			rz_new = rnorm2;
		}
		double beta = rz_new / std::max(1e-30, rz_old);
		for (size_t i=0;i<p.size();++i) p[i] = z[i] + beta * p[i];
		rz_old = rz_new;
	}
	return maxIt;
}

} // namespace top3d
