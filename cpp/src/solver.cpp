#include "core.hpp"
#include "fea.hpp"
#include "solver.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace top3d {

void restrict_to_free(const Problem& pb, const std::vector<float>& full, std::vector<float>& freev) {
	freev.resize(pb.freeDofIndex.size());
	for (size_t i=0;i<pb.freeDofIndex.size();++i) freev[i] = full[pb.freeDofIndex[i]];
}

void scatter_from_free(const Problem& pb, const std::vector<float>& freev, std::vector<float>& full) {
	for (size_t i=0;i<pb.freeDofIndex.size();++i) full[pb.freeDofIndex[i]] = freev[i];
}

int PCG_free(const Problem& pb,
			  const std::vector<float>& eleModulus,
			  const std::vector<float>& bFree,
			  std::vector<float>& xFree,
			  float tol, int maxIt,
			  Preconditioner M,
			  std::vector<float>* xfull,
			  std::vector<float>* yfull,
			  std::vector<float>* pfull,
			  std::vector<float>* Apfull,
			  std::vector<float>* freeTmp) {
	std::vector<float> r = bFree;
	std::vector<float> z(r.size(), 0.0), p(r.size(), 0.0), Ap(r.size(), 0.0);

	// Acquire buffers (use provided or local)
	std::vector<float> xfull_loc, yfull_loc, pfull_loc, Apfull_loc, freeTmp_loc;
	auto& X = xfull ? *xfull : (xfull_loc = std::vector<float>(pb.mesh.numDOFs, 0.0), xfull_loc);
	auto& Y = yfull ? *yfull : (yfull_loc = std::vector<float>(pb.mesh.numDOFs, 0.0), yfull_loc);
	auto& P = pfull ? *pfull : (pfull_loc = std::vector<float>(pb.mesh.numDOFs, 0.0), pfull_loc);
	auto& A = Apfull ? *Apfull : (Apfull_loc = std::vector<float>(pb.mesh.numDOFs, 0.0), Apfull_loc);
	auto& F = freeTmp ? *freeTmp : (freeTmp_loc = std::vector<float>(pb.freeDofIndex.size(), 0.0), freeTmp_loc);
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
	float rz_old = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
	p = z;

	const float normb = std::sqrt(std::inner_product(bFree.begin(), bFree.end(), bFree.begin(), 0.0));
	for (int it=0; it<maxIt; ++it) {
		std::fill(P.begin(), P.end(), 0.0);
		scatter_from_free(pb, p, P);
		K_times_u_finest(pb, eleModulus, P, A);

		// Fused restrict (A -> Ap) and denom = p·Ap
		float denom = 0.0;
		Ap.resize(p.size());
		for (size_t i=0;i<p.size();++i) {
			int gi = pb.freeDofIndex[i];
			float api = A[gi];
			Ap[i] = api;
			denom += p[i] * api;
		}
		denom = std::max(1e-30f, denom);
		float alpha = rz_old / denom;

		// Fused updates of x and r and residual norm
		float rnorm2 = 0.0;
		for (size_t i=0;i<p.size();++i) {
			xFree[i] += alpha * p[i];
			r[i]     -= alpha * Ap[i];
			rnorm2   += r[i] * r[i];
		}
		float res = std::sqrt(rnorm2) / std::max(1e-30f, normb);
		if (res < tol) return it+1;

		if (M) M(r, z); else z = r;
		float rz_new;
		if (M) {
			rz_new = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
		} else {
			// z == r
			rz_new = rnorm2;
		}
		float beta = rz_new / std::max(1e-30f, rz_old);
		for (size_t i=0;i<p.size();++i) p[i] = z[i] + beta * p[i];
		rz_old = rz_new;
	}
	return maxIt;
}

} // namespace top3d
