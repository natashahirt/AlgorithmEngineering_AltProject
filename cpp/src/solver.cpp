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
	std::vector<double>& r  = ws.r;
	std::vector<double>& z  = ws.z;
	std::vector<double>& p  = ws.p;
	std::vector<double>& Ap = ws.Ap;

	// Ensure workspace size
	if (r.size() != bFree.size()) {
		r.resize(bFree.size());
		z.resize(bFree.size());
		p.resize(bFree.size());
		Ap.resize(bFree.size());
	}

	// r = b
	std::copy(bFree.begin(), bFree.end(), r.begin());

	// r = b - A*x (if nonzero initial guess)
	if (!xFree.empty()) {
		K_times_u_finest(pb, eleModulus, xFree, Ap);
		#pragma omp parallel for
		for (size_t i=0;i<r.size();++i) r[i] -= Ap[i];
	} else {
		xFree.assign(r.size(), 0.0);
	}

	if (M) M(r, z); else std::copy(r.begin(), r.end(), z.begin());
	
	double rz_old = 0.0;
	#pragma omp parallel for reduction(+:rz_old)
	for (size_t i=0; i<r.size(); ++i) rz_old += r[i] * z[i];
	
	std::copy(z.begin(), z.end(), p.begin());

	double normb2 = 0.0;
	#pragma omp parallel for reduction(+:normb2)
	for (size_t i=0; i<bFree.size(); ++i) normb2 += bFree[i] * bFree[i];
	const double normb = std::sqrt(normb2);

	for (int it=0; it<maxIt; ++it) {
		// Ap = A * p
		K_times_u_finest(pb, eleModulus, p, Ap);

		// denom = p·Ap
		double denom = 0.0;
		#pragma omp parallel for reduction(+:denom)
		for (size_t i=0;i<p.size();++i) denom += p[i] * Ap[i];
		
		denom = std::max(1.0e-30, denom);
		double alpha = rz_old / denom;

		// Fused updates of x and r and residual norm
		double rnorm2 = 0.0;
		#pragma omp parallel for reduction(+:rnorm2)
		for (size_t i=0;i<p.size();++i) {
			xFree[i] += alpha * p[i];
			r[i]     -= alpha * Ap[i];
			rnorm2   += r[i] * r[i];
		}
		double res = std::sqrt(rnorm2) / std::max(1.0e-30, normb);
		if (res < tol) return it+1;

		if (M) M(r, z); else std::copy(r.begin(), r.end(), z.begin());
		double rz_new;
		if (M) {
			// Parallelize dot product for rz_new
			rz_new = 0.0;
			#pragma omp parallel for reduction(+:rz_new)
			for (size_t i=0; i<r.size(); ++i) rz_new += r[i] * z[i];
		} else {
			// z == r
			rz_new = rnorm2;
		}
		double beta = rz_new / std::max(1.0e-30, rz_old);
		#pragma omp parallel for
		for (size_t i=0;i<p.size();++i) p[i] = z[i] + beta * p[i];
		rz_old = rz_new;
	}
	return maxIt;
}

} // namespace top3d
