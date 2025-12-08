#include "core.hpp"
#include "fea.hpp"
#include "solver.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

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

double parallel_dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}


int PCG_free(const Problem& pb,
               const std::vector<double>& eleModulus,
               const std::vector<double>& bFree,
               std::vector<double>& xFree,
               double tol, int maxIt,
               const Preconditioner& M,
               PCGFreeWorkspace& ws) {

    const size_t n = bFree.size();
    if (ws.r.size() != n) {
        ws.r.resize(n);
        ws.z.resize(n);
        ws.p.resize(n);
        ws.Ap.resize(n);
    }

    std::vector<double>& r = ws.r;
    std::vector<double>& z = ws.z;
    std::vector<double>& p = ws.p;
    std::vector<double>& Ap = ws.Ap;

    if (xFree.empty()) xFree.assign(n, 0.0);

    double rz_old = 0.0, rz_new = 0.0, denom = 0.0, rnorm2 = 0.0, normb2 = 0.0;
    double alpha = 0.0, beta = 0.0, stop_tol = 0.0;
    int result = maxIt;
    bool done = false;

    #pragma omp parallel
    {
        // --- INITIALIZATION ---
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) r[i] = bFree[i];

        if (!xFree.empty()) {
            #pragma omp single
            { K_times_u_finest(pb, eleModulus, xFree, Ap); }
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) r[i] -= Ap[i];
        }

        if (M) {
            #pragma omp single
            { M(r, z); }
        } else {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) z[i] = r[i];
        }
        
        #pragma omp single
        {
            rz_old = parallel_dot(r, z);
            normb2 = parallel_dot(bFree, bFree);
            stop_tol = std::max(1.0e-30, std::sqrt(normb2)) * tol;
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) p[i] = z[i];
        
        // --- MAIN LOOP ---
        for (int it = 0; it < maxIt; ++it) {
            if(done) break;

            #pragma omp single
            { K_times_u_finest(pb, eleModulus, p, Ap); }

            denom = 0.0;
            #pragma omp for reduction(+:denom) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                denom += p[i] * Ap[i];
            }

            #pragma omp single
            {
                alpha = rz_old / std::max(1.0e-30, denom);
            }

            rnorm2 = 0.0;
            #pragma omp for reduction(+:rnorm2) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                xFree[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
                rnorm2 += r[i] * r[i];
            }
            
            #pragma omp single
            {
                if (std::sqrt(rnorm2) < stop_tol) {
                    result = it + 1;
                    done = true;
                }
            }
            if(done) break;

            if (M) {
                #pragma omp single
                { M(r, z); }
            } else {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) z[i] = r[i];
            }
            
            rz_new = 0.0;
            if (M) {
                #pragma omp for reduction(+:rz_new) schedule(static)
                for (size_t i = 0; i < n; ++i) rz_new += r[i] * z[i];
            } else {
                #pragma omp single
                { rz_new = rnorm2; }
            }

            #pragma omp single
            {
                beta = rz_new / std::max(1.0e-30, rz_old);
                rz_old = rz_new;
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                p[i] = z[i] + beta * p[i];
            }
        }
    }
    return result;
}

} // namespace top3d
