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

    const double* __restrict__ b_ptr = bFree.data();
    double* __restrict__ r_ptr = r.data();
    double* __restrict__ z_ptr = z.data();
    double* __restrict__ p_ptr = p.data();
    double* __restrict__ Ap_ptr = Ap.data();
    double* __restrict__ x_ptr = xFree.data();

    // Shared state for the entire PCG - all declared here so they're shared
    double rz_old = 0.0, rz_new = 0.0, denom = 0.0, rnorm2 = 0.0, normb2 = 0.0;
    double alpha = 0.0, beta = 0.0, stop_tol = 0.0;
    int result = maxIt;
    bool done = false;
    bool has_initial_x = false;

    // Check for initial x outside parallel region
    for (size_t i = 0; i < n && !has_initial_x; ++i) {
        if (x_ptr[i] != 0.0) has_initial_x = true;
    }

    // ========== SINGLE PARALLEL REGION FOR ENTIRE PCG ==========
    #pragma omp parallel
    {
        double local_sum;

        // --- INITIALIZATION ---
        // r = b
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) r_ptr[i] = b_ptr[i];

        // r = b - A*x if needed
        if (has_initial_x) {
            K_times_u_finest(pb, eleModulus, xFree, Ap, ws.kTimesU_ws);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) r_ptr[i] -= Ap_ptr[i];
        }

        // z = M^-1 * r
        if (M) {
            M(r, z);
        } else {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) z_ptr[i] = r_ptr[i];
        }

        // rz_old = dot(r, z), normb2 for tolerance
        local_sum = 0.0;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) local_sum += r_ptr[i] * z_ptr[i];
        #pragma omp atomic
        rz_old += local_sum;

        double local_normb = 0.0;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) local_normb += b_ptr[i] * b_ptr[i];
        #pragma omp atomic
        normb2 += local_normb;
        #pragma omp barrier

        #pragma omp single
        { stop_tol = std::sqrt(normb2) * tol; }

        // p = z
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) p_ptr[i] = z_ptr[i];

        // --- MAIN ITERATION LOOP ---
        for (int it = 0; it < maxIt && !done; ++it) {

            // Ap = K * p
            K_times_u_finest(pb, eleModulus, p, Ap, ws.kTimesU_ws);

            // denom = dot(p, Ap)
            #pragma omp single
            { denom = 0.0; }
            local_sum = 0.0;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) local_sum += p_ptr[i] * Ap_ptr[i];
            #pragma omp atomic
            denom += local_sum;
            #pragma omp barrier

            // alpha = rz_old / denom (computed by all threads identically)
            alpha = rz_old / std::max(1.0e-30, denom);

            // x += alpha*p, r -= alpha*Ap, rnorm2 = ||r||^2
            #pragma omp single
            { rnorm2 = 0.0; }
            local_sum = 0.0;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                x_ptr[i] += alpha * p_ptr[i];
                double ri = r_ptr[i] - alpha * Ap_ptr[i];
                r_ptr[i] = ri;
                local_sum += ri * ri;
            }
            #pragma omp atomic
            rnorm2 += local_sum;
            #pragma omp barrier

            // Convergence check
            #pragma omp single
            {
                if (std::sqrt(rnorm2) < stop_tol) {
                    done = true;
                    result = it + 1;
                }
            }
            // implicit barrier after single

            if (done) break;

            // z = M^-1 * r
            if (M) {
                M(r, z);
            } else {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n; ++i) z_ptr[i] = r_ptr[i];
            }

            // rz_new = dot(r, z)
            #pragma omp single
            { rz_new = 0.0; }
            local_sum = 0.0;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) local_sum += r_ptr[i] * z_ptr[i];
            #pragma omp atomic
            rz_new += local_sum;
            #pragma omp barrier

            // beta = rz_new / rz_old
            beta = rz_new / std::max(1.0e-30, rz_old);

            // p = z + beta * p
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                p_ptr[i] = z_ptr[i] + beta * p_ptr[i];
            }

            #pragma omp single
            { rz_old = rz_new; }
        }
    } // end parallel region

    return result;
}

} // namespace top3d
