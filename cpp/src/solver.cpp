#include "core.hpp"
#include "fea.hpp"
#include "solver.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#if defined(_OPENMP)
#include <omp.h>
#endif

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
		if (z.size() != r.size()) {
            // Unsafe to resize inside parallel region if multiple threads call this
            // But we assume z is pre-allocated by caller in parallel context
             z.resize(r.size());
        }
        
#if defined(_OPENMP)
        if (omp_in_parallel()) {
            #pragma omp for schedule(static)
            for (size_t i=0;i<r.size();++i) z[i] = r[i] / diag[i];
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i=0;i<r.size();++i) z[i] = r[i] / diag[i];
        }
#else
		for (size_t i=0;i<r.size();++i) z[i] = r[i] / diag[i];
#endif
	};
}

int PCG_free(const Problem& pb,
               const std::vector<double>& eleModulus,
               const std::vector<double>& bFree,
               std::vector<double>& xFree,
               double tol, int maxIt,
               const Preconditioner& M, // std::function<void(const vector&, vector&)>
               PCGFreeWorkspace& ws) {

    // 1. Setup Workspace without reallocation if possible
    size_t n = bFree.size();
    if (ws.r.size() != n) {
        ws.r.resize(n);
        ws.z.resize(n);
        ws.p.resize(n);
        ws.Ap.resize(n);
    }

    // Direct references for cleaner syntax
    std::vector<double>& r = ws.r;
    std::vector<double>& z = ws.z;
    std::vector<double>& p = ws.p;
    std::vector<double>& Ap = ws.Ap;
    
    // Ensure xFree is sized
    if (xFree.size() != n) xFree.assign(n, 0.0);

    // Shared scalars
    double rz_old = 0.0;
    double rz_new = 0.0;
    double normb2 = 0.0;
    double stop_tol = 0.0;
    double denom = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double rnorm2 = 0.0;
    int iterations = 0;

    // Use raw pointers for better optimization inside parallel region
    double* __restrict__ r_ptr = r.data();
    double* __restrict__ z_ptr = z.data();
    double* __restrict__ p_ptr = p.data();
    double* __restrict__ Ap_ptr = Ap.data();
    double* __restrict__ x_ptr = xFree.data();
    const double* __restrict__ b_ptr = bFree.data();

#pragma omp parallel
{
    // 2. Initialization
    // r = b
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) r_ptr[i] = b_ptr[i];

    // r = b - A*x
    // Compute A*x. Note: K_times_u_finest is parallel-aware.
    K_times_u_finest(pb, eleModulus, xFree, Ap, ws.kTimesU_ws);
    
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) r_ptr[i] -= Ap_ptr[i];

    // 3. Preconditioning Setup
    if (M) {
        M(r, z); // Parallel-aware
    } else {
        // z = r
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) z_ptr[i] = r_ptr[i];
    }

    // rz_old = dot(r, z)
    #pragma omp single
    rz_old = 0.0;
    
    #pragma omp for reduction(+:rz_old) schedule(static)
    for (size_t i = 0; i < n; ++i) rz_old += r_ptr[i] * z_ptr[i];

    // p = z
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; ++i) p_ptr[i] = z_ptr[i];

    // Calculate normb for convergence check
    #pragma omp single
    normb2 = 0.0;
    
    #pragma omp for reduction(+:normb2) schedule(static)
    for (size_t i = 0; i < n; ++i) normb2 += b_ptr[i] * b_ptr[i];

    #pragma omp single
    {
        stop_tol = std::max(1.0e-30, std::sqrt(normb2)) * tol;
        iterations = maxIt; // Default return if loop completes
    }

    // --- MAIN LOOP ---
    for (int it = 0; it < maxIt; ++it) {
        // 1. Matrix-Vector Multiplication: Ap = K * p
        K_times_u_finest(pb, eleModulus, p, Ap, ws.kTimesU_ws);

        // 2. Compute denom = dot(p, Ap)
        #pragma omp single
        denom = 0.0;
        
        #pragma omp for reduction(+:denom) schedule(static)
        for (size_t i = 0; i < n; ++i) denom += p_ptr[i] * Ap_ptr[i];

        #pragma omp single
        alpha = rz_old / std::max(1.0e-30, denom);
        // Implicit barrier

        // 3. Update x, r and compute rnorm2
        #pragma omp single
        rnorm2 = 0.0;

        #pragma omp for reduction(+:rnorm2) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            x_ptr[i] += alpha * p_ptr[i];
            double val_r = r_ptr[i] - alpha * Ap_ptr[i];
            r_ptr[i] = val_r;
            rnorm2 += val_r * val_r;
        }

        // Convergence Check (must be done by all threads or use single + broadcast)
        // We can just break. omp for has barrier.
        bool converged = false;
        #pragma omp single
        {
            if (std::sqrt(rnorm2) < stop_tol) {
                iterations = it + 1;
                converged = true;
            }
        }
        if (converged) break; // All threads break

        // 4. Preconditioner
        if (M) {
            M(r, z); // Parallel-aware

            // 5. Compute rz_new = dot(r, z)
            #pragma omp single
            rz_new = 0.0;
            
            #pragma omp for reduction(+:rz_new) schedule(static)
            for (size_t i = 0; i < n; ++i) rz_new += r_ptr[i] * z_ptr[i];

            #pragma omp single
            {
                beta = rz_new / std::max(1.0e-30, rz_old);
                rz_old = rz_new;
            }

            // 6. p = z + beta * p
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                p_ptr[i] = z_ptr[i] + beta * p_ptr[i];
            }
        } else {
            #pragma omp single
            {
                beta = rnorm2 / std::max(1.0e-30, rz_old);
                rz_old = rnorm2;
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                p_ptr[i] = r_ptr[i] + beta * p_ptr[i];
            }
        }
    }
} // End of parallel region

    return iterations;
}

} // namespace top3d
