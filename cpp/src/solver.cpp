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
               const Preconditioner& M, // std::function<void(const vector&, vector&)>
               PCGFreeWorkspace& ws) {

    // 1. Setup Workspace without reallocation if possible
    size_t n = bFree.size();
    if (ws.r.size() != n) {
        ws.r.resize(n);
        ws.z.resize(n); // Only needed if M is present, but keep for safety
        ws.p.resize(n);
        ws.Ap.resize(n);
    }

    // Direct references for cleaner syntax
    std::vector<double>& r = ws.r;
    std::vector<double>& z = ws.z;
    std::vector<double>& p = ws.p;
    std::vector<double>& Ap = ws.Ap;

    // 2. Initialization
    // r = b
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) r[i] = bFree[i];

    // r = b - A*x
    if (!xFree.empty()) {
        K_times_u_finest(pb, eleModulus, xFree, Ap, ws.kTimesU_ws);
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) r[i] -= Ap[i];
    } else {
        xFree.assign(n, 0.0);
    }

    // 3. Preconditioning Setup (Zero-Copy Logic)
    // If M exists, we write to z. If not, we point to r.
    const double* z_ptr = r.data();
    if (M) {
        M(r, z);
        z_ptr = z.data(); // Point to z data
    }
    // Note: If !M, we skipped the std::copy entirely!

    // Initial rz_old
    double rz_old = 0.0;
    if (M) {
        rz_old = parallel_dot(r, z);
    } else {
        // If z == r, dot(r, z) == dot(r, r)
        rz_old = parallel_dot(r, r);
    }

    // p = z (First iteration copy is unavoidable but fast)
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) p[i] = z_ptr[i];

    // Calculate normb for convergence check
    double normb2 = parallel_dot(bFree, bFree);
    const double normb = std::sqrt(normb2);
    const double stop_tol = std::max(1.0e-30, normb) * tol; // Pre-calc threshold

    // --- MAIN LOOP ---
    // Use raw pointers to skip bounds check
    double* __restrict__ r_ptr = r.data();
    double* __restrict__ z_ptr_w = z.data();
    double* __restrict__ p_ptr = p.data();
    double* __restrict__ Ap_ptr = Ap.data();
    double* __restrict__ x_ptr = xFree.data();

    for (int it = 0; it < maxIt; ++it) {

        // 1. Matrix-Vector Multiplication
        K_times_u_finest(pb, eleModulus, p, Ap, ws.kTimesU_ws);

        // 2. Compute denom = dot(p, Ap)
        double denom = 0.0;
        #pragma omp parallel for reduction(+:denom) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            denom += p_ptr[i] * Ap_ptr[i];
        }

        double alpha = rz_old / std::max(1.0e-30, denom);

        // 3. Update x, r and compute rnorm2
        double rnorm2 = 0.0;
        #pragma omp parallel for reduction(+:rnorm2) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            x_ptr[i] += alpha * p_ptr[i];
            double val_r = r_ptr[i] - alpha * Ap_ptr[i];
            r_ptr[i] = val_r;
            rnorm2 += val_r * val_r;
        }

        // Convergence Check
        if (std::sqrt(rnorm2) < stop_tol) return it + 1;

        // 4. Preconditioner
        if (M) {
            M(r, z);

            // 5. Compute rz_new = dot(r, z)
            double rz_new = 0.0;
            #pragma omp parallel for reduction(+:rz_new) schedule(static)
            for (size_t i = 0; i < n; ++i) {
                rz_new += r_ptr[i] * z_ptr_w[i];
            }

            double beta = rz_new / std::max(1.0e-30, rz_old);
            rz_old = rz_new;

            // 6. p = z + beta * p
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                p_ptr[i] = z_ptr_w[i] + beta * p_ptr[i];
            }
        } else {
            double beta = rnorm2 / std::max(1.0e-30, rz_old);
            rz_old = rnorm2;

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                p_ptr[i] = r_ptr[i] + beta * p_ptr[i];
            }
        }
    }

    return maxIt;
}

} // namespace top3d