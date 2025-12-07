
// Multigrid preconditioner composition
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include "multigrid/detail/masks.hpp"
#include "multigrid/detail/transfers.hpp"
#include "multigrid/detail/diagonal.hpp"
#include "multigrid/detail/coarsest.hpp"
#include "multigrid/detail/hierarchy.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

namespace top3d { namespace mg {

// Build H and fixed masks once, reuse later
void build_static_once(const Problem& pb, const MGPrecondConfig& cfg,
								  MGHierarchy& H, std::vector<std::vector<uint8_t>>& fixedMasks) {
	H.levels.clear();
	// Reduced limit to ensure dense solver is fast (< 1s). 
    // 2000 DOFs -> Matrix 2000x2000 -> 16MB -> Cholesky ~ 2.6 GFLOPs.
	const int NlimitDofs = 1000;
	int adaptiveMax = ComputeAdaptiveMaxLevels(pb, cfg.nonDyadic, cfg.maxLevels, NlimitDofs);
	BuildMGHierarchy(pb, cfg.nonDyadic, H, adaptiveMax);
	
	const auto& Lc = H.levels.back();
	std::cout << "[MG] levels=" << H.levels.size()
			  << " coarsest=" << Lc.resX << "x" << Lc.resY << "x" << Lc.resZ
			  << " dofs=" << (3 * Lc.numNodes) << "\n";

	MG_BuildFixedMasks(pb, H, fixedMasks);
}


// Reuse H/fixedMasks; per-iteration, rebuild diagonals and assemble SIMP-modulated coarsest K
Preconditioner make_diagonal_preconditioner_from_static(const Problem& pb,
														  const MGHierarchy& H,
														  const std::vector<std::vector<uint8_t>>& fixedMasks,
														  const std::vector<double>& eleModulus,
														  const MGPrecondConfig& cfg) {
	// 1) Build per-level diagonals
	std::vector<std::vector<double>> diag;
	MG_BuildDiagonals(pb, H, fixedMasks, eleModulus, diag);

	// 2) Build aggregated Ee at coarsest level and factorize
	std::vector<double> Lcoarse; int Ncoarse = 0;
	{
		const auto& Lc = H.levels.back();
		Ncoarse = 3*Lc.numNodes;
		const int NlimitDofs = 1000; // Match build limit
		if (H.levels.size() == 1 || Ncoarse > NlimitDofs) {
			Ncoarse = 0; // diagonal fallback
		} else {
			// 1) Finest-grid modulus in structured order
			std::vector<double> emFineFull;
			EleMod_CompactToFull_Finest(pb, eleModulus, emFineFull);

			// 2) Coarsest dense K via Galerkin triple products with fine-level BC mask
			std::vector<double> Kc;
			MG_AssembleCoarsestDenseK_Galerkin(pb, H, emFineFull, fixedMasks.front(), Kc);

			// 3) Safety: impose coarsest-level BCs too
			for (int n=0;n<Lc.numNodes;n++) {
				for (int c=0;c<3;c++) {
					if (fixedMasks.back()[3*n+c]) {
						int d = 3*n+c;
						for (int j=0;j<Ncoarse;j++) { Kc[d*Ncoarse + j] = 0.0; Kc[j*Ncoarse + d] = 0.0; }
						Kc[d*Ncoarse + d] = 1.0;
					}
				}
			}

			// 4) Factorize
			if (chol_spd_inplace(Kc, Ncoarse)) Lcoarse.swap(Kc);
			else { Lcoarse.clear(); Ncoarse = 0; }
		}
	}

    // Allocate workspace once
    auto ws = std::make_shared<MGWorkspace>();
    ws->resize(H);

	// 3) Return preconditioner closure (adapt to double free-DOF vectors)
	// Single parallel region V-cycle to minimize barrier overhead
	return [H, diag, cfg, &pb, fixedMasks, Lcoarse, Ncoarse, ws](const std::vector<double>& rFree, std::vector<double>& zFree) mutable {
        // Alias workspace vectors for brevity
        auto& rLv = ws->rLv;
        auto& xLv = ws->xLv;

		const int n0_nodes = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		const int n0_dofs = 3 * n0_nodes;
		const size_t nFreeDofs = pb.freeDofIndex.size();
		const size_t numLevels = H.levels.size();
		const size_t Lidx = numLevels - 1;
		const int coarse_nodes = H.levels[Lidx].numNodes;
		const double weight = cfg.weight;

		// Ensure output is sized
		zFree.resize(rFree.size());

		// ============ SINGLE PARALLEL REGION FOR ENTIRE V-CYCLE ============
		#pragma omp parallel
		{
			// Thread-local weight buffer for transfer operations
			std::vector<double> W_local(16); // Max span is typically 4, so 2*4-1=7 is enough

			// === ENTRY: Zero finest + scatter from free DOFs ===
			#pragma omp for schedule(static) nowait
			for (int i = 0; i < n0_dofs; ++i) {
				rLv[0][i] = 0.0;
				xLv[0][i] = 0.0;
			}

			#pragma omp barrier  // Need rLv[0] zeroed before scatter

			#pragma omp for schedule(static) nowait
			for (size_t i = 0; i < nFreeDofs; ++i) {
				int dMort = pb.freeDofIndex[i];
				int nMort = dMort / 3;
				int comp  = dMort % 3;
				int nLex = pb.mesh.nodMapBack[nMort];
				rLv[0][3*nLex + comp] = rFree[i];
			}

			#pragma omp barrier  // Need scatter complete before BC mask

			#pragma omp for schedule(static)
			for (int i = 0; i < n0_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					if (fixedMasks[0][3*i+c]) rLv[0][3*i+c] = 0.0;
				}
			}
			// implicit barrier here

			// === DOWNWARD PASS ===
			for (size_t l = 0; l + 1 < numLevels; ++l) {
				const int fn_nodes = H.levels[l].numNodes;
				const int cn_nodes = H.levels[l+1].numNodes;
				const int cn_dofs = 3 * cn_nodes;
				const auto& D = diag[l];

				// Pre-smoothing: x = w * D^-1 * r
				#pragma omp for schedule(static) nowait
				for (int i = 0; i < fn_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						const int d = 3*i+c;
						if (fixedMasks[l][d]) {
							xLv[l][d] = 0.0;
						} else {
							xLv[l][d] = weight * rLv[l][d] / std::max(1.0e-30, D[d]);
						}
					}
				}

				// Zero coarse level buffers (can run concurrently with smoothing)
				#pragma omp for schedule(static)
				for (int i = 0; i < cn_dofs; ++i) {
					rLv[l+1][i] = 0.0;
					xLv[l+1][i] = 0.0;
				}
				// implicit barrier - need both smoothing and zeroing done before restriction

				// Restriction (uses inner version - no extra parallel region)
				MG_Restrict_nodes_Vec3_Inner(H.levels[l+1], H.levels[l], rLv[l], rLv[l+1], W_local);
				// implicit barrier at end of omp for inside Inner function

				// Apply BC mask on coarse residual
				#pragma omp for schedule(static)
				for (int i = 0; i < cn_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						if (fixedMasks[l+1][3*i+c]) rLv[l+1][3*i+c] = 0.0;
					}
				}
				// implicit barrier
			}

			// === COARSEST LEVEL SOLVE ===
			#pragma omp single
			{
				if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
					chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
				} else {
					// Diagonal fallback (serial for small coarse grid)
					const auto& D = diag[Lidx];
					for (int i = 0; i < coarse_nodes; ++i) {
						for (int c = 0; c < 3; ++c) {
							const int d = 3*i+c;
							if (fixedMasks[Lidx][d]) {
								xLv[Lidx][d] = 0.0;
							} else {
								xLv[Lidx][d] = rLv[Lidx][d] / std::max(1.0e-30, D[d]);
							}
						}
					}
				}
			}
			// implicit barrier after single

			// Apply BC mask on coarsest solution
			#pragma omp for schedule(static)
			for (int i = 0; i < coarse_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					if (fixedMasks[Lidx][3*i+c]) xLv[Lidx][3*i+c] = 0.0;
				}
			}
			// implicit barrier

			// === UPWARD PASS ===
			for (int l = (int)numLevels - 2; l >= 0; --l) {
				const int fn_nodes = H.levels[l].numNodes;
				const int fn_dofs = 3 * fn_nodes;
				const auto& D = diag[l];

				// Prolongation (uses inner version - no extra parallel region)
				MG_Prolongate_nodes_Vec3_Inner(H.levels[l+1], H.levels[l], xLv[l+1], ws->tmp_xf, W_local);
				// implicit barrier at end of omp for inside Inner function

				// Accumulate prolongated correction + post-smoothing + BC mask (fused)
				#pragma omp for schedule(static)
				for (int i = 0; i < fn_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						const int d = 3*i+c;
						// Accumulate prolongation
						xLv[l][d] += ws->tmp_xf[d];
						// Post-smoothing + BC mask
						if (fixedMasks[l][d]) {
							xLv[l][d] = 0.0;
						} else {
							xLv[l][d] += weight * rLv[l][d] / std::max(1.0e-30, D[d]);
						}
					}
				}
				// implicit barrier
			}

			// === EXIT: Gather back to free DOFs ===
			#pragma omp for schedule(static) nowait
			for (size_t i = 0; i < nFreeDofs; ++i) {
				int dMort = pb.freeDofIndex[i];
				int nMort = dMort / 3;
				int comp  = dMort % 3;
				int nLex  = pb.mesh.nodMapBack[nMort];
				zFree[i] = xLv[0][3*nLex + comp];
			}
		} // end parallel region
	};
}

} } // namespace top3d::mg
