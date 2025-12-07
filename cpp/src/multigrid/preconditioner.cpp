
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
	return [H, diag, cfg, &pb, fixedMasks, Lcoarse, Ncoarse, ws](const std::vector<double>& rFree, std::vector<double>& zFree) mutable {
        // Alias workspace vectors for brevity
        auto& rLv = ws->rLv;
        auto& xLv = ws->xLv;

		const int n0_nodes = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		const int n0_dofs = 3 * n0_nodes;
		const size_t nFreeDofs = pb.freeDofIndex.size();

		// Zero finest level buffers in parallel
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n0_dofs; ++i) {
			rLv[0][i] = 0.0;
			xLv[0][i] = 0.0;
		}

		// Build Morton-ordered DOF vector from free DOFs -> Lexicographic (parallel)
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < nFreeDofs; ++i) {
			int dMort = pb.freeDofIndex[i];
			int nMort = dMort / 3;
			int comp  = dMort % 3;
			int nLex = pb.mesh.nodMapBack[nMort];
			rLv[0][3*nLex + comp] = rFree[i];
		}

		// Apply BC mask on finest residual immediately (parallel)
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n0_nodes; ++i) {
			for (int c = 0; c < 3; ++c) {
				if (fixedMasks[0][3*i+c]) rLv[0][3*i+c] = 0.0;
			}
		}

		for (size_t l=0; l+1<H.levels.size(); ++l) {
			const int fn_nodes = H.levels[l].numNodes;
			const int fn_dofs = 3 * fn_nodes;
			const int cn_nodes = H.levels[l+1].numNodes;
			const int cn_dofs = 3 * cn_nodes;

			const auto& D = diag[l];
			const double w = cfg.weight;

			// Pre-smoothing: x = w * D^-1 * r (combined with zeroing)
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < fn_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					const int d = 3*i+c;
					if (fixedMasks[l][d]) {
						xLv[l][d] = 0.0;
					} else {
						xLv[l][d] = w * rLv[l][d] / std::max(1.0e-30, D[d]);
					}
				}
			}

			// Zero coarse level buffers
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < cn_dofs; ++i) {
				rLv[l+1][i] = 0.0;
				xLv[l+1][i] = 0.0;
			}

			// Restrict Residual: r_coarse = Restrict(r_fine)
			// Process all 3 components (already parallelized internally)
			for (int c = 0; c < 3; ++c) {
				MG_Restrict_nodes_Strided(H.levels[l+1], H.levels[l], rLv[l], rLv[l+1], c, 3);
			}

			// Apply BC mask on coarse residual
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < cn_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					if (fixedMasks[l+1][3*i+c]) rLv[l+1][3*i+c] = 0.0;
				}
			}
		}

		// Coarsest level solve
		const size_t Lidx = H.levels.size()-1;
		const int coarse_nodes = H.levels[Lidx].numNodes;
		if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
			chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < coarse_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					if (fixedMasks[Lidx][3*i+c]) xLv[Lidx][3*i+c] = 0.0;
				}
			}
		} else {
			const auto& D = diag[Lidx];
			#pragma omp parallel for schedule(static)
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

		// Upward pass (prolongation + post-smoothing)
		for (int l = (int)H.levels.size()-2; l >= 0; --l) {
			const int fn_nodes = H.levels[l].numNodes;
			const int cn_nodes = H.levels[l+1].numNodes;
			const auto& D = diag[l];
			const double w = cfg.weight;

			// Prolongate Correction for all 3 components
			for (int c = 0; c < 3; ++c) {
				// Copy strided xLv[l+1] to tmp_xc
				#pragma omp parallel for schedule(static)
				for (int i = 0; i < cn_nodes; ++i) {
					ws->tmp_xc[i] = xLv[l+1][3*i+c];
				}

				// Fast stencil prolongation (stride 1->1)
				MG_Prolongate_nodes_Strided(H.levels[l+1], H.levels[l], ws->tmp_xc, ws->tmp_xf, 0, 1);

				// Accumulate back
				#pragma omp parallel for schedule(static)
				for (int i = 0; i < fn_nodes; ++i) {
					xLv[l][3*i+c] += ws->tmp_xf[i];
				}
			}

			// BC mask + Post-smoothing in one pass
			#pragma omp parallel for schedule(static)
			for (int i = 0; i < fn_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					const int d = 3*i+c;
					if (fixedMasks[l][d]) {
						xLv[l][d] = 0.0;
					} else {
						xLv[l][d] += w * rLv[l][d] / std::max(1.0e-30, D[d]);
					}
				}
			}
		}

		// Convert lex solution back to Morton indexing and extract free DOFs (parallel)
		zFree.resize(rFree.size());
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < nFreeDofs; ++i) {
			int dMort = pb.freeDofIndex[i];
			int nMort = dMort / 3;
			int comp  = dMort % 3;
			int nLex  = pb.mesh.nodMapBack[nMort];
			zFree[i] = xLv[0][3*nLex + comp];
		}
	};
}

} } // namespace top3d::mg
