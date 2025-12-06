
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
	const int NlimitDofs = 2000;
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
		const int NlimitDofs = 2000; // Match build limit
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
        
        // Zero out finest residual buffer first
        std::fill(rLv[0].begin(), rLv[0].end(), 0.0);
        
		// Build Morton-ordered DOF vector from free DOFs -> Lexicographic
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
             int dMort = pb.freeDofIndex[i];
             int nMort = dMort / 3;
             int comp  = dMort % 3;
             int nLex = pb.mesh.nodMapBack[nMort]; // nMort -> nLex
             rLv[0][3*nLex + comp] = rFree[i];
        }

		// Apply BC mask on finest residual immediately
		for (int i=0;i<n0_nodes;i++) {
			for (int c=0;c<3;c++) if (fixedMasks[0][3*i+c]) rLv[0][3*i+c] = 0.0;
		}

        // Initialize solution guess to 0 (or pre-smoothed)
        std::fill(xLv[0].begin(), xLv[0].end(), 0.0);

		for (size_t l=0; l+1<H.levels.size(); ++l) {
			const int fn_nodes = H.levels[l].numNodes;
			const int cn_nodes = H.levels[l+1].numNodes;
            
            // xLv[l] is used to accumulate the pre-smoothing correction
            std::fill(xLv[l].begin(), xLv[l].end(), 0.0); 
            
			const auto& D = diag[l];
			// Pre-smoothing: x = w * D^-1 * r
            // (Richardson iteration with initial x=0)
			#pragma omp parallel for
			for (int i=0;i<fn_nodes;i++) {
				for (int c=0;c<3;c++) {
					const int d = 3*i+c; if (fixedMasks[l][d]) continue;
					xLv[l][d] = cfg.weight * rLv[l][d] / std::max(1.0e-30, D[d]);
				}
			}

            // Restrict Residual: r_coarse = Restrict(r_fine)
            // Note: "Adapted" V-cycle in TOP3D_XL restricts the *original* residual, NOT the defect (r - Ax).
            // This is an "Additive" correction approach.
            
            std::fill(rLv[l+1].begin(), rLv[l+1].end(), 0.0);
            
			for (int c=0;c<3;c++) {
                // Copy strided rLv to tmp_rf
                #pragma omp parallel for
				for (int i=0;i<fn_nodes;i++) ws->tmp_rf[i] = rLv[l][3*i+c];
                
				MG_Restrict_nodes(H.levels[l+1], H.levels[l], ws->tmp_rf, ws->tmp_rc);
				
                // Copy tmp_rc to strided rLv[l+1]
                #pragma omp parallel for
				for (int i=0;i<cn_nodes;i++) rLv[l+1][3*i+c] = ws->tmp_rc[i];
			}
            
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l+1][3*i+c]) rLv[l+1][3*i+c] = 0.0;
            
            // xLv[l+1] will be computed in next steps (init to 0 for safety)
            std::fill(xLv[l+1].begin(), xLv[l+1].end(), 0.0);
		}

		const size_t Lidx = H.levels.size()-1;
		if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
			chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
			const int cn_nodes = H.levels[Lidx].numNodes;
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[Lidx][3*i+c]) xLv[Lidx][3*i+c] = 0.0;
		} else {
			const auto& D = diag[Lidx];
			const int cn_nodes = H.levels[Lidx].numNodes;
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) {
				const int d = 3*i+c; if (fixedMasks[Lidx][d]) { xLv[Lidx][d] = 0.0; continue; }
				xLv[Lidx][d] = rLv[Lidx][d] / std::max(1.0e-30, D[d]);
			}
		}

		for (int l=(int)H.levels.size()-2; l>=0; --l) {
			const int fn_nodes = H.levels[l].numNodes;
            
            // Prolongate Correction: x_fine = x_fine + Prolongate(x_coarse)
			for (int c=0;c<3;c++) {
                // Copy strided xLv[l+1] to tmp_xc
                #pragma omp parallel for
				for (int i=0;i<H.levels[l+1].numNodes;i++) ws->tmp_xc[i] = xLv[l+1][3*i+c];
				
				MG_Prolongate_nodes(H.levels[l+1], H.levels[l], ws->tmp_xc, ws->tmp_xf);
				
                // Accumulate back
                #pragma omp parallel for
				for (int i=0;i<fn_nodes;i++) xLv[l][3*i+c] += ws->tmp_xf[i];
			}
            
			const auto& D = diag[l];
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l][3*i+c]) xLv[l][3*i+c] = 0.0;
            
            // Post-smoothing (Additive): x = x + w * D^-1 * r
            // Note: Uses original 'rLv[l]', not updated residual.
			#pragma omp parallel for
			for (int i=0;i<fn_nodes;i++) {
				for (int c=0;c<3;c++) {
					const int d = 3*i+c; if (fixedMasks[l][d]) continue;
					xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1.0e-30, D[d]);
				}
			}
		}
        
		// Convert lex solution back to Morton indexing and extract free DOFs
		zFree.resize(rFree.size());
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
			int dMort = pb.freeDofIndex[i];
			int nMort = dMort / 3;
			int comp  = dMort % 3;
			int nLex  = pb.mesh.nodMapBack[nMort];
			zFree[i] = xLv[0][3*nLex + comp];
		}
	};
}

} } // namespace top3d::mg
