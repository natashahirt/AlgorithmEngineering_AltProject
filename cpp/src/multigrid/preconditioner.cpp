
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
	const int NlimitDofs = 200000;
	int adaptiveMax = ComputeAdaptiveMaxLevels(pb, cfg.nonDyadic, cfg.maxLevels, NlimitDofs);
	BuildMGHierarchy(pb, cfg.nonDyadic, H, adaptiveMax);
	// One-line debug print (enable by setting env TOP3D_MG_DEBUG to any value)
	if (std::getenv("TOP3D_MG_DEBUG")) {
		const auto& Lc = H.levels.back();
		std::cout << "[MG] levels=" << H.levels.size()
				  << " coarsest=" << Lc.resX << "x" << Lc.resY << "x" << Lc.resZ
				  << " dofs=" << (3 * Lc.numNodes) << "\n";
	}
	MG_BuildFixedMasks(pb, H, fixedMasks);
}


// Reuse H/fixedMasks; per-iteration, rebuild diagonals and assemble SIMP-modulated coarsest K
Preconditioner make_diagonal_preconditioner_from_static(const Problem& pb,
														  const MGHierarchy& H,
														  const std::vector<std::vector<uint8_t>>& fixedMasks,
														  const std::vector<float>& eleModulus,
														  const MGPrecondConfig& cfg) {
	// 1) Build per-level diagonals
	std::vector<std::vector<float>> diag;
	MG_BuildDiagonals(pb, H, fixedMasks, eleModulus, diag);

	// 2) Build aggregated Ee at coarsest level and factorize
	std::vector<float> Lcoarse; int Ncoarse = 0;
	{
		const auto& Lc = H.levels.back();
		Ncoarse = 3*Lc.numNodes;
		const int NlimitDofs = 200000;
		if (H.levels.size() == 1 || Ncoarse > NlimitDofs) {
			Ncoarse = 0; // diagonal fallback
		} else {
			// 1) Finest-grid modulus in structured order
			std::vector<float> emFineFull;
			EleMod_CompactToFull_Finest(pb, eleModulus, emFineFull);

			// 2) Coarsest dense K via Galerkin triple products with fine-level BC mask
			std::vector<float> Kc;
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

		// Convert input free vector to float for MG internals
		// std::vector<float> rFree_f(rFree.size()); // Avoid allocation if possible? workspace doesn't hold this intermediate
        // We'll just do it on the fly or add to workspace. For now let's just loop.
        
		const int n0_nodes = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		const int n0_dofs  = 3*n0_nodes;
        
        // Zero out finest residual buffer first
        std::fill(rLv[0].begin(), rLv[0].end(), 0.0f);
        
		// Build Morton-ordered DOF vector from free DOFs -> Lexicographic
        // We can go directly Free -> Lex to save a buffer if we are careful about mapping
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
             int dMort = pb.freeDofIndex[i];
             int nMort = dMort / 3;
             int comp  = dMort % 3;
             int nLex = pb.mesh.nodMapBack[nMort]; // nMort -> nLex
             rLv[0][3*nLex + comp] = static_cast<float>(rFree[i]);
        }

		// Apply BC mask on finest residual immediately (though scatter should have handled it if clean)
		for (int i=0;i<n0_nodes;i++) {
			for (int c=0;c<3;c++) if (fixedMasks[0][3*i+c]) rLv[0][3*i+c] = 0.0f;
		}

        // xLv[0] needs to be zeroed? Actually it's an output of the cycle, initialized to 0.
        std::fill(xLv[0].begin(), xLv[0].end(), 0.0f);

		for (size_t l=0; l+1<H.levels.size(); ++l) {
			const int fn_nodes = H.levels[l].numNodes;
			const int cn_nodes = H.levels[l+1].numNodes;
            
            // Ensure xLv is correct size (resize handled in setup, just safe check or fill)
            // xLv[l] is 3*fn_nodes
            std::fill(xLv[l].begin(), xLv[l].end(), 0.0f); // Pre-smoothing guess = 0 usually? Or Richardson step on 0?
            
			const auto& D = diag[l];
			// Pre-smoothing: x = D^-1 * r
            // Note: Richardson iteration x = x + w * D^-1 * (b - Ax). If initial x=0, then x = w * D^-1 * b.
			#pragma omp parallel for
			for (int i=0;i<fn_nodes;i++) {
				for (int c=0;c<3;c++) {
					const int d = 3*i+c; if (fixedMasks[l][d]) continue;
					xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1.0e-30f, D[d]);
				}
			}

            // Calculate residual for restriction?
            // Standard V-cycle: r_coarse = R * (r_fine - A * x_fine)
            // But here we might be doing a simpler update since it's just diagonal precond?
            // If x_fine was just computed as M^-1 * r_fine, the residual is r_fine - A*x_fine.
            // But this implementation seems to just restrict 'rLv' ?? 
            // The original code was:
            // xLv[l] += weight * rLv[l] / D
            // rf = rLv[l]  <-- Wait, this implies we are restricting the original residual, not the defect?
            // Ah, looking at the loop:
            // 1. Smooth: x = S(r)
            // 2. Restrict: rc = R * r ??
            // If we simply restrict r, we are NOT doing a standard MG correction unless we update r.
            // Usually: r_next = R * (r_current - A * x_current).
            // This implementation seems to be doing: x_pre = S(r); r_next = R * r; 
            // This suggests it ignores the pre-smoothing effect on the residual! 
            // That is valid for "Additive MG" or specific variants, but standard V-cycle usually updates residual.
            // HOWEVER, I will keep the logic EXACTLY as in the original file for now to avoid breaking math, just optimizing memory.
            // Original: for (int i=0;i<fn_nodes;i++) rf[i] = rLv[l][3*i+c]; -> MG_Restrict -> rLv[l+1]
            
            // We use workspace tmp buffers for component-wise restriction
            // NOTE: rLv[l+1] needs to be cleared or assigned? MG_Restrict assigns.
            // But we can optimize to use the tmp buffers.
            
            std::fill(rLv[l+1].begin(), rLv[l+1].end(), 0.0f);
            
			for (int c=0;c<3;c++) {
                // Copy strided rLv to tmp_rf
                #pragma omp parallel for
				for (int i=0;i<fn_nodes;i++) ws->tmp_rf[i] = rLv[l][3*i+c];
                
				MG_Restrict_nodes(H.levels[l+1], H.levels[l], ws->tmp_rf, ws->tmp_rc);
				
                // Copy tmp_rc to strided rLv[l+1]
                #pragma omp parallel for
				for (int i=0;i<cn_nodes;i++) rLv[l+1][3*i+c] = ws->tmp_rc[i];
			}
            
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l+1][3*i+c]) rLv[l+1][3*i+c] = 0.0f;
            
            // xLv[l+1] will be computed in next steps
            std::fill(xLv[l+1].begin(), xLv[l+1].end(), 0.0f);
		}

		const size_t Lidx = H.levels.size()-1;
		if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
			chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
			const int cn_nodes = H.levels[Lidx].numNodes;
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[Lidx][3*i+c]) xLv[Lidx][3*i+c] = 0.0f;
		} else {
			const auto& D = diag[Lidx];
			const int cn_nodes = H.levels[Lidx].numNodes;
			// xLv[Lidx] already 0
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) {
				const int d = 3*i+c; if (fixedMasks[Lidx][d]) { xLv[Lidx][d] = 0.0f; continue; }
				xLv[Lidx][d] = rLv[Lidx][d] / std::max(1.0e-30f, D[d]);
			}
		}

		for (int l=(int)H.levels.size()-2; l>=0; --l) {
			const int fn_nodes = H.levels[l].numNodes;
            
            // tmp_add needs to be zeroed? MG_Prolongate assigns, but we accumulate 3 comps.
            // Let's zero it.
            // Actually we can just accumulate into xLv[l].
            
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
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l][3*i+c]) xLv[l][3*i+c] = 0.0f;
            
            // Post-smoothing
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) {
				int d = 3*i+c; if (fixedMasks[l][d]) continue;
				xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1.0e-30f, D[d]);
			}
		}
		// Convert lex solution back to Morton indexing and extract free DOFs
		zFree.resize(rFree.size());
		for (size_t i=0;i<pb.freeDofIndex.size();++i) {
			int dMort = pb.freeDofIndex[i];
			int nMort = dMort / 3;
			int comp  = dMort % 3;
			int nLex  = pb.mesh.nodMapBack[nMort];
			zFree[i] = static_cast<double>(xLv[0][3*nLex + comp]);
		}
	};
}

} } // namespace top3d::mg
