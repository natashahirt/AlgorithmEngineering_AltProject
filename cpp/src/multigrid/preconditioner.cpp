
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
		const int NlimitDofs = 200000;
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

	// 3) Return preconditioner closure (adapt to double free-DOF vectors)
	return [H, diag, cfg, &pb, fixedMasks, Lcoarse, Ncoarse](const std::vector<double>& rFree, std::vector<double>& zFree) {
		// Convert input free vector to float for MG internals
		std::vector<double> rFree_f(rFree.size());
		for (size_t i=0;i<rFree.size();++i) rFree_f[i] = static_cast<double>(rFree[i]);
		const int n0_nodes = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		const int n0_dofs  = 3*n0_nodes;
		// Build Morton-ordered DOF vector from free DOFs
		std::vector<double> r0_morton(n0_dofs, 0.0f);
		for (size_t i=0;i<pb.freeDofIndex.size();++i) r0_morton[pb.freeDofIndex[i]] = rFree_f[i];
		// Convert to lexicographic node order expected by structured MG transfers
		std::vector<double> r0_lex(n0_dofs, 0.0f);
		for (int nLex=0; nLex<n0_nodes; ++nLex) {
			int nMort = pb.mesh.nodMapForward[nLex];
			for (int c=0;c<3;c++) r0_lex[3*nLex + c] = r0_morton[3*nMort + c];
		}

		std::vector<std::vector<double>> rLv(H.levels.size());
		std::vector<std::vector<double>> xLv(H.levels.size());
		rLv[0] = r0_lex; xLv[0].assign(n0_dofs, 0.0);

		for (int i=0;i<n0_nodes;i++) {
			for (int c=0;c<3;c++) if (fixedMasks[0][3*i+c]) rLv[0][3*i+c] = 0.0;
		}

		for (size_t l=0; l+1<H.levels.size(); ++l) {
			const int fn_nodes = H.levels[l].numNodes;
			const int cn_nodes = H.levels[l+1].numNodes;
			if ((int)xLv[l].size() != 3*fn_nodes) xLv[l].assign(3*fn_nodes, 0.0f);
			const auto& D = diag[l];
			for (int i=0;i<fn_nodes;i++) {
				for (int c=0;c<3;c++) {
					const int d = 3*i+c; if (fixedMasks[l][d]) continue;
					xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1.0e-30, D[d]);
				}
			}
			rLv[l+1].assign(3*cn_nodes, 0.0f);
			for (int c=0;c<3;c++) {
				std::vector<double> rf(fn_nodes), rc;
				for (int i=0;i<fn_nodes;i++) rf[i] = rLv[l][3*i+c];
				MG_Restrict_nodes(H.levels[l+1], H.levels[l], rf, rc);
				for (int i=0;i<cn_nodes;i++) rLv[l+1][3*i+c] = rc[i];
			}
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l+1][3*i+c]) rLv[l+1][3*i+c] = 0.0;
			xLv[l+1].assign(3*cn_nodes, 0.0f);
		}

		const size_t Lidx = H.levels.size()-1;
		if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
			chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
			const int cn_nodes = H.levels[Lidx].numNodes;
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[Lidx][3*i+c]) xLv[Lidx][3*i+c] = 0.0;
		} else {
			const auto& D = diag[Lidx];
			const int cn_nodes = H.levels[Lidx].numNodes;
			xLv[Lidx].assign(3*cn_nodes, 0.0f);
			for (int i=0;i<cn_nodes;i++) for (int c=0;c<3;c++) {
				const int d = 3*i+c; if (fixedMasks[Lidx][d]) { xLv[Lidx][d] = 0.0; continue; }
				xLv[Lidx][d] = rLv[Lidx][d] / std::max(1.0e-30, D[d]);
			}
		}

		for (int l=(int)H.levels.size()-2; l>=0; --l) {
			const int fn_nodes = H.levels[l].numNodes;
			std::vector<double> add(3*fn_nodes, 0.0f);
			for (int c=0;c<3;c++) {
				std::vector<double> xc(H.levels[l+1].numNodes), xf;
				for (int i=0;i<H.levels[l+1].numNodes;i++) xc[i] = xLv[l+1][3*i+c];
				MG_Prolongate_nodes(H.levels[l+1], H.levels[l], xc, xf);
				for (int i=0;i<fn_nodes;i++) add[3*i+c] = xf[i];
			}
			if ((int)xLv[l].size() != 3*fn_nodes) xLv[l].assign(3*fn_nodes, 0.0f);
			for (int i=0;i<3*fn_nodes;i++) xLv[l][i] += add[i];
			const auto& D = diag[l];
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) if (fixedMasks[l][3*i+c]) xLv[l][3*i+c] = 0.0;
			for (int i=0;i<fn_nodes;i++) for (int c=0;c<3;c++) {
				int d = 3*i+c; if (fixedMasks[l][d]) continue;
				xLv[l][d] += cfg.weight * rLv[l][d] / std::max(1.0e-30, D[d]);
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