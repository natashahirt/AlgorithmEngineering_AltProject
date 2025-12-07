
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
	// Use raw pointers to avoid std::vector operator[] bounds checking overhead
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

		// Pre-extract raw pointers for hot paths
		const int* __restrict__ freeDofIdx = pb.freeDofIndex.data();
		const int* __restrict__ nodMapBack = pb.mesh.nodMapBack.data();
		const double* __restrict__ rFree_ptr = rFree.data();
		double* __restrict__ zFree_ptr = zFree.data();
		double* __restrict__ tmp_xf_ptr = ws->tmp_xf.data();

		// ============ SINGLE PARALLEL REGION FOR ENTIRE V-CYCLE ============
		#pragma omp parallel
		{
			// Thread-local weight buffer for transfer operations
			std::vector<double> W_local(16);

			// Get raw pointers for level 0
			double* __restrict__ rLv0 = rLv[0].data();
			double* __restrict__ xLv0 = xLv[0].data();
			const uint8_t* __restrict__ mask0 = fixedMasks[0].data();

			// === ENTRY: Zero finest + scatter from free DOFs ===
			#pragma omp for schedule(static) nowait
			for (int i = 0; i < n0_dofs; ++i) {
				rLv0[i] = 0.0;
				xLv0[i] = 0.0;
			}

			#pragma omp barrier

			#pragma omp for schedule(static) nowait
			for (size_t i = 0; i < nFreeDofs; ++i) {
				int dMort = freeDofIdx[i];
				int nMort = dMort / 3;
				int comp  = dMort % 3;
				int nLex = nodMapBack[nMort];
				rLv0[3*nLex + comp] = rFree_ptr[i];
			}

			#pragma omp barrier

			#pragma omp for schedule(static)
			for (int i = 0; i < n0_nodes; ++i) {
				for (int c = 0; c < 3; ++c) {
					if (mask0[3*i+c]) rLv0[3*i+c] = 0.0;
				}
			}

			// === DOWNWARD PASS ===
			for (size_t l = 0; l + 1 < numLevels; ++l) {
				const int fn_nodes = H.levels[l].numNodes;
				const int cn_nodes = H.levels[l+1].numNodes;
				const int cn_dofs = 3 * cn_nodes;
				const double* __restrict__ D_ptr = diag[l].data();
				const uint8_t* __restrict__ maskL = fixedMasks[l].data();
				const uint8_t* __restrict__ maskL1 = fixedMasks[l+1].data();
				double* __restrict__ rL = rLv[l].data();
				double* __restrict__ xL = xLv[l].data();
				double* __restrict__ rL1 = rLv[l+1].data();
				double* __restrict__ xL1 = xLv[l+1].data();

				// Pre-smoothing
				#pragma omp for schedule(static) nowait
				for (int i = 0; i < fn_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						const int d = 3*i+c;
						if (maskL[d]) {
							xL[d] = 0.0;
						} else {
							xL[d] = weight * rL[d] / std::max(1.0e-30, D_ptr[d]);
						}
					}
				}

				// Zero coarse level
				#pragma omp for schedule(static)
				for (int i = 0; i < cn_dofs; ++i) {
					rL1[i] = 0.0;
					xL1[i] = 0.0;
				}

				// Restriction
				MG_Restrict_nodes_Vec3_Inner(H.levels[l+1], H.levels[l], rLv[l], rLv[l+1], W_local);

				// BC mask on coarse
				#pragma omp for schedule(static)
				for (int i = 0; i < cn_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						if (maskL1[3*i+c]) rL1[3*i+c] = 0.0;
					}
				}
			}

			// === COARSEST LEVEL SOLVE ===
			#pragma omp single
			{
				if (!Lcoarse.empty() && (int)rLv[Lidx].size() == Ncoarse) {
					chol_solve_lower(Lcoarse, rLv[Lidx], xLv[Lidx], Ncoarse);
				} else {
					const double* __restrict__ D_ptr = diag[Lidx].data();
					const uint8_t* __restrict__ maskC = fixedMasks[Lidx].data();
					double* __restrict__ rC = rLv[Lidx].data();
					double* __restrict__ xC = xLv[Lidx].data();
					for (int i = 0; i < coarse_nodes; ++i) {
						for (int c = 0; c < 3; ++c) {
							const int d = 3*i+c;
							if (maskC[d]) {
								xC[d] = 0.0;
							} else {
								xC[d] = rC[d] / std::max(1.0e-30, D_ptr[d]);
							}
						}
					}
				}
			}

			// BC mask on coarsest
			{
				const uint8_t* __restrict__ maskC = fixedMasks[Lidx].data();
				double* __restrict__ xC = xLv[Lidx].data();
				#pragma omp for schedule(static)
				for (int i = 0; i < coarse_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						if (maskC[3*i+c]) xC[3*i+c] = 0.0;
					}
				}
			}

			// === UPWARD PASS ===
			for (int l = (int)numLevels - 2; l >= 0; --l) {
				const int fn_nodes = H.levels[l].numNodes;
				const double* __restrict__ D_ptr = diag[l].data();
				const uint8_t* __restrict__ maskL = fixedMasks[l].data();
				double* __restrict__ rL = rLv[l].data();
				double* __restrict__ xL = xLv[l].data();

				// Prolongation
				MG_Prolongate_nodes_Vec3_Inner(H.levels[l+1], H.levels[l], xLv[l+1], ws->tmp_xf, W_local);

				// Accumulate + post-smooth + BC mask
				#pragma omp for schedule(static)
				for (int i = 0; i < fn_nodes; ++i) {
					for (int c = 0; c < 3; ++c) {
						const int d = 3*i+c;
						xL[d] += tmp_xf_ptr[d];
						if (maskL[d]) {
							xL[d] = 0.0;
						} else {
							xL[d] += weight * rL[d] / std::max(1.0e-30, D_ptr[d]);
						}
					}
				}
			}

			// === EXIT: Gather back to free DOFs ===
			#pragma omp for schedule(static) nowait
			for (size_t i = 0; i < nFreeDofs; ++i) {
				int dMort = freeDofIdx[i];
				int nMort = dMort / 3;
				int comp  = dMort % 3;
				int nLex  = nodMapBack[nMort];
				zFree_ptr[i] = xLv0[3*nLex + comp];
			}
		} // end parallel region
	};
}

} } // namespace top3d::mg
