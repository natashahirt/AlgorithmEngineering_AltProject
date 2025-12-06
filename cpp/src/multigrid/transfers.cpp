// Transfer operators (prolongation and restriction)
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

// Optional OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace top3d { namespace mg {

// ===== Optimized Transfer Operators (No Atomics) =====

// Gather-based Prolongation (Interpolation)
// Parallelizes over FINE nodes (output), avoiding race conditions.
void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
								  const std::vector<double>& xc, std::vector<double>& xf) {
	const int fnnx = Lf.resX+1, fnny = Lf.resY+1, fnnz = Lf.resZ+1;
const int span = Lc.spanWidth;
	const int grid = span+1;

    // Output size check handled by caller usually, but let's be safe
	if (xf.size() != (size_t)(fnnx*fnny*fnnz)) xf.resize(fnnx*fnny*fnnz);

    // Helper for Coarse Index
    // Note: The original code uses `Lc.resY - cey` for Y index.
    // idxC(cex, Y_index, cez)
	const int cnnx = Lc.resX+1, cnny = Lc.resY+1;
	auto idxC_raw = [&](int cx, int cy, int cz){ return cnnx*cnny*cz + cnnx*cy + cx; };

    // Parallelize over Fine Nodes
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
	for (int fiz = 0; fiz < fnnz; ++fiz) {
		for (int fiy = 0; fiy < fnny; ++fiy) {
			for (int fix = 0; fix < fnnx; ++fix) {
                
                // 1. Identify "Canonical" Coarse Element (cex, cey, cez) and local offsets (ix, iy, iz)
                // We clamp to the valid element range to handle boundaries correctly.
                
                // X
                int cex = fix / span;
                int ix  = fix % span;
                if (cex >= Lc.resX) { cex = Lc.resX - 1; ix = span; }

                // Z
                int cez = fiz / span;
                int iz  = fiz % span;
                if (cez >= Lc.resZ) { cez = Lc.resZ - 1; iz = span; }
                
                // Y - Inverted logic from original code
                // fy ranges 0 .. resY * span
                // inv_fy = resY * span - fy
                // cey = inv_fy / span
                // iy = inv_fy % span
                int inv_fy = Lc.resY * span - fiy;
                int cey = inv_fy / span;
                int iy  = inv_fy % span;
                if (cey >= Lc.resY) { cey = Lc.resY - 1; iy = span; }
                
                // 2. Retrieve Weights for local position (ix, iy, iz)
                // weightsNode is stored as [((iz*grid + iy)*grid + ix)*8 + a]
                const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
                
                // 3. Retrieve Coarse Node Values
                // Map the 8 corners of element (cex, cey, cez) to global coarse indices
                // Original cidx order:
                // 0: (cex,   resY-cey,   cez)
                // 1: (cex+1, resY-cey,   cez)
                // 2: (cex+1, resY-cey-1, cez)
                // 3: (cex,   resY-cey-1, cez)
                // 4: (cex,   resY-cey,   cez+1)
                // 5: (cex+1, resY-cey,   cez+1)
                // 6: (cex+1, resY-cey-1, cez+1)
                // 7: (cex,   resY-cey-1, cez+1)
                
                int cy_top = Lc.resY - cey;     // Higher Y index
                int cy_bot = Lc.resY - cey - 1; // Lower Y index
                
                double cvals[8];
                cvals[0] = xc[idxC_raw(cex,   cy_top, cez)];
                cvals[1] = xc[idxC_raw(cex+1, cy_top, cez)];
                cvals[2] = xc[idxC_raw(cex+1, cy_bot, cez)];
                cvals[3] = xc[idxC_raw(cex,   cy_bot, cez)];
                cvals[4] = xc[idxC_raw(cex,   cy_top, cez+1)];
                cvals[5] = xc[idxC_raw(cex+1, cy_top, cez+1)];
                cvals[6] = xc[idxC_raw(cex+1, cy_bot, cez+1)];
                cvals[7] = xc[idxC_raw(cex,   cy_bot, cez+1)];
                
                // 4. Interpolate
                double sum = 0.0;
                // #pragma omp simd reduction(+:sum) // Hint for vectorization
                for (int a=0; a<8; ++a) {
                    sum += W[a] * cvals[a];
                }
                
                // 5. Write
                int fidx = fnnx*fnny*fiz + fnnx*fiy + fix;
                xf[fidx] = sum;
            }
        }
    }
}

// Gather-based Restriction
// Parallelizes over COARSE nodes (output), gathering contributions from Fine neighbors.
void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
								 const std::vector<double>& rf, std::vector<double>& rc) {
	const int cnnx = Lc.resX+1, cnny = Lc.resY+1, cnnz = Lc.resZ+1;
    const int fnnx = Lf.resX+1, fnny = Lf.resY+1; // fnnz unused
    const int span = Lc.spanWidth;
	const int grid = span+1;

    if (rc.size() != (size_t)(cnnx*cnny*cnnz)) rc.resize(cnnx*cnny*cnnz);

    auto idxF_raw = [&](int fx, int fy, int fz){ return fnnx*fnny*fz + fnnx*fy + fx; };

#ifdef _OPENMP
	#pragma omp parallel for collapse(3)
#endif
	for (int cz = 0; cz < cnnz; ++cz) {
		for (int cy = 0; cy < cnny; ++cy) {
			for (int cx = 0; cx < cnnx; ++cx) {
                
                double sum = 0.0;
                double wsum = 0.0;

                // Iterate over the 8 coarse elements (octants) sharing this node
                // oz: -1 (Back relative to node), 0 (Front relative to node)
                // ox: -1 (Left), 0 (Right)
                // oy_offset: Determine cey. 
                // cy = resY - cey (Top of elem) OR cy = resY - cey - 1 (Bottom of elem)
                // If Top: cey = resY - cy. (oy_offset = 0)
                // If Bottom: cey = resY - cy - 1. (oy_offset = -1)
                
                for (int oz = -1; oz <= 0; ++oz) {
                    int cez = cz + oz;
                    if (cez < 0 || cez >= Lc.resZ) continue;
                    
                    for (int ox = -1; ox <= 0; ++ox) {
                        int cex = cx + ox;
                        if (cex < 0 || cex >= Lc.resX) continue;
                        
                        for (int oy_off = -1; oy_off <= 0; ++oy_off) {
                            int cey = Lc.resY - cy + oy_off;
                            if (cey < 0 || cey >= Lc.resY) continue;
                            
                            // Valid coarse element E(cex, cey, cez) found.
                            // Determine which corner `a` corresponds to C(cx, cy, cz).
                            // Based on offsets:
                            // ox=-1 (Right of E), ox=0 (Left of E)
                            // oz=-1 (Back of E), oz=0 (Front of E)
                            // oy_off=0 (cy is Top), oy_off=-1 (cy is Bottom)
                            
                            // Map to original corner indexing:
                            // 0: L, T, F
                            // 1: R, T, F
                            // 2: R, B, F
                            // 3: L, B, F
                            // 4: L, T, B
                            // 5: R, T, B
                            // 6: R, B, B
                            // 7: L, B, B
                            
                            bool is_right  = (ox == -1); // Node is right of elem
                            bool is_bottom = (oy_off == -1); // Node is bottom of elem
                            bool is_back   = (oz == -1); // Node is back of elem
                            
                            int a = 0;
                            if (is_back) a += 4;
                            // Front/Back group established.
                            // Within group: 0=LT, 1=RT, 2=RB, 3=LB
                            if (!is_bottom) { // Top
                                if (!is_right) a += 0; // LT
                                else           a += 1; // RT
                            } else { // Bottom
                                if (is_right)  a += 2; // RB
                                else           a += 3; // LB
                            }

                            // Accumulate from all fine nodes in this coarse element
                            // Element E covers fine nodes:
                            // fx from cex*span to cex*span + span
                            // fz from cez*span to cez*span + span
                            // fy ... Y is inverted.
                            // fy top (iy=0): (resY - cey)*span
                            // fy bot (iy=span): (resY - cey)*span - span
                            
                            int fx0 = cex * span;
                            int fz0 = cez * span;
                            int fy_top = (Lc.resY - cey) * span; // Corresponds to iy=0
                            
                            for (int iz = 0; iz <= span; ++iz) {
                                for (int iy = 0; iy <= span; ++iy) {
                                    for (int ix = 0; ix <= span; ++ix) {
                                        double w = Lc.weightsNode[((iz*grid+iy)*grid+ix)*8 + a];
                                        if (w == 0.0) continue;
                                        
                                        int fix = fx0 + ix;
                                        int fiz = fz0 + iz;
                                        int fiy = fy_top - iy; 
                                        
                                        double val = rf[idxF_raw(fix, fiy, fiz)];
                                        sum += w * val;
                                        wsum += w;
                                    }
                                }
                            }
                        }
                    }
                }
                
                int cidx = cnnx*cnny*cz + cnnx*cy + cx;
                rc[cidx] = (wsum > 1e-12) ? (sum / wsum) : 0.0;
            }
        }
    }
}

} } // namespace top3d::mg
