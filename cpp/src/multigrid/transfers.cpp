// Transfer operators (prolongation and restriction)
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include "multigrid/detail/transfers.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace top3d { namespace mg {

// Optimized Strided Prolongation
// Interpolates from Coarse (xc) to Fine (xf) using linear basis
void MG_Prolongate_nodes_Strided(const MGLevel& Lc, const MGLevel& Lf,
                                 const std::vector<double>& xc, std::vector<double>& xf,
                                 int component, int stride) {
    const int cResX = Lc.resX;
    const int cResY = Lc.resY;
    const int cResZ = Lc.resZ;
    const int span  = Lc.spanWidth;

    const int fNnx = Lf.resX + 1;
    const int fNny = Lf.resY + 1;
    const int fNnz = Lf.resZ + 1;

    // Safety resize
    size_t required = (size_t)fNnx * fNny * fNnz * stride;
    if (xf.size() < required) xf.resize(required);

    const int cNnx = cResX + 1;
    const int cNny = cResY + 1;

    // 1. Precompute weights for the span (Small LUT)
    // We only need weights for offsets 0..span-1.
    // At offset=span, it belongs to the next cell.
    std::vector<double> W(span);
    double invSpan = 1.0 / span;
    for(int i=0; i<span; ++i) W[i] = (double)i * invSpan;

    // 2. Iterate over COARSE elements (Tiling the fine grid)
#pragma omp parallel for collapse(2)
    for (int cz = 0; cz < cResZ; ++cz) {
        for (int cy = 0; cy < cResY; ++cy) {
            // Inner loop logic is now invariant for the whole row
            for (int cx = 0; cx < cResX; ++cx) {

                // Base indices for the 8 corners of this coarse cell
                // cIdx = (cz * cNny + cy) * cNnx + cx;
                int cIdx000 = (cz * cNny + cy) * cNnx + cx;
                int cIdx100 = cIdx000 + 1;
                int cIdx010 = cIdx000 + cNnx;
                int cIdx110 = cIdx010 + 1;

                int cIdx001 = cIdx000 + cNnx * cNny;
                int cIdx101 = cIdx001 + 1;
                int cIdx011 = cIdx001 + cNnx;
                int cIdx111 = cIdx011 + 1;

                // Load coarse values into registers
                double v000 = xc[cIdx000 * stride + component];
                double v100 = xc[cIdx100 * stride + component];
                double v010 = xc[cIdx010 * stride + component];
                double v110 = xc[cIdx110 * stride + component];
                double v001 = xc[cIdx001 * stride + component];
                double v101 = xc[cIdx101 * stride + component];
                double v011 = xc[cIdx011 * stride + component];
                double v111 = xc[cIdx111 * stride + component];

                // 3. Fill the fine nodes inside this coarse cell
                int fStartZ = cz * span;
                int fStartY = cy * span;
                int fStartX = cx * span;

                for (int rz = 0; rz < span; ++rz) {
                    double wz = W[rz];
                    double wz0 = 1.0 - wz;

                    int fz = fStartZ + rz;
                    // Precompute Z-row offset
                    int fRowZ = fz * fNny * fNnx;

                    for (int ry = 0; ry < span; ++ry) {
                        double wy = W[ry];
                        double wy0 = 1.0 - wy;

                        // Interpolate along Z-Y planes
                        double val00 = v000 * wz0 + v001 * wz;
                        double val10 = v100 * wz0 + v101 * wz;
                        double val01 = v010 * wz0 + v011 * wz;
                        double val11 = v110 * wz0 + v111 * wz;

                        double val0 = val00 * wy0 + val01 * wy; // x=0 side
                        double val1 = val10 * wy0 + val11 * wy; // x=1 side

                        int fy = fStartY + ry;
                        int fRowY = fy * fNnx;

                        // Vectorizable inner loop
#pragma omp simd
                        for (int rx = 0; rx < span; ++rx) {
                            double wx = W[rx];
                            double val = val0 * (1.0 - wx) + val1 * wx;

                            int fx = fStartX + rx;
                            xf[(fRowZ + fRowY + fx) * stride + component] = val;
                        }
                    }
                }
            }
        }
    }

    // Note: This leaves the last boundary layer (fx=Max, fy=Max, etc) untouched.
    // In MG, boundaries are usually fixed (Dirichlet = 0) or handled separately.
    // If you need exact boundary values, you can run a simple loop for the edges.
}

// Optimized Strided Restriction
// Scatters from Fine (rf) to Coarse (rc) using linear hat function (weighted average)
// Parallelizes over COARSE nodes to avoid atomics
void MG_Restrict_nodes_Strided(const MGLevel& Lc, const MGLevel& Lf,
                               const std::vector<double>& rf, std::vector<double>& rc,
                               int component, int stride) {
    const int cnnx = Lc.resX + 1;
    const int cnny = Lc.resY + 1;
    const int cnnz = Lc.resZ + 1;
    
    const int fnnx = Lf.resX + 1;
    const int fnny = Lf.resY + 1;
    const int fnnz = Lf.resZ + 1;
    
    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    if (rc.size() < (size_t)(cnnx * cnny * cnnz * stride)) 
        rc.resize(cnnx * cnny * cnnz * stride);

    #pragma omp parallel for collapse(3)
    for (int cz = 0; cz < cnnz; ++cz) {
        for (int cy = 0; cy < cnny; ++cy) {
            for (int cx = 0; cx < cnnx; ++cx) {
                // Center of the hat function in fine coords
                int fx0 = cx * span;
                int fy0 = cy * span;
                int fz0 = cz * span;

                double sum = 0.0;
                double wsum = 0.0;

                // Iterate fine nodes in the support [-span+1, span-1]
                // Weight is 0 at +/- span, so we can exclude those bounds for efficiency
                // Clip bounds to fine grid limits
                
                int iz_min = std::max(-span + 1, -fz0);
                int iz_max = std::min(span - 1, fnnz - 1 - fz0);
                
                int iy_min = std::max(-span + 1, -fy0);
                int iy_max = std::min(span - 1, fnny - 1 - fy0);
                
                int ix_min = std::max(-span + 1, -fx0);
                int ix_max = std::min(span - 1, fnnx - 1 - fx0);

                for (int iz = iz_min; iz <= iz_max; ++iz) {
                    double wz = 1.0 - std::abs(iz) * invSpan;
                    int fz = fz0 + iz;
                    int fz_offset = fnnx * fnny * fz;
                    
                    for (int iy = iy_min; iy <= iy_max; ++iy) {
                        double wy = 1.0 - std::abs(iy) * invSpan;
                        double wzy = wz * wy;
                        int fy = fy0 + iy;
                        int fy_offset = fnnx * fy;
                        
                        for (int ix = ix_min; ix <= ix_max; ++ix) {
                            double wx = 1.0 - std::abs(ix) * invSpan;
                            double w = wzy * wx;
                            
                            if (w > 1e-9) {
                                int fx = fx0 + ix;
                                int fidx = fz_offset + fy_offset + fx;
                                sum += rf[fidx * stride + component] * w;
                                wsum += w;
                            }
                        }
                    }
                }

                int cidx = cnnx * cnny * cz + cnnx * cy + cx;
                rc[cidx * stride + component] = (wsum > 1e-12) ? (sum / wsum) : 0.0;
            }
        }
    }
}

// Wrapper wrappers for backward compatibility
void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
                         const std::vector<double>& xc, std::vector<double>& xf) {
    MG_Prolongate_nodes_Strided(Lc, Lf, xc, xf, 0, 1);
}

void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
                       const std::vector<double>& rf, std::vector<double>& rc) {
    MG_Restrict_nodes_Strided(Lc, Lf, rf, rc, 0, 1);
}

} } // namespace top3d::mg
