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
    // 1. Ensure Output Size
    size_t numCoarseNodes = (size_t)(Lc.resX + 1) * (Lc.resY + 1) * (Lc.resZ + 1);
    if (rc.size() != numCoarseNodes * stride) {
        rc.resize(numCoarseNodes * stride);
    }

    // Pointers for raw access
    const double* __restrict__ rf_ptr = rf.data();
    double* __restrict__ rc_ptr = rc.data();

    const int cNnx = Lc.resX + 1;
    const int cNny = Lc.resY + 1;
    const int cNnz = Lc.resZ + 1;

    const int fNnx = Lf.resX + 1;
    const int fNny = Lf.resY + 1;
    const int fNnz = Lf.resZ + 1;

    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    // 2. Precompute 1D Weights (The "Hat" Function)
    // Offset range: -span+1 to +span-1
    int wSize = 2 * span - 1;
    std::vector<double> W(wSize);
    for (int i = 0; i < wSize; ++i) {
        // Map index i to offset: i - (span - 1)
        W[i] = 1.0 - std::abs(i - (span - 1)) * invSpan;
    }

    // 3. Parallel Restriction
#pragma omp parallel for collapse(2) schedule(static)
    for (int cz = 0; cz < cNnz; ++cz) {
        for (int cy = 0; cy < cNny; ++cy) {

            // Pre-calculate Z and Y bounds for the fine grid
            int fz0 = cz * span;
            int fy0 = cy * span;

            // Inner Loop over Coarse X
            for (int cx = 0; cx < cNnx; ++cx) {
                int fx0 = cx * span;

                double valSum = 0.0;
                double wSum = 0.0;

                // --- 3D Stencil Loop (Unrolled via loop bounds) ---
                // We clamp the loops to the intersection of the Kernel and the Grid
                // to avoid "if" checks inside the innermost loop.

                int iz_start = (fz0 - span + 1 < 0) ? (span - 1 - fz0) : 0;
                int iz_end   = (fz0 + span - 1 >= fNnz) ? (wSize - (fz0 + span - fNnz)) : wSize;

                for (int iz = iz_start; iz < iz_end; ++iz) {
                    double wz = W[iz];
                    int fz = fz0 + (iz - (span - 1));
                    size_t zOffset = (size_t)fz * fNny * fNnx;

                    int iy_start = (fy0 - span + 1 < 0) ? (span - 1 - fy0) : 0;
                    int iy_end   = (fy0 + span - 1 >= fNny) ? (wSize - (fy0 + span - fNny)) : wSize;

                    for (int iy = iy_start; iy < iy_end; ++iy) {
                        double wy = W[iy];
                        double wzy = wz * wy;
                        size_t yOffset = (size_t)(fy0 + (iy - (span - 1))) * fNnx;

                        // X is the innermost, optimizing this is key
                        int ix_start = (fx0 - span + 1 < 0) ? (span - 1 - fx0) : 0;
                        int ix_end   = (fx0 + span - 1 >= fNnx) ? (wSize - (fx0 + span - fNnx)) : wSize;

#pragma omp simd reduction(+:valSum, wSum)
                        for (int ix = ix_start; ix < ix_end; ++ix) {
                            double wx = W[ix];
                            double w = wzy * wx;

                            int fx = fx0 + (ix - (span - 1));

                            // Direct array access: z*Ny*Nx + y*Nx + x
                            size_t fIdx = zOffset + yOffset + fx;

                            valSum += rf_ptr[fIdx * stride + component] * w;
                            wSum   += w;
                        }
                    }
                }

                // Write Result
                size_t cIdx = (size_t)cz * cNny * cNnx + cy * cNnx + cx;
                if (wSum > 1e-12) rc_ptr[cIdx * stride + component] = valSum / wSum;
                else              rc_ptr[cIdx * stride + component] = 0.0;
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
