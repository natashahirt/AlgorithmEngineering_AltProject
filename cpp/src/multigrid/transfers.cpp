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
    const int fnnx = Lf.resX + 1;
    const int fnny = Lf.resY + 1;
    const int fnnz = Lf.resZ + 1;
    
    const int cnnx = Lc.resX + 1;
    const int cnny = Lc.resY + 1;
    
    const int span = Lc.spanWidth;
    // Inverse span for weight calculation
    const double invSpan = 1.0 / span;

    if (xf.size() < (size_t)(fnnx * fnny * fnnz * stride)) 
        xf.resize(fnnx * fnny * fnnz * stride);

    #pragma omp parallel for collapse(3)
    for (int fz = 0; fz < fnnz; ++fz) {
        for (int fy = 0; fy < fnny; ++fy) {
            for (int fx = 0; fx < fnnx; ++fx) {
                // Coarse cell index (top-left-front)
                int cx = fx / span;
                int cy = fy / span;
                int cz = fz / span;

                // Local offset in coarse cell [0, span]
                int rx = fx % span;
                int ry = fy % span;
                int rz = fz % span;

                // If on boundary, clamp to valid cell
                if (cx >= Lc.resX) { cx = Lc.resX - 1; rx = span; }
                if (cy >= Lc.resY) { cy = Lc.resY - 1; ry = span; }
                if (cz >= Lc.resZ) { cz = Lc.resZ - 1; rz = span; }

                // Trilinear weights for the 8 corners of the coarse cell
                // w = (1 - |dist|/S)
                // For local offset r in [0, S]: 
                //   Left/Top/Front weight = (S - r)/S
                //   Right/Bot/Back weight = r/S
                double wx1 = (double)rx * invSpan; double wx0 = 1.0 - wx1;
                double wy1 = (double)ry * invSpan; double wy0 = 1.0 - wy1;
                double wz1 = (double)rz * invSpan; double wz0 = 1.0 - wz1;

                // Coarse Node Indices
                // Ordering: z * (NX*NY) + y * NX + x
                int cbase00 = cnnx * cnny * cz + cnnx * cy + cx;
                int cbase01 = cbase00 + 1;       // +x
                int cbase10 = cbase00 + cnnx;    // +y
                int cbase11 = cbase10 + 1;       // +x+y
                
                int cnext00 = cbase00 + cnnx * cnny; // +z
                int cnext01 = cnext00 + 1;
                int cnext10 = cnext00 + cnnx;
                int cnext11 = cnext10 + 1;

                double val = 
                    xc[cbase00 * stride + component] * wx0 * wy0 * wz0 +
                    xc[cbase01 * stride + component] * wx1 * wy0 * wz0 +
                    xc[cbase10 * stride + component] * wx0 * wy1 * wz0 +
                    xc[cbase11 * stride + component] * wx1 * wy1 * wz0 +
                    xc[cnext00 * stride + component] * wx0 * wy0 * wz1 +
                    xc[cnext01 * stride + component] * wx1 * wy0 * wz1 +
                    xc[cnext10 * stride + component] * wx0 * wy1 * wz1 +
                    xc[cnext11 * stride + component] * wx1 * wy1 * wz1;

                // Write to fine grid
                int fidx = fnnx * fnny * fz + fnnx * fy + fx;
                xf[fidx * stride + component] = val;
            }
        }
    }
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

// Fused 3-component prolongation: processes all xyz in one parallel region
// Iterates over ALL fine nodes (matching MG_Prolongate_nodes_Strided)
void MG_Prolongate_nodes_Vec3(const MGLevel& Lc, const MGLevel& Lf,
                              const std::vector<double>& xc, std::vector<double>& xf) {
    const int fNnx = Lf.resX + 1;
    const int fNny = Lf.resY + 1;
    const int fNnz = Lf.resZ + 1;

    const int cNnx = Lc.resX + 1;
    const int cNny = Lc.resY + 1;

    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    // Ensure output size (3 components per node)
    size_t required = (size_t)fNnx * fNny * fNnz * 3;
    if (xf.size() < required) xf.resize(required);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int fz = 0; fz < fNnz; ++fz) {
        for (int fy = 0; fy < fNny; ++fy) {
            for (int fx = 0; fx < fNnx; ++fx) {
                // Coarse cell index (top-left-front)
                int cx = fx / span;
                int cy = fy / span;
                int cz = fz / span;

                // Local offset in coarse cell [0, span]
                int rx = fx % span;
                int ry = fy % span;
                int rz = fz % span;

                // If on boundary, clamp to valid cell
                if (cx >= Lc.resX) { cx = Lc.resX - 1; rx = span; }
                if (cy >= Lc.resY) { cy = Lc.resY - 1; ry = span; }
                if (cz >= Lc.resZ) { cz = Lc.resZ - 1; rz = span; }

                // Trilinear weights
                double wx1 = (double)rx * invSpan; double wx0 = 1.0 - wx1;
                double wy1 = (double)ry * invSpan; double wy0 = 1.0 - wy1;
                double wz1 = (double)rz * invSpan; double wz0 = 1.0 - wz1;

                // Coarse Node Indices
                int cbase00 = cNnx * cNny * cz + cNnx * cy + cx;
                int cbase01 = cbase00 + 1;       // +x
                int cbase10 = cbase00 + cNnx;    // +y
                int cbase11 = cbase10 + 1;       // +x+y

                int cnext00 = cbase00 + cNnx * cNny; // +z
                int cnext01 = cnext00 + 1;
                int cnext10 = cnext00 + cNnx;
                int cnext11 = cnext10 + 1;

                // Write to fine grid
                size_t fIdx = ((size_t)fNnx * fNny * fz + fNnx * fy + fx) * 3;

                for (int c = 0; c < 3; ++c) {
                    double val =
                        xc[cbase00 * 3 + c] * wx0 * wy0 * wz0 +
                        xc[cbase01 * 3 + c] * wx1 * wy0 * wz0 +
                        xc[cbase10 * 3 + c] * wx0 * wy1 * wz0 +
                        xc[cbase11 * 3 + c] * wx1 * wy1 * wz0 +
                        xc[cnext00 * 3 + c] * wx0 * wy0 * wz1 +
                        xc[cnext01 * 3 + c] * wx1 * wy0 * wz1 +
                        xc[cnext10 * 3 + c] * wx0 * wy1 * wz1 +
                        xc[cnext11 * 3 + c] * wx1 * wy1 * wz1;

                    xf[fIdx + c] = val;
                }
            }
        }
    }
}

// Fused 3-component restriction: processes all xyz in one parallel region
void MG_Restrict_nodes_Vec3(const MGLevel& Lc, const MGLevel& Lf,
                            const std::vector<double>& rf, std::vector<double>& rc) {
    const int cNnx = Lc.resX + 1;
    const int cNny = Lc.resY + 1;
    const int cNnz = Lc.resZ + 1;

    const int fNnx = Lf.resX + 1;
    const int fNny = Lf.resY + 1;
    const int fNnz = Lf.resZ + 1;

    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    // Ensure output size
    size_t numCoarseNodes = (size_t)cNnx * cNny * cNnz;
    if (rc.size() != numCoarseNodes * 3) {
        rc.resize(numCoarseNodes * 3);
    }

    const double* __restrict__ rf_ptr = rf.data();
    double* __restrict__ rc_ptr = rc.data();

    // Precompute 1D weights
    const int wSize = 2 * span - 1;

#pragma omp parallel
    {
        // Thread-local weight array
        std::vector<double> W(wSize);
        for (int i = 0; i < wSize; ++i) {
            W[i] = 1.0 - std::abs(i - (span - 1)) * invSpan;
        }

        #pragma omp for collapse(2) schedule(static)
        for (int cz = 0; cz < cNnz; ++cz) {
            for (int cy = 0; cy < cNny; ++cy) {
                int fz0 = cz * span;
                int fy0 = cy * span;

                for (int cx = 0; cx < cNnx; ++cx) {
                    int fx0 = cx * span;

                    double valSum[3] = {0.0, 0.0, 0.0};
                    double wSum = 0.0;

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

                            int ix_start = (fx0 - span + 1 < 0) ? (span - 1 - fx0) : 0;
                            int ix_end   = (fx0 + span - 1 >= fNnx) ? (wSize - (fx0 + span - fNnx)) : wSize;

                            for (int ix = ix_start; ix < ix_end; ++ix) {
                                double wx = W[ix];
                                double w = wzy * wx;

                                int fx = fx0 + (ix - (span - 1));
                                size_t fIdx = (zOffset + yOffset + fx) * 3;

                                valSum[0] += rf_ptr[fIdx + 0] * w;
                                valSum[1] += rf_ptr[fIdx + 1] * w;
                                valSum[2] += rf_ptr[fIdx + 2] * w;
                                wSum += w;
                            }
                        }
                    }

                    size_t cIdx = ((size_t)cz * cNny * cNnx + cy * cNnx + cx) * 3;
                    if (wSum > 1e-12) {
                        double invW = 1.0 / wSum;
                        rc_ptr[cIdx + 0] = valSum[0] * invW;
                        rc_ptr[cIdx + 1] = valSum[1] * invW;
                        rc_ptr[cIdx + 2] = valSum[2] * invW;
                    } else {
                        rc_ptr[cIdx + 0] = 0.0;
                        rc_ptr[cIdx + 1] = 0.0;
                        rc_ptr[cIdx + 2] = 0.0;
                    }
                }
            }
        }
    }
}

// Inner version of prolongation - called from within existing parallel region
// Iterates over ALL fine nodes (matching MG_Prolongate_nodes_Strided)
// W parameter kept for API compatibility but no longer used
void MG_Prolongate_nodes_Vec3_Inner(const MGLevel& Lc, const MGLevel& Lf,
                                    const std::vector<double>& xc, std::vector<double>& xf,
                                    std::vector<double>& /*W*/) {
    const int fNnx = Lf.resX + 1;
    const int fNny = Lf.resY + 1;
    const int fNnz = Lf.resZ + 1;

    const int cNnx = Lc.resX + 1;
    const int cNny = Lc.resY + 1;

    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    #pragma omp for collapse(3) schedule(static)
    for (int fz = 0; fz < fNnz; ++fz) {
        for (int fy = 0; fy < fNny; ++fy) {
            for (int fx = 0; fx < fNnx; ++fx) {
                // Coarse cell index (top-left-front)
                int cx = fx / span;
                int cy = fy / span;
                int cz = fz / span;

                // Local offset in coarse cell [0, span]
                int rx = fx % span;
                int ry = fy % span;
                int rz = fz % span;

                // If on boundary, clamp to valid cell
                if (cx >= Lc.resX) { cx = Lc.resX - 1; rx = span; }
                if (cy >= Lc.resY) { cy = Lc.resY - 1; ry = span; }
                if (cz >= Lc.resZ) { cz = Lc.resZ - 1; rz = span; }

                // Trilinear weights
                double wx1 = (double)rx * invSpan; double wx0 = 1.0 - wx1;
                double wy1 = (double)ry * invSpan; double wy0 = 1.0 - wy1;
                double wz1 = (double)rz * invSpan; double wz0 = 1.0 - wz1;

                // Coarse Node Indices
                int cbase00 = cNnx * cNny * cz + cNnx * cy + cx;
                int cbase01 = cbase00 + 1;       // +x
                int cbase10 = cbase00 + cNnx;    // +y
                int cbase11 = cbase10 + 1;       // +x+y

                int cnext00 = cbase00 + cNnx * cNny; // +z
                int cnext01 = cnext00 + 1;
                int cnext10 = cnext00 + cNnx;
                int cnext11 = cnext10 + 1;

                // Write to fine grid
                size_t fIdx = ((size_t)fNnx * fNny * fz + fNnx * fy + fx) * 3;

                for (int c = 0; c < 3; ++c) {
                    double val =
                        xc[cbase00 * 3 + c] * wx0 * wy0 * wz0 +
                        xc[cbase01 * 3 + c] * wx1 * wy0 * wz0 +
                        xc[cbase10 * 3 + c] * wx0 * wy1 * wz0 +
                        xc[cbase11 * 3 + c] * wx1 * wy1 * wz0 +
                        xc[cnext00 * 3 + c] * wx0 * wy0 * wz1 +
                        xc[cnext01 * 3 + c] * wx1 * wy0 * wz1 +
                        xc[cnext10 * 3 + c] * wx0 * wy1 * wz1 +
                        xc[cnext11 * 3 + c] * wx1 * wy1 * wz1;

                    xf[fIdx + c] = val;
                }
            }
        }
    }
}

// Inner version of restriction - called from within existing parallel region
// W must be pre-allocated by caller with size >= 2*span-1
void MG_Restrict_nodes_Vec3_Inner(const MGLevel& Lc, const MGLevel& Lf,
                                  const std::vector<double>& rf, std::vector<double>& rc,
                                  std::vector<double>& W) {
    const int cNnx = Lc.resX + 1;
    const int cNny = Lc.resY + 1;
    const int cNnz = Lc.resZ + 1;

    const int fNnx = Lf.resX + 1;
    const int fNny = Lf.resY + 1;
    const int fNnz = Lf.resZ + 1;

    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;
    const int wSize = 2 * span - 1;

    const double* __restrict__ rf_ptr = rf.data();
    double* __restrict__ rc_ptr = rc.data();

    // Initialize weights - W is thread-local so no barrier needed
    if ((int)W.size() < wSize) W.resize(wSize);
    for (int i = 0; i < wSize; ++i) {
        W[i] = 1.0 - std::abs(i - (span - 1)) * invSpan;
    }

    #pragma omp for collapse(2) schedule(static)
    for (int cz = 0; cz < cNnz; ++cz) {
        for (int cy = 0; cy < cNny; ++cy) {
            int fz0 = cz * span;
            int fy0 = cy * span;

            for (int cx = 0; cx < cNnx; ++cx) {
                int fx0 = cx * span;

                double valSum[3] = {0.0, 0.0, 0.0};
                double wSum = 0.0;

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

                        int ix_start = (fx0 - span + 1 < 0) ? (span - 1 - fx0) : 0;
                        int ix_end   = (fx0 + span - 1 >= fNnx) ? (wSize - (fx0 + span - fNnx)) : wSize;

                        for (int ix = ix_start; ix < ix_end; ++ix) {
                            double wx = W[ix];
                            double w = wzy * wx;

                            int fx = fx0 + (ix - (span - 1));
                            size_t fIdx = (zOffset + yOffset + fx) * 3;

                            valSum[0] += rf_ptr[fIdx + 0] * w;
                            valSum[1] += rf_ptr[fIdx + 1] * w;
                            valSum[2] += rf_ptr[fIdx + 2] * w;
                            wSum += w;
                        }
                    }
                }

                size_t cIdx = ((size_t)cz * cNny * cNnx + cy * cNnx + cx) * 3;
                if (wSum > 1e-12) {
                    double invW = 1.0 / wSum;
                    rc_ptr[cIdx + 0] = valSum[0] * invW;
                    rc_ptr[cIdx + 1] = valSum[1] * invW;
                    rc_ptr[cIdx + 2] = valSum[2] * invW;
                } else {
                    rc_ptr[cIdx + 0] = 0.0;
                    rc_ptr[cIdx + 1] = 0.0;
                    rc_ptr[cIdx + 2] = 0.0;
                }
            }
        }
    }
}

} } // namespace top3d::mg
