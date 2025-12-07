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

    if (xf.size() < (size_t)(fnnx * fnny * fnnz * stride)) {
        #pragma omp single
        xf.resize(fnnx * fnny * fnnz * stride);
    }

    #pragma omp for collapse(3) schedule(static) nowait
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

// Buffer variant (xf_buf length = (Lf.resX+1)*(Lf.resY+1)*(Lf.resZ+1))
void MG_Prolongate_nodes_Strided_buf(const MGLevel& Lc, const MGLevel& Lf,
                                     const std::vector<double>& xc, double* xf_buf,
                                     int component, int stride) {
    const int fnnx = Lf.resX + 1;
    const int fnny = Lf.resY + 1;
    const int fnnz = Lf.resZ + 1;
    
    const int cnnx = Lc.resX + 1;
    const int cnny = Lc.resY + 1;
    
    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    #pragma omp for collapse(3) schedule(static) nowait
    for (int fz = 0; fz < fnnz; ++fz) {
        for (int fy = 0; fy < fnny; ++fy) {
            for (int fx = 0; fx < fnnx; ++fx) {
                int cx = fx / span;
                int cy = fy / span;
                int cz = fz / span;

                int rx = fx % span;
                int ry = fy % span;
                int rz = fz % span;

                if (cx >= Lc.resX) { cx = Lc.resX - 1; rx = span; }
                if (cy >= Lc.resY) { cy = Lc.resY - 1; ry = span; }
                if (cz >= Lc.resZ) { cz = Lc.resZ - 1; rz = span; }

                double wx1 = (double)rx * invSpan; double wx0 = 1.0 - wx1;
                double wy1 = (double)ry * invSpan; double wy0 = 1.0 - wy1;
                double wz1 = (double)rz * invSpan; double wz0 = 1.0 - wz1;

                int cbase00 = cnnx * cnny * cz + cnnx * cy + cx;
                int cbase01 = cbase00 + 1;
                int cbase10 = cbase00 + cnnx;
                int cbase11 = cbase10 + 1;
                
                int cnext00 = cbase00 + cnnx * cnny;
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

                int fidx = fnnx * fnny * fz + fnnx * fy + fx;
                xf_buf[fidx] = val;
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
    const int cnnx = Lc.resX + 1;
    const int cnny = Lc.resY + 1;
    const int cnnz = Lc.resZ + 1;
    
    const int fnnx = Lf.resX + 1;
    const int fnny = Lf.resY + 1;
    const int fnnz = Lf.resZ + 1;
    
    const int span = Lc.spanWidth;
    const double invSpan = 1.0 / span;

    if (rc.size() < (size_t)(cnnx * cnny * cnnz * stride)) {
        #pragma omp single
        rc.resize(cnnx * cnny * cnnz * stride);
    }

    #pragma omp for collapse(3) schedule(static) nowait
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
