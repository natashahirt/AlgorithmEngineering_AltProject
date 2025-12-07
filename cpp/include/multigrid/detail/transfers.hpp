// Internal: multigrid transfer operator declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>

namespace top3d { namespace mg {

void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
                         const std::vector<double>& xc, std::vector<double>& xf);

void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
                       const std::vector<double>& rf, std::vector<double>& rc);

// Optimized Strided interface (avoids data copies)
// stride=1 implies contiguous data (component=0)
// stride=3 implies interleaved vector (component 0,1,2)
void MG_Prolongate_nodes_Strided(const MGLevel& Lc, const MGLevel& Lf,
                                 const std::vector<double>& xc, std::vector<double>& xf,
                                 int component, int stride);

void MG_Restrict_nodes_Strided(const MGLevel& Lc, const MGLevel& Lf,
                               const std::vector<double>& rf, std::vector<double>& rc,
                               int component, int stride);

// Fused 3-component transfers: process all xyz components in a single parallel region
// Input/output are interleaved as [x0,y0,z0, x1,y1,z1, ...]
void MG_Prolongate_nodes_Vec3(const MGLevel& Lc, const MGLevel& Lf,
                              const std::vector<double>& xc, std::vector<double>& xf);

void MG_Restrict_nodes_Vec3(const MGLevel& Lc, const MGLevel& Lf,
                            const std::vector<double>& rf, std::vector<double>& rc);

// "Inner" versions: called from within an existing parallel region (no barrier at end)
// These use #pragma omp for instead of #pragma omp parallel for
// W is a thread-local weight buffer that must be pre-allocated by the caller
void MG_Prolongate_nodes_Vec3_Inner(const MGLevel& Lc, const MGLevel& Lf,
                                    const std::vector<double>& xc, std::vector<double>& xf,
                                    std::vector<double>& W);

void MG_Restrict_nodes_Vec3_Inner(const MGLevel& Lc, const MGLevel& Lf,
                                  const std::vector<double>& rf, std::vector<double>& rc,
                                  std::vector<double>& W);

} } // namespace top3d::mg
