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

} } // namespace top3d::mg
