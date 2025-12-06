// Internal: multigrid transfer operator declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>

namespace top3d { namespace mg {

void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
                         const std::vector<double>& xc, std::vector<double>& xf);

void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
                       const std::vector<double>& rf, std::vector<double>& rc);

} } // namespace top3d::mg


