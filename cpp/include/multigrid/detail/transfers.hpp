// Internal: multigrid transfer operator declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>

namespace top3d { namespace mg {

void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
                         const std::vector<float>& xc, std::vector<float>& xf);

void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
                       const std::vector<float>& rf, std::vector<float>& rc);

} } // namespace top3d::mg


