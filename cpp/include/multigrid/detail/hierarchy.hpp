// Internal: multigrid hierarchy helpers
#pragma once
#include "multigrid/multigrid.hpp"

namespace top3d { namespace mg {

int ComputeAdaptiveMaxLevels(const Problem&,
                             bool nonDyadic,
                             int cap,
                             int NlimitDofs);

} } // namespace top3d::mg


