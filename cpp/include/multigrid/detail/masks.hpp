// Internal: multigrid masks declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cstdint>

namespace top3d { namespace mg {

void MG_BuildFixedMasks(const Problem& pb, const MGHierarchy& H,
                        std::vector<std::vector<uint8_t>>& fixedMasks);

} } // namespace top3d::mg


