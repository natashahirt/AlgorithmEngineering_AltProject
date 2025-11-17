// Internal: multigrid diagonal build declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cstdint>

namespace top3d { namespace mg {

void MG_BuildDiagonals(const Problem&,
                       const MGHierarchy&,
                       const std::vector<std::vector<uint8_t>>& fixedMasks,
                       const std::vector<double>& eleModulus,
                       std::vector<std::vector<double>>& diagLevels);

} } // namespace top3d::mg


