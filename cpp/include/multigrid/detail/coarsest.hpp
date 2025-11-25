// Internal: multigrid coarsest-level and Cholesky declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cstdint>

namespace top3d { namespace mg {

void EleMod_CompactToFull_Finest(const Problem& pb,
                                 const std::vector<float>& eleModCompact,
                                 std::vector<float>& eleModFull);

void MG_AssembleCoarsestDenseK_Galerkin(const Problem& pb,
                                        const MGHierarchy& H,
                                        const std::vector<float>& eleModFineFull,
                                        const std::vector<uint8_t>& fineFixedDofMask,
                                        std::vector<float>& Kc);

bool chol_spd_inplace(std::vector<float>& A, int N);
void chol_solve_lower(const std::vector<float>& L,
                      const std::vector<float>& b,
                      std::vector<float>& x, int N);

} } // namespace top3d::mg


