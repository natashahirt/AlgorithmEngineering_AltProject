// Internal: multigrid coarsest-level and Cholesky declarations
#pragma once
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cstdint>

namespace top3d { namespace mg {

void EleMod_CompactToFull_Finest(const Problem& pb,
                                 const std::vector<double>& eleModCompact,
                                 std::vector<double>& eleModFull);

void MG_AssembleCoarsestDenseK_Galerkin(const Problem& pb,
                                        const MGHierarchy& H,
                                        const std::vector<double>& eleModFineFull,
                                        const std::vector<uint8_t>& fineFixedDofMask,
                                        std::vector<double>& Kc);

bool chol_spd_inplace(std::vector<double>& A, int N);
void chol_solve_lower(const std::vector<double>& L,
                      const std::vector<double>& b,
                      std::vector<double>& x, int N);

} } // namespace top3d::mg


