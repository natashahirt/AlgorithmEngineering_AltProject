#pragma once
#include "core.hpp"
#include <vector>

namespace top3d {
    
void restrict_to_free(const Problem&, const std::vector<float>& full, std::vector<float>& freev);
void scatter_from_free(const Problem&, const std::vector<float>& freev, std::vector<float>& full);

int PCG_free(const Problem&, const std::vector<float>& eleE,
             const std::vector<float>& bFree, std::vector<float>& xFree,
             float tol, int maxIt, Preconditioner M,
             std::vector<float>* xfull=nullptr, std::vector<float>* yfull=nullptr,
             std::vector<float>* pfull=nullptr, std::vector<float>* Apfull=nullptr,
             std::vector<float>* freeTmp=nullptr);

} // namespace top3d