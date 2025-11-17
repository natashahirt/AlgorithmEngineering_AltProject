#pragma once
#include "core.hpp"
#include <vector>

namespace top3d {
    
void restrict_to_free(const Problem&, const std::vector<double>& full, std::vector<double>& freev);
void scatter_from_free(const Problem&, const std::vector<double>& freev, std::vector<double>& full);

int PCG_free(const Problem&, const std::vector<double>& eleE,
             const std::vector<double>& bFree, std::vector<double>& xFree,
             double tol, int maxIt, Preconditioner M,
             std::vector<double>* xfull=nullptr, std::vector<double>* yfull=nullptr,
             std::vector<double>* pfull=nullptr, std::vector<double>* Apfull=nullptr,
             std::vector<double>* freeTmp=nullptr);

} // namespace top3d