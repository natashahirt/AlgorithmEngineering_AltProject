#pragma once
#include "core.hpp"
#include <vector>

namespace top3d {
    
void restrict_to_free(const Problem&, const DOFData& full, std::vector<double>& freev);
void scatter_from_free(const Problem&, const std::vector<double>& freev, DOFData& full);

struct PCGFreeWorkspace {
	std::vector<double> r;
	std::vector<double> z;
	std::vector<double> p;
	std::vector<double> Ap;
};

// Simple Jacobi (diagonal) preconditioner built from element Ke and eleE
Preconditioner make_jacobi_preconditioner(const Problem& pb, const std::vector<double>& eleE);

int PCG_free(const Problem&, const std::vector<double>& eleE,
             const std::vector<double>& bFree, std::vector<double>& xFree,
             double tol, int maxIt, const Preconditioner& M,
             PCGFreeWorkspace& ws);

} // namespace top3d