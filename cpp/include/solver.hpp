#pragma once
#include "core.hpp"
#include <vector>

namespace top3d {
    
void restrict_to_free(const Problem&, const std::vector<double>& full, std::vector<double>& freev);
void scatter_from_free(const Problem&, const std::vector<double>& freev, std::vector<double>& full);

struct PCGFreeWorkspace {
	// Full DOF vectors (size = mesh.numDOFs)
	std::vector<double> xfull;  // holds full-space vector (e.g. pfull, xfull)
	std::vector<double> yfull;  // holds K * xfull result (e.g. Apfull, yfull)
	// Temporary free-space buffer, if needed
	std::vector<double> tmpFree;
};

// Simple Jacobi (diagonal) preconditioner built from element Ke and eleE
Preconditioner make_jacobi_preconditioner(const Problem& pb, const std::vector<double>& eleE);

int PCG_free(const Problem&, const std::vector<double>& eleE,
             const std::vector<double>& bFree, std::vector<double>& xFree,
             double tol, int maxIt, Preconditioner M,
             PCGFreeWorkspace& ws);

} // namespace top3d