#pragma once
#include "core.hpp"
#include <vector>

namespace top3d {
    
void restrict_to_free(const Problem&, const std::vector<float>& full, std::vector<float>& freev);
void scatter_from_free(const Problem&, const std::vector<float>& freev, std::vector<float>& full);

struct PCGFreeWorkspace {
	// Full DOF vectors (size = mesh.numDOFs)
	std::vector<float> xfull;  // holds full-space vector (e.g. pfull, xfull)
	std::vector<float> yfull;  // holds K * xfull result (e.g. Apfull, yfull)
	// Temporary free-space buffer, if needed
	std::vector<float> tmpFree;
};

// Simple Jacobi (diagonal) preconditioner built from element Ke and eleE
Preconditioner make_jacobi_preconditioner(const Problem& pb, const std::vector<float>& eleE);

int PCG_free(const Problem&, const std::vector<float>& eleE,
             const std::vector<float>& bFree, std::vector<float>& xFree,
             float tol, int maxIt, Preconditioner M,
             PCGFreeWorkspace& ws);

} // namespace top3d