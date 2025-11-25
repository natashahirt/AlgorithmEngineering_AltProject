#pragma once
#include "core.hpp"
#include <array>
#include <vector>
 
namespace top3d {
 
struct PDEFilter {
	// 8-node kernel (8x8) stored row-major
	std::array<float,8*8> kernel{};
	std::vector<float> diagPrecondNode; // numNodes
	// Warm-start storage
	std::vector<float> lastXNode;       // numNodes
	std::vector<float> lastRhsNode;     // numNodes
};
 
PDEFilter SetupPDEFilter(const Problem&, float filterRadius);
 
struct PDEFilterWorkspace {
	std::vector<float> rhs; // node-level RHS
	std::vector<float> x;   // node solution
	std::vector<float> r;   // residual
	std::vector<float> z;   // preconditioned residual
	std::vector<float> p;   // search direction
	std::vector<float> Ap;  // A*p
};
 
void ApplyPDEFilter(const Problem&, PDEFilter&, const std::vector<float>& srcEle, std::vector<float>& dstEle, PDEFilterWorkspace& ws);
 
} // namespace top3d