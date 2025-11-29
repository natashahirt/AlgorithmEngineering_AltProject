#pragma once
#include "core.hpp"
#include <array>
#include <vector>
 
namespace top3d {
 
struct PDEFilter {
	// 8-node kernel (8x8) stored row-major
	alignas(64) std::array<double,8*8> kernel{};
	std::vector<double> diagPrecondNode; // numNodes
	// Warm-start storage
	std::vector<double> lastXNode;       // numNodes
	std::vector<double> lastRhsNode;     // numNodes
};
 
PDEFilter SetupPDEFilter(const Problem&, float filterRadius);
 
struct PDEFilterWorkspace {
	std::vector<double> rhs; // node-level RHS
	std::vector<double> x;   // node solution
	std::vector<double> r;   // residual
	std::vector<double> z;   // preconditioned residual
	std::vector<double> p;   // search direction
	std::vector<double> Ap;  // A*p
};
 
void ApplyPDEFilter(const Problem&, PDEFilter&, const std::vector<float>& srcEle, std::vector<float>& dstEle, PDEFilterWorkspace& ws);
 
} // namespace top3d