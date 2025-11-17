#pragma once
#include "core.hpp"
#include <array>
#include <vector>
 
namespace top3d {
 
struct PDEFilter {
	// 8-node kernel (8x8) stored row-major
	std::array<double,8*8> kernel{};
	std::vector<double> diagPrecondNode; // numNodes
	// Warm-start storage
	std::vector<double> lastXNode;       // numNodes
	std::vector<double> lastRhsNode;     // numNodes
};
 
PDEFilter SetupPDEFilter(const Problem&, double filterRadius);
void ApplyPDEFilter(const Problem&, PDEFilter&, const std::vector<double>& srcEle, std::vector<double>& dstEle);
 
} // namespace top3d