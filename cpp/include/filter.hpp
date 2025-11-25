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
void ApplyPDEFilter(const Problem&, PDEFilter&, const std::vector<float>& srcEle, std::vector<float>& dstEle);
 
} // namespace top3d