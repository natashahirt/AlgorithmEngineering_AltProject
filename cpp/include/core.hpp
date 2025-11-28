#ifndef TOP3D_XL_CORE_HPP
#define TOP3D_XL_CORE_HPP

#include <vector>
#include <array>
#include <string>
#include <functional>
#include <cstdint>
#include <optional>
#include <vector>

namespace top3d {

struct ElementColoring {
	int numColors = 0;
	std::vector<int> elemColor;                 // size = numElements, color index per element
	std::vector<std::vector<int>> colorBuckets; // buckets[c] = list of elements with color c
};

struct DOFData {
	std::vector<double> ux;
	std::vector<double> uy;
	std::vector<double> uz;
};

struct GlobalParams {
	// Physical
	float youngsModulus = 1.0f;
	float poissonRatio = 0.3f;
	float youngsModulusMin = 1.0e-6f;
	float simpPenalty = 3.0f;
	float cellSize = 1.0f;

	// Solver
	float cgTol = 1.0e-3f;
	int cgMaxIt = 800;

	// Optimization
	int passiveLayersBoundary = 0;
	int passiveLayersLoads = 0;
	int passiveLayersFixations = 0;
};

struct CartesianMesh {
	int resX = 0;
	int resY = 0;
	int resZ = 0;

	// Original (pre-padding) element resolutions
	int origResX = 0;
	int origResY = 0;
	int origResZ = 0;

	std::array<float,3> eleSize {1.0f,1.0f,1.0f};

	int numElements = 0;
	int numNodes = 0;
	int numDOFs = 0;

	std::vector<int32_t> eleMapBack;      // indices of solid elements (flattened)
	std::vector<int32_t> eleMapForward;   // map from all elements -> compact solid index
	std::vector<int32_t> nodMapBack;      // map compact nodes -> full node ids
	std::vector<int32_t> nodMapForward;   // map full node ids -> compact nodes

	std::vector<int32_t> nodesOnBoundary; // compact nodes indices
	std::vector<int32_t> elementsOnBoundary; // compact element indices

	// Element to node connectivity (compact indices), shape: numElements x 8
	std::vector<int32_t> eNodMat; // stored row-major: [ele][8]
	// Element to global DOF connectivity, shape: numElements x 24 (8 nodes x 3 dofs)
	std::vector<int32_t> eDofMat; // stored row-major: [ele][24]

	// Element stiffness (24x24) for unit modulus
	alignas(64) std::array<double,24*24> Ke{};

	// Optional element graph coloring for race-free parallel assembly
	ElementColoring coloring;
};


struct Problem {
    GlobalParams params;
    CartesianMesh mesh;

    // Loads: one load case supported in minimal port as SoA (per component per node)
    DOFData F;                      // size of each = numNodes
    std::vector<uint8_t> isFreeDOF; // size = numDOFs (1 free, 0 fixed)
    std::vector<int> freeDofIndex;  // compact list of free dofs
    // Cached per-free-DOF mapping to node and component for fast gather/scatter
    std::vector<int> freeNodeIndex; // same length as freeDofIndex
    std::vector<int> freeCompIndex; // values in {0,1,2}

    // Design variables
    std::vector<float> density;    // per element x, size = numElements
};

// Entry points
void InitialSettings(GlobalParams& out);

// Preconditioner functor: z = M^{-1} r (operates on free-DOF vectors)
using Preconditioner = std::function<void(const std::vector<double>& rFree, std::vector<double>& zFree)>;

} // namespace top3d

#endif // TOP3D_XL_CORE_HPP


