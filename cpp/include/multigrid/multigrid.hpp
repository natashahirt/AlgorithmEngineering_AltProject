#pragma once
#include "core.hpp"
#include <cstdint>
#include <vector>
 
namespace top3d { namespace mg {
 
struct MGPrecondConfig {
	bool nonDyadic = true;   // first jump 1->3 (span=4)
	int  maxLevels = 5;      // cap levels
	double weight = 0.6;     // diagonal relaxation factor
};
 
struct MGLevel {
	int resX = 0, resY = 0, resZ = 0;
	int spanWidth = 2;                 // 2 (dyadic) or 4 (non-dyadic jump)
	int numElements = 0, numNodes = 0, numDOFs = 0;
 
	// Structured connectivity at this level
	std::vector<int32_t> eNodMat;      // numElements x 8
	std::vector<int32_t> nodMapBack;   // [0..numNodes-1]
	std::vector<int32_t> nodMapForward;
 
	// Per-element nodal weights from 8 coarse vertices to embedded (span+1)^3 fine vertices
	// Stored as weightsNode[(iz*grid + iy)*grid + ix]*8 + a, grid = spanWidth+1, a in [0..7]
	std::vector<double> weightsNode;   // size = (span+1)^3 * 8
};
 
struct MGHierarchy {
	std::vector<MGLevel> levels;       // levels[0] = finest
	bool nonDyadic = true;             // if true, first coarsening uses span=4
};
 
void BuildMGHierarchy(const Problem&, bool nonDyadic, MGHierarchy&, int maxLevels);
void build_static_once(const Problem&, const MGPrecondConfig&, MGHierarchy&, std::vector<std::vector<uint8_t>>&);
Preconditioner make_diagonal_preconditioner_from_static(const Problem&, const MGHierarchy&,
    const std::vector<std::vector<uint8_t>>& fixedMasks,
    const std::vector<double>& eleE, const MGPrecondConfig&);
      
} } // namespace top3d::mg