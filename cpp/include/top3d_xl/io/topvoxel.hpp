#ifndef TOPVOXEL_HPP
#define TOPVOXEL_HPP

#include <vector>
#include <array>
#include <string>

namespace top3d {

// External model/BCs (TopVoxel v1.0)
struct ExternalBC {
	std::vector<std::array<int,4>> fixations;   // [node(1-based), fixX, fixY, fixZ]
	std::vector<int> passiveFull;               // 1-based MATLAB-style element ids (full grid)
	std::vector<int> passiveCompact;            // mapped compact element ids (0-based)
};
struct ExternalLoads {
	// Each load case: entries [node(1-based), Fx, Fy, Fz]
	std::vector<std::vector<std::array<double,4>>> cases;
	std::vector<double> weights; // objective weights per case
};

struct Problem; // forward declare

// Build from external TopVoxel v1.0 file and discretize
void CreateVoxelFEAmodel_TopVoxel(Problem& pb, const std::string& path);

}

#endif // TOPVOXEL_HPP



