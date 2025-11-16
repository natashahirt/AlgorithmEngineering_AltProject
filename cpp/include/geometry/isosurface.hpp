#ifndef ISOSURFACE_HPP
#define ISOSURFACE_HPP

#include <vector>
#include <array>
#include <cstdint>

namespace mcube {

// Marching cubes over a scalar volume dims (ny, nx, nz) in row-major [y][x][z]
// isovalue in [0,1]. Returns triangle vertices and face indices.
void marching_cubes(const std::vector<float>& vol, int ny, int nx, int nz, float iso,
					 std::vector<std::array<float,3>>& outVerts,
					 std::vector<std::array<uint32_t,3>>& outFaces);

}

#endif


