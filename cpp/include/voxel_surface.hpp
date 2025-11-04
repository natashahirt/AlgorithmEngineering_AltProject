#ifndef VOXEL_SURFACE_HPP
#define VOXEL_SURFACE_HPP

#include <vector>
#include <array>
#include <cstdint>

namespace voxsurf {

// Extract surface triangles from a binary/scalar grid using face extraction at threshold.
// vol is [ny][nx][nz] row-major by y,x,z.
void extract_faces(const std::vector<float>& vol, int ny, int nx, int nz, float iso,
				   std::vector<std::array<float,3>>& outVerts,
				   std::vector<std::array<uint32_t,3>>& outFaces);

}

#endif


