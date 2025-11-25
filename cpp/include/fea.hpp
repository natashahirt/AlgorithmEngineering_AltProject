#pragma once
#include "core.hpp"
#include <array>
#include <vector>

namespace top3d {

std::array<float,24*24> ComputeVoxelKe(float nu, float cellSize);
void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz);
void ApplyBoundaryConditions(Problem& pb);
void K_times_u_finest(const Problem&, const std::vector<float>& eleE, const std::vector<float>& u, std::vector<float>& y);
float ComputeCompliance(const Problem&, const std::vector<float>& eleE, const std::vector<float>& U, std::vector<float>& ce);

} // namespace top3d