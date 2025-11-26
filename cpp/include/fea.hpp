#pragma once
#include "core.hpp"
#include <array>
#include <vector>

namespace top3d {

std::array<double,24*24> ComputeVoxelKe(double nu, double cellSize);
void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz);
void ApplyBoundaryConditions(Problem& pb);
void K_times_u_finest(const Problem&, const std::vector<double>& eleE, const DOFData& U, DOFData& Y);
double ComputeCompliance(const Problem&, const std::vector<double>& eleE, const DOFData& U, std::vector<double>& ce);

} // namespace top3d