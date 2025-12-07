#pragma once
#include "core.hpp"
#include <array>
#include <vector>
#include <cstdint>

namespace top3d {

std::array<double,24*24> ComputeVoxelKe(double nu, double cellSize);
void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz);
void ApplyBoundaryConditions(Problem& pb);

void K_times_u_finest(const Problem& pb, const std::vector<double>& eleE, const std::vector<double>& uFree, std::vector<double>& yFree);

double ComputeCompliance(const Problem&, const std::vector<double>& eleE, const DOFData& U, std::vector<double>& ce);

} // namespace top3d