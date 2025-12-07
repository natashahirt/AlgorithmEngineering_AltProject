#pragma once
#include "core.hpp"
#include <array>
#include <vector>
#include <cstdint>

namespace top3d {

// Workspace for batched K*u (MATLAB-style)
// Stores intermediate matrices for batch matvec + sorting-based scatter
struct KTimesUWorkspace {
    // Batched element displacements: uMat[e * 24 + a] = displacement for element e, local DOF a
    std::vector<double> uMat;    // numElements * 24
    // Batched element forces: fMat[e * 24 + a] = force for element e, local DOF a
    std::vector<double> fMat;    // numElements * 24

    // Precomputed scatter indices (built once per problem)
    // scatterIdx[k] = element_idx * 24 + local_dof (index into fMat)
    // Grouped by global_free_dof for efficient accumulation
    std::vector<int32_t> scatterIdx;
    std::vector<int32_t> dofBoundaries;  // boundaries[i] = start of DOF i in scatterIdx
    bool scatterIndexBuilt = false;

    int numElements = 0;
    size_t numFreeDofs = 0;

    void resize(int nElements, size_t nFreeDofs) {
        if (numElements == nElements && numFreeDofs == nFreeDofs) return;
        numElements = nElements;
        numFreeDofs = nFreeDofs;
        uMat.resize(static_cast<size_t>(nElements) * 24);
        fMat.resize(static_cast<size_t>(nElements) * 24);
        scatterIndexBuilt = false;  // Need to rebuild scatter indices
    }
};

std::array<double,24*24> ComputeVoxelKe(double nu, double cellSize);
void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz);
void ApplyBoundaryConditions(Problem& pb);

// Original signature (allocates workspace internally - for backward compatibility)
void K_times_u_finest(const Problem& pb, const std::vector<double>& eleE, const std::vector<double>& uFree, std::vector<double>& yFree);

// Workspace-enabled signature (no allocation during call)
void K_times_u_finest(const Problem& pb, const std::vector<double>& eleE, const std::vector<double>& uFree, std::vector<double>& yFree, KTimesUWorkspace& ws);

double ComputeCompliance(const Problem&, const std::vector<double>& eleE, const DOFData& U, std::vector<double>& ce);

} // namespace top3d