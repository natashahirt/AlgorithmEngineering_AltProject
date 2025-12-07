#pragma once
#include "core.hpp"
#include <array>
#include <vector>

namespace top3d {

// Workspace for K_times_u with thread-local accumulators
// Allocate once, reuse across all matvec calls to avoid repeated allocation
struct KTimesUWorkspace {
    // Flat storage: threadLocalY_flat[t * numDOFs + i] for thread t, DOF i
    // This layout allows better cache reuse during reduction
    std::vector<double> threadLocalY_flat;
    int numThreads = 0;
    size_t numDOFs = 0;

    void resize(int nThreads, size_t nDOFs) {
        if (numThreads == nThreads && numDOFs == nDOFs) return;
        numThreads = nThreads;
        numDOFs = nDOFs;
        threadLocalY_flat.resize(static_cast<size_t>(nThreads) * nDOFs, 0.0);
    }

    // Get pointer to thread t's buffer
    double* getThreadBuffer(int t) {
        return threadLocalY_flat.data() + static_cast<size_t>(t) * numDOFs;
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