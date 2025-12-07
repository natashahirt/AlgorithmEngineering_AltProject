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

	// Thread-local storage for scatter
	std::vector<std::vector<double>> y_thread_local;

    int numElements = 0;
    size_t numFreeDofs = 0;

    void resize(int nElements, size_t nFreeDofs) {
        if (numElements == nElements && numFreeDofs == nFreeDofs) return;
        numElements = nElements;
        numFreeDofs = nFreeDofs;
        uMat.resize(static_cast<size_t>(nElements) * 24);
        fMat.resize(static_cast<size_t>(nElements) * 24);
    }

	void resize_thread_local(int nthreads, size_t nFreeDofs) {
		if (y_thread_local.size() == nthreads) return;
		y_thread_local.resize(nthreads, std::vector<double>(nFreeDofs));
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