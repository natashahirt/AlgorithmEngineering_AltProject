#pragma once
#include "core.hpp"
#include <cstdint>
#include <vector>
#include <memory>

namespace top3d { namespace mg {

struct MGPrecondConfig {
	bool nonDyadic = true;   // first jump 1->3 (span=4)
	int  maxLevels = 5;      // cap levels
	float weight = 0.6f;     // diagonal relaxation factor
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

struct MGWorkspace {
    std::vector<std::vector<double>> rLv;
    std::vector<std::vector<double>> xLv;
    
    // Intermediate buffers for component-wise transfers
    std::vector<double> tmp_rf; 
    std::vector<double> tmp_rc;
    std::vector<double> tmp_xc;
    std::vector<double> tmp_xf;
    std::vector<double> tmp_add; // Used in prolongation

    void resize(const MGHierarchy& H) {
        rLv.resize(H.levels.size());
        xLv.resize(H.levels.size());
        size_t maxNodes = 0;
        for(size_t i=0; i<H.levels.size(); ++i) {
            int nDofs = 3 * H.levels[i].numNodes;
            rLv[i].resize(nDofs);
            xLv[i].resize(nDofs);
            if((size_t)H.levels[i].numNodes > maxNodes) maxNodes = (size_t)H.levels[i].numNodes;
        }
        tmp_rf.resize(maxNodes);
        tmp_rc.resize(maxNodes);
        tmp_xc.resize(maxNodes);
        tmp_xf.resize(maxNodes);
        tmp_add.resize(3*maxNodes);
    }
};

void BuildMGHierarchy(const Problem&, bool nonDyadic, MGHierarchy&, int maxLevels);
void build_static_once(const Problem&, const MGPrecondConfig&, MGHierarchy&, std::vector<std::vector<uint8_t>>&);
Preconditioner make_diagonal_preconditioner_from_static(const Problem&, const MGHierarchy&,
    const std::vector<std::vector<uint8_t>>& fixedMasks,
    const std::vector<double>& eleE, const MGPrecondConfig&);

// Transfer helpers
void MG_Prolongate_nodes_Strided(const MGLevel& Lc, const MGLevel& Lf,
                                 const std::vector<double>& xc, std::vector<double>& xf,
                                 int component, int stride);
void MG_Prolongate_nodes_Strided_buf(const MGLevel& Lc, const MGLevel& Lf,
                                     const std::vector<double>& xc, double* xf_buf,
                                     int component, int stride);
void MG_Restrict_nodes_Strided(const MGLevel& Lc, const MGLevel& Lf,
                               const std::vector<double>& rf, std::vector<double>& rc,
                               int component, int stride);
      
} } // namespace top3d::mg
