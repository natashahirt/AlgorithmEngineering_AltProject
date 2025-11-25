
// Multigrid hierarchy construction
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

namespace top3d { namespace mg {

// ===== MG scaffolding: trilinear weights and hierarchy (Step 2) =====

// Trilinear shape for 8-node hex at natural coords (xi, eta, zeta) in [-1,1]
static inline void shape_trilinear8(float xi, float eta, float zeta, float N[8]) {
	N[0] = 0.125*(1-xi)*(1-eta)*(1-zeta);
	N[1] = 0.125*(1+xi)*(1-eta)*(1-zeta);
	N[2] = 0.125*(1+xi)*(1+eta)*(1-zeta);
	N[3] = 0.125*(1-xi)*(1+eta)*(1-zeta);
	N[4] = 0.125*(1-xi)*(1-eta)*(1+zeta);
	N[5] = 0.125*(1+xi)*(1-eta)*(1+zeta);
	N[6] = 0.125*(1+xi)*(1+eta)*(1+zeta);
	N[7] = 0.125*(1-xi)*(1+eta)*(1+zeta);
}

// Build per-element nodal weights table for a given span (2 or 4)
// Mapping: coarse element's 8 vertices -> (span+1)^3 embedded fine-grid vertices
static std::vector<float> make_trilinear_weights_table(int span) {
	const int grid = span + 1;
	std::vector<float> W(grid*grid*grid*8, 0.0);
	for (int iz=0; iz<=span; ++iz) {
		for (int iy=0; iy<=span; ++iy) {
			for (int ix=0; ix<=span; ++ix) {
				float xi   = -1.0 + 2.0 * (float(ix) / float(span));
				float eta  = -1.0 + 2.0 * (float(iy) / float(span));
				float zeta = -1.0 + 2.0 * (float(iz) / float(span));
				float N[8]; shape_trilinear8(xi, eta, zeta, N);
				const int v = (iz*grid + iy)*grid + ix;
				for (int a=0; a<8; ++a) W[v*8 + a] = N[a];
			}
		}
	}
	return W;
}

// Build a structured MG level of resolution (nx,ny,nz)
static MGLevel build_structured_level(int nx, int ny, int nz, int span) {
	MGLevel L;
	L.resX = nx; L.resY = ny; L.resZ = nz;
	L.spanWidth = span;
	L.numElements = nx*ny*nz;
	const int nnx = nx+1, nny = ny+1, nnz = nz+1;
	L.numNodes = nnx*nny*nnz;
	L.numDOFs = L.numNodes * 3;

	L.nodMapBack.resize(L.numNodes);
	std::iota(L.nodMapBack.begin(), L.nodMapBack.end(), 0);
	L.nodMapForward = L.nodMapBack;

	L.eNodMat.resize(L.numElements*8);
	auto nodeIndex = [&](int ix,int iy,int iz){ return (nnx*nny*iz + nnx*iy + ix); };
	int eComp=0;
	for (int ez=0; ez<nz; ++ez) {
		for (int ex=0; ex<nx; ++ex) {
			for (int ey=0; ey<ny; ++ey) {
				int n1 = nodeIndex(ex,   ny-ey,   ez);
				int n2 = nodeIndex(ex+1, ny-ey,   ez);
				int n3 = nodeIndex(ex+1, ny-ey-1, ez);
				int n4 = nodeIndex(ex,   ny-ey-1, ez);
				int n5 = nodeIndex(ex,   ny-ey,   ez+1);
				int n6 = nodeIndex(ex+1, ny-ey,   ez+1);
				int n7 = nodeIndex(ex+1, ny-ey-1, ez+1);
				int n8 = nodeIndex(ex,   ny-ey-1, ez+1);
				int base = eComp*8;
				L.eNodMat[base+0]=n1; L.eNodMat[base+1]=n2; L.eNodMat[base+2]=n3; L.eNodMat[base+3]=n4;
				L.eNodMat[base+4]=n5; L.eNodMat[base+5]=n6; L.eNodMat[base+6]=n7; L.eNodMat[base+7]=n8;
				++eComp;
			}
		}
	}
	L.weightsNode = make_trilinear_weights_table(span);
	return L;
}

// Heuristic: adapt MG level count to keep coarsest DOFs <= NlimitDofs and stop when tiny
int ComputeAdaptiveMaxLevels(const Problem& pb, bool nonDyadic, int cap, int NlimitDofs) {
	int nx = pb.mesh.resX;
	int ny = pb.mesh.resY;
	int nz = pb.mesh.resZ;
	int levels = 1; // include finest

	for (int li=1; li<cap; ++li) {
		int span = (li==1 && nonDyadic ? 4 : 2);
		// Require exact divisibility (to match structured transfers)
		if (nx % span != 0 || ny % span != 0 || nz % span != 0) break;

		int nnx = nx / span, nny = ny / span, nnz = nz / span;
		if (nnx < 1 || nny < 1 || nnz < 1) break;

		long long nodes = 1LL*(nnx+1)*(nny+1)*(nnz+1);
		long long dofs  = 3LL * nodes;
		// Stop before exceeding coarsest dense factorization limit
		if (dofs > NlimitDofs) break;

		// Accept this level
		nx = nnx; ny = nny; nz = nnz;
		levels++;

		// Mirror MATLAB's guard: stop when very small
		if (std::min({nx,ny,nz}) <= 2) break;
	}
	return levels;
}

void BuildMGHierarchy(const Problem& pb, bool nonDyadic, MGHierarchy& H, int maxLevels) {
	H.levels.clear();
	H.nonDyadic = nonDyadic;

	// Finest level: same resolution; span not used here
	{
		MGLevel L0 = build_structured_level(pb.mesh.resX, pb.mesh.resY, pb.mesh.resZ, /*span=*/1);
		L0.weightsNode.clear();
		H.levels.push_back(std::move(L0));
	}

	int nx = pb.mesh.resX;
	int ny = pb.mesh.resY;
	int nz = pb.mesh.resZ;

    for (int li=1; li<maxLevels; ++li) {
        int span = (li==1 && nonDyadic ? 4 : 2);
        // Enforce divisibility to avoid misalignment; stop if indivisible
        if (nx % span != 0 || ny % span != 0 || nz % span != 0) break;
        if (nx/span < 1 || ny/span < 1 || nz/span < 1) break;
        nx = nx / span; ny = ny / span; nz = nz / span;
        H.levels.push_back(build_structured_level(nx, ny, nz, span));
        if (std::min({nx,ny,nz}) <= 2) break;
    }
}

} } // namespace top3d::mg
