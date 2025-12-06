
// Diagonal build and coarsening
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cstdint>

namespace top3d { namespace mg {

static void ComputeJacobiDiagonalFull(const Problem& pb,
									  const std::vector<float>& eleModulus,
									  std::vector<float>& diagFull) {
	const auto& mesh = pb.mesh;
	diagFull.assign(mesh.numDOFs, 0.0);
	const auto& Ke = mesh.Ke;
	
	// Use OpenMP with coloring or atomic. Atomic is simpler here for conciseness.
	#pragma omp parallel for
	for (int e=0; e<mesh.numElements; ++e) {
		float Ee = eleModulus[e];
		for (int a=0; a<8; ++a) {
			int n = mesh.eNodMat[e*8 + a];
			for (int c=0; c<3; ++c) {
				int local = 3*a + c;
				double val = Ke[local*24 + local] * Ee;
				#pragma omp atomic
				diagFull[3*n + c] += val;
			}
		}
	}
	for (float& v : diagFull) if (!(v > 0.0)) v = 1.0;
}


static void MG_CoarsenDiagonal(const MGLevel& Lc, const MGLevel& Lf,
								  const std::vector<float>& diagFine,
								  const std::vector<uint8_t>& fineFixedMask,
								  const std::vector<uint8_t>& coarseFixedMask,
								  std::vector<float>& diagCoarse) {
    const int span = Lc.spanWidth;
    const int grid = span+1;

	diagCoarse.assign((Lc.resX+1)*(Lc.resY+1)*(Lc.resZ+1)*3, 0.0);
	std::vector<float> wsum(diagCoarse.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return (Lf.resX+1)*(Lf.resY+1)*iz + (Lf.resX+1)*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return (Lc.resX+1)*(Lc.resY+1)*iz + (Lc.resX+1)*iy + ix; };

	#pragma omp parallel for collapse(3)
	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
				for (int iz=0; iz<=span; ++iz) {
					for (int iy=0; iy<=span; ++iy) {
						for (int ix=0; ix<=span; ++ix) {
							int fxi = fx0 + ix; int fyi = fy0 - iy; int fzi = fz0 + iz;
							int fn = idxF(fxi, fyi, fzi);
                            const float* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
							int cidx[8] = {
								idxC(cex,   Lc.resY-cey,   cez),
								idxC(cex+1, Lc.resY-cey,   cez),
								idxC(cex+1, Lc.resY-cey-1, cez),
								idxC(cex,   Lc.resY-cey-1, cez),
								idxC(cex,   Lc.resY-cey,   cez+1),
								idxC(cex+1, Lc.resY-cey,   cez+1),
								idxC(cex+1, Lc.resY-cey-1, cez+1),
								idxC(cex,   Lc.resY-cey-1, cez+1)
							};
							for (int a=0;a<8;a++) {
								float w2 = W[a]*W[a];
								for (int c=0;c<3;c++) {
									const int fineD = 3*fn + c;
									const int coarseD = 3*cidx[a] + c;
									if (coarseFixedMask[coarseD]) {
										// We'll set fixed coarse diagonals later; skip accumulation
										continue;
									}
									if (fineFixedMask[fineD]) {
										// Skip fixed fine contributions to avoid inflating neighboring coarse diagonals
										continue;
									}
									#pragma omp atomic
									diagCoarse[coarseD] += w2 * diagFine[fineD];
									#pragma omp atomic
									wsum[coarseD] += w2;
								}
							}
						}
					}
				}
			}
		}
	}
	for (size_t i=0;i<diagCoarse.size();++i) {
		if (coarseFixedMask[i]) {
			diagCoarse[i] = 1.0;
		} else if (wsum[i]>0) {
			diagCoarse[i] /= wsum[i];
			if (!(diagCoarse[i] > 0.0)) diagCoarse[i] = 1.0;
		} else {
			diagCoarse[i] = 1.0;
		}
	}
}

void MG_BuildDiagonals(const Problem& pb, const MGHierarchy& H,
							   const std::vector<std::vector<uint8_t>>& fixedMasks,
							   const std::vector<float>& eleModulus,
							   std::vector<std::vector<float>>& diagLevels) {
	diagLevels.resize(H.levels.size());
	ComputeJacobiDiagonalFull(pb, eleModulus, diagLevels[0]);
	// Impose BCs at finest level on the diagonal
	{
		const auto& mask0 = fixedMasks[0];
		for (size_t d=0; d<diagLevels[0].size() && d<mask0.size(); ++d) {
			if (mask0[d]) diagLevels[0][d] = 1.0;
		}
	}
	for (size_t l=0; l+1<H.levels.size(); ++l) {
		MG_CoarsenDiagonal(H.levels[l+1], H.levels[l],
			diagLevels[l],
			fixedMasks[l],
			fixedMasks[l+1],
			diagLevels[l+1]);
	}
}

} } // namespace top3d::mg
