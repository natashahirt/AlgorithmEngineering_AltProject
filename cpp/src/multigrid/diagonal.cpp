
// Diagonal build and coarsening
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cstdint>
#include <algorithm> // Added for std::max/min if needed

namespace top3d { namespace mg {

static void ComputeJacobiDiagonalFull(const Problem& pb,
									  const std::vector<double>& eleModulus,
									  std::vector<double>& diagFull) {
	const auto& mesh = pb.mesh;
	diagFull.assign(mesh.numDOFs, 0.0);
	const auto& Ke = mesh.Ke;
	
	// OPTIMIZATION 1: Use coloring to remove atomics
	if (mesh.coloring.numColors > 0) {
		const auto& buckets = mesh.coloring.colorBuckets;
		const int numColors = mesh.coloring.numColors;
		
		#pragma omp parallel
		{
			for (int c = 0; c < numColors; ++c) {
				const auto& elems = buckets[c];
				size_t nElems = elems.size();
				
				#pragma omp for schedule(static)
				for (size_t i = 0; i < nElems; ++i) {
					int e = elems[i];
					double Ee = eleModulus[e];
					for (int a=0; a<8; ++a) {
						int n = mesh.eNodMat[e*8 + a];
						for (int dim=0; dim<3; ++dim) {
							int local = 3*a + dim;
							double val = Ke[local*24 + local] * Ee;
							// Race-free update due to coloring
							diagFull[3*n + dim] += val;
						}
					}
				}
			}
		}
	} else {
		// Fallback: Atomic updates
		#pragma omp parallel for
		for (int e=0; e<mesh.numElements; ++e) {
			double Ee = eleModulus[e];
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
	}

	for (double& v : diagFull) if (!(v > 0.0)) v = 1.0;
}


static void MG_CoarsenDiagonal(const MGLevel& Lc, const MGLevel& Lf,
								  const std::vector<double>& diagFine,
								  const std::vector<uint8_t>& fineFixedMask,
								  const std::vector<uint8_t>& coarseFixedMask,
								  std::vector<double>& diagCoarse) {
    const int span = Lc.spanWidth;
    const int grid = span+1;

	diagCoarse.assign((Lc.resX+1)*(Lc.resY+1)*(Lc.resZ+1)*3, 0.0);
	std::vector<double> wsum(diagCoarse.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return (Lf.resX+1)*(Lf.resY+1)*iz + (Lf.resX+1)*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return (Lc.resX+1)*(Lc.resY+1)*iz + (Lc.resX+1)*iy + ix; };

	// OPTIMIZATION 2: 8-Color loop for coarse grid to avoid atomics
	for (int color=0; color<8; ++color) {
		#pragma omp parallel for collapse(3)
		for (int cez=0; cez<Lc.resZ; ++cez) {
			for (int cex=0; cex<Lc.resX; ++cex) {
				for (int cey=0; cey<Lc.resY; ++cey) {
					// Check color parity (XYZ)
					int myColor = (cex&1) | ((cey&1)<<1) | ((cez&1)<<2);
					if (myColor != color) continue;

	                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
					for (int iz=0; iz<=span; ++iz) {
						for (int iy=0; iy<=span; ++iy) {
							for (int ix=0; ix<=span; ++ix) {
								int fxi = fx0 + ix; int fyi = fy0 - iy; int fzi = fz0 + iz;
								int fn = idxF(fxi, fyi, fzi);
	                            const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
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
									double w2 = W[a]*W[a];
									for (int c=0;c<3;c++) {
										const int fineD = 3*fn + c;
										const int coarseD = 3*cidx[a] + c;
										if (coarseFixedMask[coarseD]) continue;
										if (fineFixedMask[fineD]) continue;

										// No atomics needed because adjacent active elements don't share nodes
										diagCoarse[coarseD] += w2 * diagFine[fineD];
										wsum[coarseD] += w2;
									}
								}
							}
						}
					}
				}
			}
		}
	} // end color loop

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
							   const std::vector<double>& eleModulus,
							   std::vector<std::vector<double>>& diagLevels) {
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
