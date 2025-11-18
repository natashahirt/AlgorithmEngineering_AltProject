// Transfer operators (prolongation and restriction)
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <vector>
// Optional OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace top3d { namespace mg {

// ===== MG diagonal-only V-cycle (Adapted) =====

void MG_Prolongate_nodes(const MGLevel& Lc, const MGLevel& Lf,
								  const std::vector<double>& xc, std::vector<double>& xf) {
	const int fnnx = Lf.resX+1, fnny = Lf.resY+1, fnnz = Lf.resZ+1;
	const int cnnx = Lc.resX+1, cnny = Lc.resY+1, cnnz = Lc.resZ+1;
const int span = Lc.spanWidth;
	const int grid = span+1;

	xf.assign(fnnx*fnny*fnnz, 0.0);
	std::vector<double> wsum(xf.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return fnnx*fnny*iz + fnnx*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return cnnx*cnny*iz + cnnx*iy + ix; };

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
				int c1 = idxC(cex,   Lc.resY-cey,   cez);
				int c2 = idxC(cex+1, Lc.resY-cey,   cez);
				int c3 = idxC(cex+1, Lc.resY-cey-1, cez);
				int c4 = idxC(cex,   Lc.resY-cey-1, cez);
				int c5 = idxC(cex,   Lc.resY-cey,   cez+1);
				int c6 = idxC(cex+1, Lc.resY-cey,   cez+1);
				int c7 = idxC(cex+1, Lc.resY-cey-1, cez+1);
				int c8 = idxC(cex,   Lc.resY-cey-1, cez+1);
				double cv[8] = {xc[c1],xc[c2],xc[c3],xc[c4],xc[c5],xc[c6],xc[c7],xc[c8]};

                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
				for (int iz=0; iz<=span; ++iz) {
					for (int iy=0; iy<=span; ++iy) {
						for (int ix=0; ix<=span; ++ix) {
							int fxi = fx0 + ix;
							int fyi = fy0 - iy;
							int fzi = fz0 + iz;
							int fidx = idxF(fxi, fyi, fzi);
const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
							double sum = 0.0; for (int a=0;a<8;a++) sum += W[a]*cv[a];
#ifdef _OPENMP
							#pragma omp atomic
							xf[fidx] += sum;
							#pragma omp atomic
							wsum[fidx] += 1.0;
#else
							xf[fidx] += sum; wsum[fidx] += 1.0;
#endif
						}
					}
				}
			}
		}
	}
	// Normalize accumulated contributions. Parallelize if OpenMP available.
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i=0;i<xf.size();++i) if (wsum[i]>0) xf[i] /= wsum[i];
}

void MG_Restrict_nodes(const MGLevel& Lc, const MGLevel& Lf,
								 const std::vector<double>& rf, std::vector<double>& rc) {
	const int fnnx = Lf.resX+1, fnny = Lf.resY+1, fnnz = Lf.resZ+1;
	const int cnnx = Lc.resX+1, cnny = Lc.resY+1, cnnz = Lc.resZ+1;
const int span = Lc.spanWidth;
	const int grid = span+1;

	rc.assign(cnnx*cnny*cnnz, 0.0);
	std::vector<double> wsum(rc.size(), 0.0);

	auto idxF = [&](int ix,int iy,int iz){ return fnnx*fnny*iz + fnnx*iy + ix; };
	auto idxC = [&](int ix,int iy,int iz){ return cnnx*cnny*iz + cnnx*iy + ix; };

	for (int cez=0; cez<Lc.resZ; ++cez) {
		for (int cex=0; cex<Lc.resX; ++cex) {
			for (int cey=0; cey<Lc.resY; ++cey) {
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

                int fx0 = cex*span, fy0 = (Lc.resY - cey)*span, fz0 = cez*span;
				for (int iz=0; iz<=span; ++iz) {
					for (int iy=0; iy<=span; ++iy) {
						for (int ix=0; ix<=span; ++ix) {
							int fxi = fx0 + ix;
							int fyi = fy0 - iy;
							int fzi = fz0 + iz;
							int fidx = idxF(fxi, fyi, fzi);
const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
							double val = rf[fidx];
							for (int a=0;a<8;a++) { rc[cidx[a]] += W[a]*val; wsum[cidx[a]] += W[a]; }
						}
					}
				}
			}
		}
	}
	for (size_t i=0;i<rc.size();++i) if (wsum[i]>0) rc[i] /= wsum[i];
}

} } // namespace top3d::mg
