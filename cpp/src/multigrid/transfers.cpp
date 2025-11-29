// Transfer operators (prolongation and restriction)
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <vector>
#include <iostream>
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
    const size_t minCoarseElemsForParallel = 90; // Need to tune
    const int nFineElements = fnnx*fnny*fnnz;
    const int nCoarseElements = cnnx*cnny*cnnz;

	// final results
    xf.assign(nFineElements, 0.0);
    std::vector<double> wsum(xf.size(), 0.0);
    auto idxF = [&](int ix,int iy,int iz){ return fnnx*fnny*iz + fnnx*iy + ix; };
    auto idxC = [&](int ix,int iy,int iz){ return cnnx*cnny*iz + cnnx*iy + ix; };

    // Decide runtime threading
	#ifdef _OPENMP
		int maxThreads =  omp_get_max_threads();
	#else
		int maxThreads = 1;
	#endif
		bool doParallel = (maxThreads > 1) && (nCoarseElements >= minCoarseElemsForParallel);
		if (doParallel) {
			std::cout << "Incorrectly parallel \n";
			// allocate contiguous thread-local buffers: [tid * Nf + idx]
			std::vector<double> xf_loc((size_t)maxThreads * nFineElements, 0.0);
			std::vector<double> wsum_loc((size_t)maxThreads * nFineElements, 0.0);

			#ifdef _OPENMP
			#pragma omp parallel for collapse(3) if(doParallel) schedule(static)
			#endif
			for (int cez=0; cez<Lc.resZ; ++cez) {
				for (int cex=0; cex<Lc.resX; ++cex) {
					for (int cey=0; cey<Lc.resY; ++cey) {
					#ifdef _OPENMP
					int tid = omp_get_thread_num();
					#else
					int tid = 0;
					#endif
                    size_t base = (size_t)tid * nFineElements;

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
                                size_t fidx = (size_t)idxF(fxi, fyi, fzi);
                                const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
                                double sum = 0.0; for (int a=0;a<8;a++) sum += W[a]*cv[a];

                                xf_loc[base + fidx] += sum;
                                wsum_loc[base + fidx] += 1.0;
                            }
                        }
                    }
                }
            }
        }

        // Final reduction: sum per-thread slices into global xf/wsum and normalize.
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
        for (size_t i = 0; i < nFineElements; ++i) {
            double s = 0.0, ws = 0.0;
            size_t off = i;
            for (int t = 0; t < maxThreads; ++t, off += nFineElements) {
                s += xf_loc[off];
                ws += wsum_loc[off];
            }
            if (ws > 0.0) xf[i] = s / ws;
            else xf[i] = 0.0;
        }

    } else {
        // Serial
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
                                size_t fidx = (size_t)idxF(fxi, fyi, fzi);
                                const double* W = &Lc.weightsNode[((iz*grid+iy)*grid+ix)*8];
                                double sum = 0.0; for (int a=0;a<8;a++) sum += W[a]*cv[a];
                                xf[fidx] += sum;
                                wsum[fidx] += 1.0;
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
