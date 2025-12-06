
// Coarsest-level assembly and Cholesky helpers
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include <vector>
#include <cmath>
#include <algorithm> // Required for std::fill

namespace top3d { namespace mg {

// ===== Coarsest-level assembly and Cholesky (to mirror MATLAB direct solve) =====

bool chol_spd_inplace(std::vector<double>& A, int N) {
	// Optimization: Fallback to serial for small matrices to avoid OpenMP overhead
	// The threshold is heuristic; 256 is a conservative guess where parallelism might start paying off.
	if (N < 256) {
		for (int i=0;i<N;i++) {
			for (int j=0;j<=i;j++) {
				double sum = A[i*N + j];
				// Vectorized dot product
				#pragma omp simd reduction(-:sum)
				for (int k=0;k<j;k++) sum -= A[i*N + k]*A[j*N + k];
				
				if (i==j) {
					if (sum <= 0.0) return false;
					A[i*N + j] = std::sqrt(sum);
				} else {
					A[i*N + j] = sum / A[j*N + j];
				}
			}
			// Zero upper triangle for consistency (optional)
			for (int j=i+1; j<N; j++) A[i*N + j] = 0.0;
		}
		return true;
	}

	// Parallel column-oriented Cholesky factorization
	// Uses "Left-looking" (or Crout/Cholesky-Banachiewicz) variant adapted for column-wise parallelism
	// to maximize cache locality (dot products) and parallelize column updates.
	bool success = true;
	#pragma omp parallel shared(A, success)
	{
		for (int j = 0; j < N; j++) {
			if (!success) continue;

			// 1. Compute diagonal element (Single Thread)
			#pragma omp single
			{
				double sum = A[j*N + j];
				// Vectorized dot product for diagonal
				#pragma omp simd reduction(-:sum)
				for (int k = 0; k < j; k++) {
					sum -= A[j*N + k] * A[j*N + k];
				}

				if (sum <= 0.0) {
					success = false;
				} else {
					A[j*N + j] = std::sqrt(sum);
				}
			} // Implicit barrier

			if (!success) continue;

			double invDiag = 1.0 / A[j*N + j];

			// 2. Compute off-diagonal elements in column j (Parallel)
			// Each thread computes a subset of rows i > j for the current column j.
			#pragma omp for schedule(static)
			for (int i = j + 1; i < N; i++) {
				double sum = A[i*N + j];
				// Vectorized dot product using row i and row j (both accessed with stride 1 = good cache)
				#pragma omp simd reduction(-:sum)
				for (int k = 0; k < j; k++) {
					sum -= A[i*N + k] * A[j*N + k];
				}
				A[i*N + j] = sum * invDiag;
			}
		}
	}

	if (!success) return false;

	// Clean upper triangle (parallelized)
	#pragma omp parallel for
	for (int i=0; i<N; i++) {
		for (int j=i+1; j<N; j++) A[i*N + j] = 0.0;
	}
	return true;
}

void chol_solve_lower(const std::vector<double>& L,
							  const std::vector<double>& b,
							  std::vector<double>& x, int N) {
	// Optimized forward/backward substitution
	// Uses x as workspace to minimize allocations
	if (x.size() != (size_t)N) x.resize(N);
	// Copy b to x
	for(int i=0; i<N; ++i) x[i] = b[i];

	// Forward substitution: L * y = b (y stored in x)
	for (int i=0;i<N;i++) {
		double sum = x[i];
		// Vectorized dot product (stride 1 access for L)
		#pragma omp simd reduction(-:sum)
		for (int k=0;k<i;k++) sum -= L[i*N+k]*x[k];
		x[i] = sum / L[i*N+i];
	}

	// Backward substitution: L^T * x = y
	// Uses "Saxpy" / column-oriented update to access L row-wise (stride 1)
	// instead of standard dot product which would access L column-wise (stride N).
	for (int i=N-1;i>=0;i--) {
		x[i] /= L[i*N+i];
		const double xi = x[i];
		// Update previous RHS values: x[k] -= L[i,k] * x[i]
		#pragma omp simd
		for (int k=0;k<i;k++) {
			x[k] -= L[i*N+k] * xi;
		}
	}
}


// Expand compact element moduli to full structured fine grid order (level 0)
void EleMod_CompactToFull_Finest(const Problem& pb,
										const std::vector<double>& eleModCompact,
										std::vector<double>& eleModFull) {
	int nx = pb.mesh.resX, ny = pb.mesh.resY, nz = pb.mesh.resZ;
	eleModFull.assign(nx*ny*nz, 0.0);
	for (int e=0; e<pb.mesh.numElements; ++e) {
		int full = pb.mesh.eleMapBack[e];
		if (full >= 0 && full < (int)eleModFull.size()) eleModFull[full] = eleModCompact[e];
	}
}

// Assemble coarsest dense K using Galerkin triple-products with BC at fine level
void MG_AssembleCoarsestDenseK_Galerkin(const Problem& pb,
											   const MGHierarchy& H,
											   const std::vector<double>& eleModFineFull,
											   const std::vector<uint8_t>& fineFixedDofMask,
											   std::vector<double>& Kc) {
	const MGLevel& Lf = H.levels.front();
	const MGLevel& Lc = H.levels.back();
	const int N = 3*Lc.numNodes;
	
    // OPTIMIZATION 1: Direct Global Write.
	if ((int)Kc.size() != N*N) Kc.resize(N*N);
    // Parallel zero-fill
    #pragma omp parallel for schedule(static)
    for (int i=0; i<N*N; ++i) Kc[i] = 0.0;

	const int s = Lc.spanWidth;
	const int grid = s + 1;

	auto idxElemF = [&](int ex,int ey,int ez)->int {
		return (Lf.resY*Lf.resX)*ez + (Lf.resY)*ex + (Lf.resY - 1 - ey);
	};

    // OPTIMIZATION 2: Cache Templates
    // Precompute T^T * Ke * T templates once and reuse until grid changes.
    static std::vector<std::vector<double>> templates;
    static int cached_s = -1;
    
    // Check if we need to (re)compute templates
    bool need_recompute = (cached_s != s) || templates.empty();
    
    if (need_recompute) {
        templates.resize(s*s*s);
        cached_s = s;

        // Serial template construction (happens rarely)
        for (int iz=0; iz<s; ++iz) {
            for (int iy=0; iy<s; ++iy) {
                for (int ix=0; ix<s; ++ix) {
                    int idx = (iz*s + iy)*s + ix;
                    templates[idx].assign(24*24, 0.0);
                    
                    // Build T (24x24) for this sub-position
                    double T[24*24]; std::fill(std::begin(T), std::end(T), 0.0);
                    for (int v=0; v<8; ++v) {
                        int vx = (v==0||v==3||v==4||v==7) ? 0 : 1;
                        int vy = (v==0||v==1||v==4||v==5) ? 0 : 1;
                        int vz = (v<=3) ? 0 : 1;
                        const double* W = &Lc.weightsNode[(((iz+vz)*grid + (iy+vy))*grid + (ix+vx))*8];
                        for (int a=0; a<8; ++a) {
                            for (int c=0;c<3;c++) {
                                T[(3*v+c)*24 + (3*a+c)] = W[a];
                            }
                        }
                    }
                    
                    // M = Ke * T
                    double M[24*24];
                    for (int i=0;i<24;i++) {
                        for (int j=0;j<24;j++) {
                            double sum=0.0;
                            // Vectorize
                            #pragma omp simd reduction(+:sum)
                            for (int k=0;k<24;k++) sum += pb.mesh.Ke[i*24+k] * T[k*24+j];
                            M[i*24+j] = sum;
                        }
                    }
                    // Ksub = T^T * M
                    for (int i=0;i<24;i++) {
                        for (int j=0;j<24;j++) {
                            double sum=0.0;
                            #pragma omp simd reduction(+:sum)
                            for (int k=0;k<24;k++) sum += T[k*24+i] * M[k*24+j];
                            templates[idx][i*24+j] = sum;
                        }
                    }
                }
            }
        }
    }

    // OPTIMIZATION 3: Full Parallel Assembly with Atomics
    // Removed coloring barriers. Threads process elements in parallel and atomic-add to global matrix.
    // Also restricted to Lower Triangle updates for 2x reduction in atomic ops.
	#pragma omp parallel for collapse(3) schedule(guided)
    for (int cez=0; cez<Lc.resZ; ++cez) {
        for (int cex=0; cex<Lc.resX; ++cex) {
            for (int cey=0; cey<Lc.resY; ++cey) {
                // coarse element global dofs
                int c_dof[24];
                for (int a=0;a<8;a++) {
                    int n = Lc.eNodMat[(cez*Lc.resX*Lc.resY + cex*Lc.resY + cey)*8 + a];
                    c_dof[3*a+0] = 3*n+0;
                    c_dof[3*a+1] = 3*n+1;
                    c_dof[3*a+2] = 3*n+2;
                }
                double Kce[24*24]; std::fill(std::begin(Kce), std::end(Kce), 0.0);

                int fx0 = cex*s;
                int fy0 = (Lc.resY - cey)*s;
                int fz0 = cez*s;
                
                // iterate s^3 sub-elements
                for (int iz=0; iz<s; ++iz) {
                    for (int iy=0; iy<s; ++iy) {
                        for (int ix=0; ix<s; ++ix) {
                            int fex = fx0 + ix;
                            int fey = fy0 - iy - 1;
                            int fez = fz0 + iz;
                            int ef = idxElemF(fex, fey, fez);
                            double Ee = std::max((double)pb.params.youngsModulusMin, eleModFineFull[ef]);

                            // Check for BC mask on this fine element
                            bool isMasked = false;
                            for (int v=0; v<8; ++v) {
                                int n_f = Lf.eNodMat[ef*8 + v];
                                if (fineFixedDofMask[3*n_f] | fineFixedDofMask[3*n_f+1] | fineFixedDofMask[3*n_f+2]) {
                                    isMasked = true; break;
                                }
                            }

                            if (!isMasked) {
                                // FAST PATH: Use precomputed template
                                int tplIdx = (iz*s + iy)*s + ix;
                                const double* Ksub = templates[tplIdx].data();
                                #pragma omp simd
                                for (int i=0; i<24*24; ++i) Kce[i] += Ee * Ksub[i];
                            } else {
                                // SLOW PATH: Full assembly for masked elements
                                double Kf[24*24];
                                for (int i=0;i<24*24;i++) Kf[i] = Ee * pb.mesh.Ke[i];
                                
                                for (int v=0; v<8; ++v) {
                                    int n_f = Lf.eNodMat[ef*8 + v];
                                    for (int c=0;c<3;c++) {
                                        if (fineFixedDofMask[3*n_f + c]) {
                                            int d = 3*v + c;
                                            for (int j=0;j<24;j++) { Kf[d*24+j] = 0.0; Kf[j*24+d] = 0.0; }
                                            Kf[d*24 + d] = 1.0;
                                        }
                                    }
                                }
                                double T[24*24]; std::fill(std::begin(T), std::end(T), 0.0);
                                for (int v=0; v<8; ++v) {
                                    int vx = (v==0||v==3||v==4||v==7) ? 0 : 1;
                                    int vy = (v==0||v==1||v==4||v==5) ? 0 : 1;
                                    int vz = (v<=3) ? 0 : 1;
                                    const double* W = &Lc.weightsNode[(((iz+vz)*grid + (iy+vy))*grid + (ix+vx))*8];
                                    for (int a=0; a<8; ++a) {
                                        for (int c=0;c<3;c++) T[(3*v+c)*24 + (3*a+c)] = W[a];
                                    }
                                }
                                double M[24*24];
                                for (int i=0;i<24;i++) {
                                    for (int j=0;j<24;j++) {
                                        double ssum=0.0;
                                        for (int k=0;k<24;k++) ssum += Kf[i*24+k]*T[k*24+j];
                                        M[i*24+j] = ssum;
                                    }
                                }
                                for (int i=0;i<24;i++) {
                                    for (int j=0;j<24;j++) {
                                        double ssum=0.0;
                                        for (int k=0;k<24;k++) ssum += T[k*24+i]*M[k*24+j];
                                        Kce[i*24 + j] += ssum;
                                    }
                                }
                            } // end masked
                        }
                    }
                }
                // Scatter Kce to GLOBAL using ATOMICS
                // Only write Lower Triangle (gi >= gj)
                for (int i=0;i<24;i++) {
                    int gi = c_dof[i];
                    for (int j=0;j<24;j++) {
                        int gj = c_dof[j];
                        if (gi >= gj) {
                            #pragma omp atomic update
                            Kc[gi*N + gj] += Kce[i*24 + j];
                        }
                    }
                }
            }
        }
    }
}

} } // namespace top3d::mg