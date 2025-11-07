#include "opt/MMAseq.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

// This file implements a sequential MMA subproblem and solver that mirrors
// the PETSc TopOpt implementation by Aage & Lazarov. See:
// - MMA.h: https://github.com/topopt/TopOpt_in_PETSc/blob/master/MMA.h
// - MMA.cc: https://github.com/topopt/TopOpt_in_PETSc/blob/master/MMA.cc
// License: https://github.com/topopt/TopOpt_in_PETSc/blob/master/lesser.txt

namespace top3d {
namespace mma {

namespace {

struct MMAContext {
	int n=0;
	int m=0;
	double asyminit = 0.5;
	double asymdec  = 0.7;
	double asyminc  = 1.2;
	int RobustAsymptotesType = 0; // 0 default, 1 robust
	bool constraintModification = false; // PETSc default = false
	std::vector<double> a, c, d;
	std::vector<double> y, lam, mu, grad, s;
	double z = 0.0;
	std::vector<double> L, U, alpha, beta;
	std::vector<double> p0, q0;
	std::vector<std::vector<double>> pij, qij;
	std::vector<double> b;
	std::vector<double> xo1, xo2, xVal;
	std::vector<double> pjlam, qjlam;
	std::vector<std::vector<double>> Hess;
	int k = 0; // iteration counter used for asymptote initialization

	void init_sizes(int mm, int nn) {
		m = mm; n = nn;
		a.assign(m, 0.0);
		c.assign(m, 1000.0);
		d.assign(m, 0.0);
		y.assign(m, 0.0);
		lam.assign(m, 0.0);
		mu.assign(m, 0.0);
		grad.assign(m, 0.0);
		s.assign(2*m, 0.0);
		L.assign(n, 0.0);
		U.assign(n, 0.0);
		alpha.assign(n, 0.0);
		beta.assign(n, 0.0);
		p0.assign(n, 0.0);
		q0.assign(n, 0.0);
		pij.assign(m, std::vector<double>(n, 0.0));
		qij.assign(m, std::vector<double>(n, 0.0));
		b.assign(m, 0.0);
		xo1.assign(n, 0.0);
		xo2.assign(n, 0.0);
		xVal.assign(n, 0.0);
		pjlam.assign(n, 0.0);
		qjlam.assign(n, 0.0);
		Hess.assign(m, std::vector<double>(m, 0.0));
	}

	void GenSub(const std::vector<double>& dfdx_row,
				const std::vector<double>& gx_col,
				const std::vector<std::vector<double>>& dgdx,
				const std::vector<double>& xmin,
				const std::vector<double>& xmax) {
		k++;
		// Initialize asymptotes
		for (int j=0;j<n;j++) {
			L[j] = xVal[j] - asyminit*(xmax[j] - xmin[j]);
			U[j] = xVal[j] + asyminit*(xmax[j] - xmin[j]);
		}
		// Dynamic asymptote update for k>2 using xo1/xo2, mirroring PETSc logic
		if (k > 2) {
			for (int i=0;i<n;i++) {
				double helpvar = (xVal[i] - xo1[i]) * (xo1[i] - xo2[i]);
				double gamma = 1.0;
				if (helpvar < 0.0) gamma = asymdec;
				else if (helpvar > 0.0) gamma = asyminc;
				L[i] = xVal[i] - gamma * (xo1[i] - L[i]);
				U[i] = xVal[i] + gamma * (U[i] - xo1[i]);
				double xmi = std::max(1.0e-5, xmax[i] - xmin[i]);
				if (RobustAsymptotesType == 0) {
					L[i] = std::max(L[i], xVal[i] - 10.0 * xmi);
					L[i] = std::min(L[i], xVal[i] - 1.0e-2 * xmi);
					U[i] = std::max(U[i], xVal[i] + 1.0e-2 * xmi);
					U[i] = std::min(U[i], xVal[i] + 10.0 * xmi);
				} else if (RobustAsymptotesType == 1) {
					L[i] = std::max(L[i], xVal[i] - 100.0 * xmi);
					L[i] = std::min(L[i], xVal[i] - 1.0e-4 * xmi);
					U[i] = std::max(U[i], xVal[i] + 1.0e-4 * xmi);
					U[i] = std::min(U[i], xVal[i] + 100.0 * xmi);
					double xbmin = xmin[i] - 1.0e-5;
					double xbmax = xmax[i] + 1.0e-5;
					if (xVal[i] < xbmin) {
						L[i] = xVal[i] - (xbmax - xVal[i]) / 0.9;
						U[i] = xVal[i] + (xbmax - xVal[i]) / 0.9;
					}
					if (xVal[i] > xbmax) {
						L[i] = xVal[i] - (xVal[i] - xbmin) / 0.9;
						U[i] = xVal[i] + (xVal[i] - xbmin) / 0.9;
					}
				}
			}
		}
		const double feps = 1.0e-6;
		for (int j=0;j<n;j++) {
			alpha[j] = 0.9*L[j] + 0.1*xVal[j];
			if (alpha[j] - xmin[j] < 0.0) alpha[j] = xmin[j];
			beta[j]  = 0.9*U[j] + 0.1*xVal[j];
			if (beta[j] - xmax[j] > 0.0) beta[j] = xmax[j];
		}
		for (int j=0;j<n;j++) {
			double dfdxp = std::max(0.0, dfdx_row[j]);
			double dfdxm = std::max(0.0, -dfdx_row[j]);
			double denom = std::max(1e-16, U[j]-L[j]);
			p0[j] = (U[j]-xVal[j])*(U[j]-xVal[j])*(dfdxp + 0.001*std::abs(dfdx_row[j]) + 0.5*feps/denom);
			q0[j] = (xVal[j]-L[j])*(xVal[j]-L[j])*(dfdxm + 0.001*std::abs(dfdx_row[j]) + 0.5*feps/denom);
		}
		for (int i=0;i<m;i++) {
			for (int j=0;j<n;j++) {
				double gp = std::max(0.0, dgdx[i][j]);
				double gm = std::max(0.0, -dgdx[i][j]);
				if (constraintModification) {
					double denom = std::max(1e-16, U[j]-L[j]);
					pij[i][j] = (U[j]-xVal[j])*(U[j]-xVal[j])*(gp + 0.001*std::abs(dgdx[i][j]) + 0.5*feps/denom);
					qij[i][j] = (xVal[j]-L[j])*(xVal[j]-L[j])*(gm + 0.001*std::abs(dgdx[i][j]) + 0.5*feps/denom);
				} else {
					pij[i][j] = (U[j]-xVal[j])*(U[j]-xVal[j]) * gp;
					qij[i][j] = (xVal[j]-L[j])*(xVal[j]-L[j]) * gm;
				}
			}
		}
		for (int i=0;i<m;i++) {
			double ssum=0.0;
			for (int j=0;j<n;j++) {
				ssum += pij[i][j]/std::max(1e-16, U[j]-xVal[j]) + qij[i][j]/std::max(1e-16, xVal[j]-L[j]);
			}
			b[i] = -gx_col[i] + ssum;
		}
	}

	void XYZofLAMBDA() {
		for (int i=0;i<m;i++) if (lam[i] < 0.0) lam[i] = 0.0;
		for (int i=0;i<m;i++) {
			y[i] = lam[i] - c[i];
			if (y[i] < 0.0) y[i] = 0.0;
		}
		double lamai=0.0;
		for (int i=0;i<m;i++) lamai += lam[i]*a[i];
		z = std::max(0.0, 10.0*(lamai - 1.0));
		for (int j=0;j<n;j++) {
			double sp = p0[j], sq = q0[j];
			for (int i=0;i<m;i++) { sp += pij[i][j]*lam[i]; sq += qij[i][j]*lam[i]; }
			pjlam[j] = sp; qjlam[j] = sq;
		}
		for (int j=0;j<n;j++) {
			double sp = std::sqrt(std::max(0.0, pjlam[j]));
			double sq = std::sqrt(std::max(0.0, qjlam[j]));
			double den = std::max(1e-16, sp+sq);
			double xv = (sp*L[j] + sq*U[j]) / den;
			if (xv - alpha[j] < 0.0) xv = alpha[j];
			if (xv - beta[j]  > 0.0) xv = beta[j];
			xVal[j] = xv;
		}
	}

	void DualGrad() {
		for (int i=0;i<m;i++) grad[i] = -b[i] - a[i]*z - y[i];
		for (int i=0;i<m;i++) {
			double ssum=0.0;
			for (int j=0;j<n;j++) {
				ssum += pij[i][j]/std::max(1e-16, U[j]-xVal[j]) + qij[i][j]/std::max(1e-16, xVal[j]-L[j]);
			}
			grad[i] += ssum;
		}
	}

	void DualHess() {
		for (int j=0;j<n;j++) {
			double sp = p0[j], sq = q0[j];
			for (int i=0;i<m;i++) { sp += pij[i][j]*lam[i]; sq += qij[i][j]*lam[i]; }
			pjlam[j] = sp; qjlam[j] = sq;
		}
		std::vector<double> df2(n, 0.0), xp(n, 0.0);
		for (int j=0;j<n;j++) {
			double den = 2.0*pjlam[j]/std::pow(std::max(1e-16, U[j]-xVal[j]), 3.0)
					   + 2.0*qjlam[j]/std::pow(std::max(1e-16, xVal[j]-L[j]), 3.0);
			df2[j] = (den > 0.0) ? -1.0/den : 0.0;
			double sp = std::sqrt(std::max(0.0, pjlam[j]));
			double sq = std::sqrt(std::max(0.0, qjlam[j]));
			double d  = std::max(1e-16, sp+sq);
			xp[j] = (sp*L[j] + sq*U[j]) / d;
			if (xp[j] - alpha[j] < 0.0) df2[j] = 0.0;
			if (xp[j] - beta[j]  > 0.0) df2[j] = 0.0;
		}
		for (int r=0;r<m;r++) for (int c=0;c<m;c++) Hess[r][c] = 0.0;
		for (int j=0;j<n;j++) {
			if (df2[j] == 0.0) continue;
			std::vector<double> PQcol(m, 0.0);
			for (int i=0;i<m;i++) {
				PQcol[i] = pij[i][j]/std::max(1e-16, (U[j]-xVal[j])*(U[j]-xVal[j]))
						 - qij[i][j]/std::max(1e-16, (xVal[j]-L[j])*(xVal[j]-L[j]));
			}
			for (int r=0;r<m;r++) {
				for (int c=0;c<m;c++) Hess[r][c] += PQcol[r] * (df2[j] * PQcol[c]);
			}
		}
		double lamai=0.0;
		for (int j=0;j<m;j++) {
			if (lam[j] < 0.0) lam[j] = 0.0;
			lamai += lam[j]*a[j];
			if (lam[j] > c[j]) Hess[j][j] -= 1.0;
			Hess[j][j] -= mu[j]/std::max(1e-16, lam[j]);
		}
		if (lamai > 0.0) {
			for (int r=0;r<m;r++) for (int c=0;c<m;c++) Hess[r][c] -= 10.0 * a[r] * a[c];
		}
		double tr=0.0; for (int i=0;i<m;i++) tr += Hess[i][i];
		double corr = 1e-4 * (tr / double(m));
		if (-1.0*corr < 1.0e-7) corr = -1.0e-7;
		for (int i=0;i<m;i++) Hess[i][i] += corr;
	}

	void SolveDIP() {
		lam.assign(m, 0.0);
		for (int i=0;i<m;i++) lam[i] = c[i]/2.0;
		mu.assign(m, 1.0);
		const double tol = 1.0e-9*std::sqrt(double(m + n));
		double epsi = 1.0;
		double err  = 1.0;

		auto Factorize = [&](std::vector<double>& K, int nn) {
			for (int ss = 0; ss < nn - 1; ss++) {
				for (int i = ss + 1; i < nn; i++) {
					K[i*nn + ss] = K[i*nn + ss] / std::max(1e-16, K[ss*nn + ss]);
					for (int j = ss + 1; j < nn; j++) {
						K[i*nn + j] = K[i*nn + j] - K[i*nn + ss] * K[ss*nn + j];
					}
				}
			}
		};

		auto Solve = [&](std::vector<double>& K, std::vector<double>& x, int nn) {
			for (int i = 1; i < nn; i++) {
				double a = 0.0;
				for (int j = 0; j < i; j++) a -= K[i*nn + j] * x[j];
				x[i] += a;
			}
			x[nn - 1] = x[nn - 1] / std::max(1e-16, K[(nn - 1)*nn + (nn - 1)]);
			for (int i = nn - 2; i >= 0; i--) {
				double a = x[i];
				for (int j = i + 1; j < nn; j++) a -= K[i*nn + j] * x[j];
				x[i] = a / std::max(1e-16, K[i*nn + i]);
			}
		};

		auto DualResidual = [&](double epsi_local)->double {
			std::vector<double> res(2*m, 0.0);
			for (int j = 0; j < m; j++) {
				for (int i = 0; i < n; i++) {
					res[j] += pij[j][i]/std::max(1e-16, U[i]-xVal[i]) + qij[j][i]/std::max(1e-16, xVal[i]-L[i]);
				}
			}
			for (int j=0;j<m;j++) {
				res[j] += -b[j] - a[j]*z - y[j] + mu[j];
				res[m+j] = mu[j]*lam[j] - epsi_local;
			}
			double nrI = 0.0;
			for (int i=0;i<2*m;i++) nrI = std::max(nrI, std::abs(res[i]));
			return nrI;
		};

		while (epsi > tol) {
			int loop = 0;
			err = 1.0;
			while (err > 0.9*epsi && loop < 100) {
				loop++;
				XYZofLAMBDA();
				DualGrad();
				for (int i=0;i<m;i++) grad[i] = -1.0*grad[i] - epsi/std::max(1e-16, lam[i]);
				DualHess();
				// Flatten Hess for in-place LU factorization as in PETSc code
				std::vector<double> Hflat(m*m, 0.0);
				for (int r=0;r<m;r++) for (int c=0;c<m;c++) Hflat[r*m + c] = Hess[r][c];
				std::vector<double> step = grad;
				Factorize(Hflat, m);
				Solve(Hflat, step, m);
				for (int i=0;i<m;i++) s[i] = step[i];
				for (int i=0;i<m;i++) s[m+i] = -mu[i] + epsi/std::max(1e-16, lam[i]) - s[i]*mu[i]/std::max(1e-16, lam[i]);
				// Dual line search exactly as PETSc
				double theta = 1.005;
				for (int i=0;i<m;i++) {
					double t1 = -1.01*s[i]   / std::max(1e-16, lam[i]);
					double t2 = -1.01*s[i+m] / std::max(1e-16, mu[i]);
					if (theta < t1) theta = t1;
					if (theta < t2) theta = t2;
				}
				theta = 1.0 / theta;
				for (int i=0;i<m;i++) { lam[i] += theta*s[i]; mu[i] += theta*s[m+i]; }
				XYZofLAMBDA();
				err = DualResidual(epsi);
			}
			epsi *= 0.1;
		}
	}
};

inline void reshape_dgdx_transpose(int n, int m, const std::vector<double>& dgdx_colMajor, std::vector<std::vector<double>>& out_mxn) {
	out_mxn.assign(m, std::vector<double>(n, 0.0));
	for (int i=0;i<m;i++) {
		for (int j=0;j<n;j++) {
			size_t idx = size_t(j) + size_t(n)*size_t(i);
			out_mxn[i][j] = (idx < dgdx_colMajor.size() ? dgdx_colMajor[idx] : 0.0);
		}
	}
}

} // namespace

void MMAseq(int m, int n,
			const std::vector<double>& xvalTmp,
			const std::vector<double>& xmin,
			const std::vector<double>& xmax,
			std::vector<double>& xold1,
			std::vector<double>& xold2,
			const std::vector<double>& dfdx_row,
			const std::vector<double>& gx_col,
			const std::vector<double>& dgdx_colMajor,
			std::vector<double>& xnew_out) {
	MMAContext ctx;
	ctx.init_sizes(m, n);
	ctx.xo1 = xold1;
	ctx.xo2 = xold2;
	ctx.xVal = xvalTmp;
	std::vector<std::vector<double>> dgdx_mxn;
	reshape_dgdx_transpose(n, m, dgdx_colMajor, dgdx_mxn);
	ctx.GenSub(dfdx_row, gx_col, dgdx_mxn, xmin, xmax);
	ctx.xo2 = ctx.xo1;
	ctx.xo1 = ctx.xVal;
	ctx.SolveDIP();
	xnew_out = ctx.xVal;
	xold1 = ctx.xo1;
	xold2 = ctx.xo2;
}

} // namespace mma
} // namespace top3d


