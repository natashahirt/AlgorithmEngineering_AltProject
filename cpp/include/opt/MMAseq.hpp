#pragma once
#include <vector>
namespace top3d {
namespace mma {

// Exact port of MATLAB's MMAseq used in TOP3D_XL.m.
// Arguments mirror the MATLAB call:
// MMAseq(m, n, xval, xmin, xmax, xold1, xold2, dfdx_row, gx_col, dgdx_colMajor, xnew_out)
// - dfdx_row: size n (row vector of objective gradient)
// - gx_col: size m (column vector of constraint values g(x) <= 0)
// - dgdx_colMajor: flattened size n*m representing reshape(n,m) in MATLAB; we internally
//   transpose to m x n like MATLAB's reshape(...,n,m)'.
void MMAseq(int m, int n,
			const std::vector<double>& xvalTmp,
			const std::vector<double>& xmin,
			const std::vector<double>& xmax,
			std::vector<double>& xold1,
			std::vector<double>& xold2,
			const std::vector<double>& dfdx_row,
			const std::vector<double>& gx_col,
			const std::vector<double>& dgdx_colMajor,
			std::vector<double>& xnew_out);

} // namespace mma
} // namespace top3d


