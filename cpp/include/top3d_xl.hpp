#ifndef TOP3D_XL_HPP
#define TOP3D_XL_HPP

#include <vector>
#include <array>
#include <string>
#include <cstdint>

namespace top3d {

struct GlobalParams {
	// Physical
	double youngsModulus = 1.0;
	double poissonRatio = 0.3;
	double youngsModulusMin = 1.0e-6;
	double simpPenalty = 3.0;
	double cellSize = 1.0;

	// Solver
	double cgTol = 1.0e-3;
	int cgMaxIt = 800;

	// Optimization
	int passiveLayersBoundary = 0;
	int passiveLayersLoads = 0;
	int passiveLayersFixations = 0;
};

struct CartesianMesh {
	int resX = 0;
	int resY = 0;
	int resZ = 0;

	std::array<double,3> eleSize {1.0,1.0,1.0};

	int numElements = 0;
	int numNodes = 0;
	int numDOFs = 0;

	std::vector<int32_t> eleMapBack;      // indices of solid elements (flattened)
	std::vector<int32_t> eleMapForward;   // map from all elements -> compact solid index
	std::vector<int32_t> nodMapBack;      // map compact nodes -> full node ids
	std::vector<int32_t> nodMapForward;   // map full node ids -> compact nodes

	std::vector<int32_t> nodesOnBoundary; // compact nodes indices
	std::vector<int32_t> elementsOnBoundary; // compact element indices

	// Element to node connectivity (compact indices), shape: numElements x 8
	std::vector<int32_t> eNodMat; // stored row-major: [ele][8]

	// Element stiffness (24x24) for unit modulus
	std::array<double,24*24> Ke{};
};

struct Problem {
	GlobalParams params;
	CartesianMesh mesh;

	// Loads: one load case supported in minimal port: numNodes x 3 vector (flattened DOF)
	std::vector<double> F;          // size = numDOFs
	std::vector<uint8_t> isFreeDOF; // size = numDOFs (1 free, 0 fixed)
	std::vector<int> freeDofIndex;  // compact list of free dofs

	// Design variables
	std::vector<double> density;    // per element x, size = numElements
};

struct PDEFilter {
	// 8-node kernel (8x8) stored row-major
	std::array<double,8*8> kernel{};
	std::vector<double> diagPrecondNode; // numNodes
};

// PDE filter setup and application
PDEFilter SetupPDEFilter(const Problem& pb, double filterRadius);
// Apply PDE filtering (element -> node (sum/8), solve, node -> element (sum/8))
void ApplyPDEFilter(const Problem& pb, const PDEFilter& pf, const std::vector<double>& srcEle, std::vector<double>& dstEle);

// Entry points
void InitialSettings(GlobalParams& out);

// Build a simple cuboid boolean model (all true) and discretize
void CreateVoxelFEAmodel_Cuboid(Problem& pb, int nely, int nelx, int nelz);

// Apply built-in boundary conditions similar to MATLAB demo
void ApplyBoundaryConditions(Problem& pb);

// Compute unit hexahedral voxel stiffness matrix
std::array<double,24*24> ComputeVoxelKe(double nu, double cellSize);

// Matrix-free K·u product on finest level
void K_times_u_finest(const Problem& pb, const std::vector<double>& eleModulus,
					   const std::vector<double>& uFull, std::vector<double>& yFull);

// PCG on free DOFs with matrix-free operator
int PCG_free(const Problem& pb,
			  const std::vector<double>& eleModulus,
			  const std::vector<double>& bFree,
			  std::vector<double>& xFree,
			  double tol, int maxIt);

// Optional: Multigrid preconditioner hooks (2-level minimal)
struct MGPrecondConfig { int levels = 1; };
void EnableMultigridPreconditioner(const Problem& pb, const MGPrecondConfig& cfg);

// Compute compliance per element and total
double ComputeCompliance(const Problem& pb,
						   const std::vector<double>& eleModulus,
						   const std::vector<double>& uFull,
						   std::vector<double>& ceList);

// Run GLOBAL topology optimization (simplified, no PDE filter)
void TOP3D_XL_GLOBAL(int nely, int nelx, int nelz, double V0, int nLoop, double rMin);

// Run LOCAL (PIO) topology optimization with Heaviside + p-norm LVF
void TOP3D_XL_LOCAL(int nely, int nelx, int nelz, double Ve0, int nLoop, double rMin, double rHat);

} // namespace top3d

#endif // TOP3D_XL_HPP


