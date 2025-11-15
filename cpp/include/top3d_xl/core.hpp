#ifndef TOP3D_XL_CORE_HPP
#define TOP3D_XL_CORE_HPP

#include <vector>
#include <array>
#include <string>
#include <functional>
#include <cstdint>
#include <optional>

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

	// Original (pre-padding) element resolutions
	int origResX = 0;
	int origResY = 0;
	int origResZ = 0;

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
	// Warm-start storage
	std::vector<double> lastXNode;       // numNodes
	std::vector<double> lastRhsNode;     // numNodes
};

// PDE filter setup and application
PDEFilter SetupPDEFilter(const Problem& pb, double filterRadius);
// Apply PDE filtering (element -> node (sum/8), solve, node -> element (sum/8))
void ApplyPDEFilter(const Problem& pb, PDEFilter& pf, const std::vector<double>& srcEle, std::vector<double>& dstEle);

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

// Preconditioner functor: z = M^{-1} r (operates on free-DOF vectors)
using Preconditioner = std::function<void(const std::vector<double>& rFree, std::vector<double>& zFree)>;

// Build Jacobi diagonal on free DOFs (from finest-level element Ke and eleModulus)
void ComputeJacobiDiagonalFree(const Problem& pb,
					   const std::vector<double>& eleModulus,
					   std::vector<double>& diagFree);

// PCG on free DOFs with matrix-free operator and optional preconditioner
int PCG_free(const Problem& pb,
		  const std::vector<double>& eleModulus,
		  const std::vector<double>& bFree,
		  std::vector<double>& xFree,
		  double tol, int maxIt,
		  Preconditioner M = Preconditioner{});

// Optional: Multigrid preconditioner config (diagonal-only V-cycle)
struct MGPrecondConfig {
	bool nonDyadic = true;   // first jump 1->3 (span=4)
	int  maxLevels = 5;      // cap levels
	double weight = 0.6;     // diagonal relaxation factor
};

// Compute compliance per element and total
double ComputeCompliance(const Problem& pb,
					   const std::vector<double>& eleModulus,
					   const std::vector<double>& uFull,
					   std::vector<double>& ceList);

// Run GLOBAL topology optimization (simplified, no PDE filter)
void TOP3D_XL_GLOBAL(int nely, int nelx, int nelz, double V0, int nLoop, double rMin);

// ================= Multigrid scaffolding (Step 2) =================
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

// Build lightweight geometric hierarchy (no coarse Ks yet). Stops when coarse dims <2 or maxLevels.
void BuildMGHierarchy(const Problem& pb, bool nonDyadic, MGHierarchy& H, int maxLevels = 5);

// Build an MG diagonal-only V-cycle preconditioner (captures hierarchy + diagonals)
Preconditioner MakeMGDiagonalPreconditioner(const Problem& pb,
								const std::vector<double>& eleModulus,
								const MGPrecondConfig& cfg);

} // namespace top3d

#endif // TOP3D_XL_CORE_HPP


