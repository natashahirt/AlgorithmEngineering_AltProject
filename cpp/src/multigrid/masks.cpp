
// Multigrid masks helpers
#include "core.hpp"
#include "multigrid/multigrid.hpp"
#include "multigrid/detail/transfers.hpp"
#include <vector>

namespace top3d { namespace mg {

// Build per-level fixed-DOF masks by restricting the finest mask
void MG_BuildFixedMasks(const Problem& pb, const MGHierarchy& H,
							   std::vector<std::vector<uint8_t>>& fixedMasks) {
	fixedMasks.resize(H.levels.size());
	// Finest-level mask from pb.isFreeDOF (3 DOFs per node)
	{
		const int n0 = (pb.mesh.resX+1)*(pb.mesh.resY+1)*(pb.mesh.resZ+1);
		fixedMasks[0].assign(3*n0, 0);
		// pb.isFreeDOF is stored in the mesh's current node numbering (Morton after reordering).
		// Convert to lexicographic order for structured MG transfers by mapping lex -> morton.
		for (int n=0;n<pb.mesh.numNodes;n++) {
			int nm = pb.mesh.nodMapForward[n]; // lex index n -> Morton index nm
			for (int c=0;c<3;c++) fixedMasks[0][3*n+c] = pb.isFreeDOF[3*nm+c] ? 0 : 1;
		}
	}
	// Restrict masks to coarser levels (component-wise), threshold 0.5
	for (size_t l=0; l+1<H.levels.size(); ++l) {
		const int fnn = H.levels[l].numNodes;
		const int cnn = H.levels[l+1].numNodes;
		fixedMasks[l+1].assign(3*cnn, 0);
		for (int c=0;c<3;c++) {
			std::vector<double> rf(fnn), rc;
			for (int n=0;n<fnn;n++) rf[n] = fixedMasks[l][3*n+c] ? 1.0 : 0.0;
			MG_Restrict_nodes(H.levels[l+1], H.levels[l], rf, rc);
			for (int n=0;n<cnn;n++) fixedMasks[l+1][3*n+c] = (rc[n] > 0.5) ? 1 : 0;
		}
	}
}

} } // namespace top3d::mg
