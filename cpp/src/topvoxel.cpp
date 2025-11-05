#include "topvoxel.hpp"
#include "top3d_xl.hpp"

#include <fstream>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <sstream>

namespace top3d {

static void validate_topvoxel_mesh(const Problem& pb) {
    const auto& m = pb.mesh;
    std::ostringstream oss;
    // Basic size checks
    if (m.resX<=0 || m.resY<=0 || m.resZ<=0) {
        oss << "Invalid padded resolution: " << m.resX << "x" << m.resY << "x" << m.resZ;
        std::cerr << "[TopVoxel Validation] " << oss.str() << std::endl; throw std::runtime_error(oss.str());
    }
    if ((int)m.eNodMat.size() != m.numElements*8) {
        oss << "eNodMat size mismatch: eNodMat=" << m.eNodMat.size() << ", expected=" << (m.numElements*8);
        std::cerr << "[TopVoxel Validation] " << oss.str() << std::endl; throw std::runtime_error(oss.str());
    }
    if ((int)m.nodMapBack.size() != m.numNodes) {
        oss << "nodMapBack size mismatch: nodMapBack=" << m.nodMapBack.size() << ", numNodes=" << m.numNodes;
        std::cerr << "[TopVoxel Validation] " << oss.str() << std::endl; throw std::runtime_error(oss.str());
    }
    // eNodMat range check (first offending)
    for (int e=0;e<m.numElements;e++) {
        int base = e*8;
        for (int j=0;j<8;j++) {
            int n = m.eNodMat[base+j];
            if (n < 0 || n >= m.numNodes) {
                oss << "eNodMat out of range at element " << e << ", local " << j << ": node=" << n << ", numNodes=" << m.numNodes;
                std::cerr << "[TopVoxel Validation] " << oss.str() << std::endl; throw std::runtime_error(oss.str());
            }
        }
    }
    // Free DOF array alignment (will be allocated later, just ensure counts consistent)
    const int expectedDOFs = 3*m.numNodes;
    if (expectedDOFs <= 0) {
        oss << "Computed numDOFs invalid: " << expectedDOFs;
        std::cerr << "[TopVoxel Validation] " << oss.str() << std::endl; throw std::runtime_error(oss.str());
    }
}

void CreateVoxelFEAmodel_TopVoxel(Problem& pb, const std::string& path) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("Cannot open TopVoxel file: " + path);

	// Robustly locate Version and Resolution tags like MATLAB reader (skips arbitrary header tokens)
	std::string t;
	double version = 0.0;
	bool gotVersion = false, gotResolution = false;
    int nelx=0, nely=0, nelz=0;
	while (in >> t) {
		if (!gotVersion && t == "Version:") { in >> version; gotVersion = true; }
		else if (t == "Resolution:") { in >> nelx >> nely >> nelz; gotResolution = true; break; }
	}
	if (!gotVersion || version != 1.0) throw std::runtime_error("Unsupported TopVoxel version");
	if (!gotResolution) throw std::runtime_error("Missing Resolution in TopVoxel file");

	// Density included flag (expect two tokens then integer)
	in >> t >> t; int densityIncluded=0; in >> densityIncluded;

	// Solid voxels
	in >> t >> t; int numSolid=0; in >> numSolid;
	std::vector<int> solid; solid.reserve(numSolid);
	std::vector<double> rho; if (densityIncluded) rho.reserve(numSolid);
	if (densityIncluded) {
		for (int i=0;i<numSolid;i++){ int idx; double v; in >> idx >> v; solid.push_back(idx); rho.push_back(v); }
	} else {
		for (int i=0;i<numSolid;i++){ int idx; in >> idx; solid.push_back(idx); }
	}

	// Passive elements
	in >> t; in >> t; int numPassive=0; in >> numPassive;
	std::vector<int> passiveFull(numPassive);
	for (int i=0;i<numPassive;i++) in >> passiveFull[i];

	// Fixations
	in >> t; int numFix=0; in >> numFix;
	std::vector<std::array<int,4>> fixations;
	for (int i=0;i<numFix;i++){ int n,fx,fy,fz; in >> n >> fx >> fy >> fz; fixations.push_back({n,fx,fy,fz}); }

	// Loads (first case)
	in >> t; int numL=0; in >> numL;
	std::vector<std::array<double,4>> lc;
	for (int i=0;i<numL;i++){ int n; double fx,fy,fz; in >> n >> fx >> fy >> fz; lc.push_back({(double)n,fx,fy,fz}); }

	// Additional loads
	in >> t >> t; int numAdd=0; in >> numAdd;
	std::vector<std::vector<std::array<double,4>>> allLoads; allLoads.push_back(lc);
	for (int k=0;k<numAdd;k++){
		in >> t; int nL=0; in >> nL; int idxLoad=0; in >> idxLoad;
		std::vector<std::array<double,4>> lck;
		for (int i=0;i<nL;i++){ int n; double fx,fy,fz; in >> n >> fx >> fy >> fz; lck.push_back({(double)n,fx,fy,fz}); }
		allLoads.push_back(std::move(lck));
	}

    // Build mesh from solid list (MATLAB volume shape [nely x nelx x nelz])
	CartesianMesh& mesh = pb.mesh;
    mesh.eleSize = {1.0,1.0,1.0};

    // ===== Padding to match MATLAB (dyadic-friendly multigrid) =====
    const int COARSEST_RES_CTRL = 50000; // matches MATLAB coarsestResolutionControl_
    // Estimate numLevels based on number of solid voxels (like MATLAB)
    long long numSolidVoxels = (long long)solid.size();
    int numLevels = 0;
    while (numSolidVoxels >= COARSEST_RES_CTRL) { numLevels++; numSolidVoxels = (numSolidVoxels+7)/8; }
    numLevels = std::max(3, numLevels);
    const int spanPow = 1 << numLevels; // 2^numLevels
    const int adjNelx = ((nelx + spanPow - 1) / spanPow) * spanPow;
    const int adjNely = ((nely + spanPow - 1) / spanPow) * spanPow;
    const int adjNelz = ((nelz + spanPow - 1) / spanPow) * spanPow;

    // Padded element volume (zeros outside original extents)
    std::vector<uint8_t> matVolPad((long long)adjNelx*adjNely*adjNelz, 0);
    for (int v : solid) {
        int idx0 = v - 1; if (idx0 < 0) continue;
        int ez = idx0 / (nely*nelx);
        int rem = idx0 % (nely*nelx);
        int ex = rem / nely;
        int ey = rem % nely;
        if (ex < adjNelx && ey < adjNely && ez < adjNelz) {
            long long idxPad0 = (long long)ey + (long long)ex*adjNely + (long long)ez*adjNely*adjNelx;
            matVolPad[idxPad0] = 1;
        }
    }

    // Use padded resolution
    mesh.resX = adjNelx; mesh.resY = adjNely; mesh.resZ = adjNelz;

    // Identify solid elements on padded grid
    mesh.eleMapBack.clear();
    for (int ez=0; ez<adjNelz; ++ez){
        for (int ex=0; ex<adjNelx; ++ex){
            for (int ey=0; ey<adjNely; ++ey){
                long long idxPad0 = (long long)ey + (long long)ex*adjNely + (long long)ez*adjNely*adjNelx;
                if (matVolPad[idxPad0]){
                    int fullIdx = adjNely*adjNelx*ez + adjNely*ex + (adjNely-1 - ey);
                    mesh.eleMapBack.push_back(fullIdx);
                }
            }
        }
    }
    mesh.numElements = (int)mesh.eleMapBack.size();
    mesh.eleMapForward.assign(adjNely*adjNelx*adjNelz, 0);
    for (int i=0;i<mesh.numElements;i++) mesh.eleMapForward[ mesh.eleMapBack[i] ] = i+1;

    const int nx=adjNelx, ny=adjNely, nz=adjNelz;
    const int nnx = nx+1, nny = ny+1, nnz = nz+1;
	// Build element-to-node connectivity using full-grid node indexing (no compression)
	mesh.eNodMat.resize(mesh.numElements*8);
	for (int ez=0; ez<nz; ++ez){
		for (int ex=0; ex<nx; ++ex){
			for (int ey=0; ey<ny; ++ey){
				int fullIdx = ny*nx*ez + ny*ex + (ny-1 - ey);
				int comp = mesh.eleMapForward[fullIdx]; if (comp==0) continue;
				int eComp = comp-1;
				auto nodeIndex=[&](int ix,int iy,int iz){ return (nnx*nny*iz + nnx*iy + ix); };
				int n1=nodeIndex(ex,ny-ey,ez), n2=nodeIndex(ex+1,ny-ey,ez),
				    n3=nodeIndex(ex+1,ny-ey-1,ez), n4=nodeIndex(ex,ny-ey-1,ez),
				    n5=nodeIndex(ex,ny-ey,ez+1), n6=nodeIndex(ex+1,ny-ey,ez+1),
				    n7=nodeIndex(ex+1,ny-ey-1,ez+1), n8=nodeIndex(ex,ny-ey-1,ez+1);
				int base=eComp*8;
				mesh.eNodMat[base+0]=n1; mesh.eNodMat[base+1]=n2; mesh.eNodMat[base+2]=n3; mesh.eNodMat[base+3]=n4;
				mesh.eNodMat[base+4]=n5; mesh.eNodMat[base+5]=n6; mesh.eNodMat[base+6]=n7; mesh.eNodMat[base+7]=n8;
			}
		}
	}
	// Full-grid node mapping (identity) to align with structured multigrid
	const int fullNodes = nnx*nny*nnz;
	mesh.nodMapBack.resize(fullNodes);
	std::iota(mesh.nodMapBack.begin(), mesh.nodMapBack.end(), 0);
	mesh.nodMapForward.resize(fullNodes);
	for (int i=0;i<fullNodes;i++) mesh.nodMapForward[i] = i;
	mesh.numNodes = fullNodes;
	mesh.numDOFs = 3*mesh.numNodes;
	// Boundary identification on full grid considering only used nodes
	std::vector<int> nodDegree(mesh.numNodes,0);
	for (int e=0;e<mesh.numElements;e++) for (int j=0;j<8;j++) nodDegree[ mesh.eNodMat[e*8+j] ]++;
	mesh.nodesOnBoundary.clear();
	for (int i=0;i<mesh.numNodes;i++) if (nodDegree[i]>0 && nodDegree[i]<8) mesh.nodesOnBoundary.push_back(i);
	std::vector<uint8_t> isB(mesh.numNodes,0); for (int v: mesh.nodesOnBoundary) isB[v]=1;
	mesh.elementsOnBoundary.clear();
	for (int e=0;e<mesh.numElements;e++){
		bool onB=false; for (int j=0;j<8;j++) if (isB[mesh.eNodMat[e*8+j]]){ onB=true; break; }
		if (onB) mesh.elementsOnBoundary.push_back(e);
	}

	mesh.Ke = ComputeVoxelKe(0.3, 1.0);

    // Initialize density
    pb.density.assign(mesh.numElements, 1.0);
    if (densityIncluded){
        pb.initialDensityFromFile.assign(mesh.numElements, 1.0);
        int k=0;
        for (int ez=0; ez<nelz; ++ez) for (int ex=0; ex<nelx; ++ex) for (int ey=0; ey<nely; ++ey) {
            // original grid coords -> padded ele index
            long long idxPad0 = (long long)ey + (long long)ex*adjNely + (long long)ez*adjNely*adjNelx;
            // only if original element was marked solid
            if (ex<adjNelx && ey<adjNely && ez<adjNelz && matVolPad[idxPad0]) {
                int fullIdx = adjNely*adjNelx*ez + adjNely*ex + (adjNely-1 - ey);
                int eComp = mesh.eleMapForward[fullIdx]-1; if (eComp>=0 && k < (int)rho.size()) pb.initialDensityFromFile[eComp] = rho[k++];
            }
        }
        for (int e=0;e<mesh.numElements;e++) pb.density[e] = pb.initialDensityFromFile[e];
    }

	// External BC and loads
	pb.extBC = ExternalBC{};
	pb.extBC->fixations = std::move(fixations);

    // Adapt passive elements to padded grid
    pb.extBC->passiveFull = std::move(passiveFull);
    pb.extBC->passiveCompact.clear();
    for (int idx1_based : pb.extBC->passiveFull){
        int idx0 = idx1_based - 1; if (idx0<0) continue;
        int ez = idx0 / (nely*nelx);
        int rem = idx0 % (nely*nelx);
        int ex = rem / nely;
        int ey = rem % nely;
        if (ex>=0 && ex<adjNelx && ey>=0 && ey<adjNely && ez>=0 && ez<adjNelz) {
            int fullIdxPad = adjNely*adjNelx*ez + adjNely*ex + (adjNely-1 - ey);
            int eComp = mesh.eleMapForward[fullIdxPad]-1;
            if (eComp>=0) pb.extBC->passiveCompact.push_back(eComp);
        }
    }

    // ADAPT FIXATION AND LOAD NODE INDICES TO PADDED GRID (1-based -> 1-based padded)
    auto adaptNodeIndex = [&](int node1BasedOrig)->int {
        if (node1BasedOrig <= 0) return node1BasedOrig;
        const int onnx = nelx+1, onny = nely+1, onnz = nelz+1;
        const int pnnx = adjNelx+1, pnny = adjNely+1, pnnz = adjNelz+1;
        int idx0 = node1BasedOrig - 1;
        int iz = idx0 / (onny*onnx);
        int rem = idx0 % (onny*onnx);
        int ix = rem / onny;
        int iy = rem % onny;
        if (ix<0 || ix>=onnx || iy<0 || iy>=onny || iz<0 || iz>=onnz) return -1;
        // Map to padded grid coordinates directly
        // Since padding only adds elements outside the original bounding box,
        // the original nodes have identical coordinates in the padded grid (just at offset zero).
        // Now, check that the mapped index is safely within the padded grid
        if (ix>=pnnx || iy>=pnny || iz>=pnnz) return -1; // padding should ensure this is true
        long long newIdx0 = (long long)iy + (long long)ix*pnny + (long long)iz*pnny*pnnx;
        return (int)(newIdx0 + 1);
    };

    // Adapt fixations
    if (!pb.extBC->fixations.empty()) {
        for (auto &f : pb.extBC->fixations) {
            f[0] = adaptNodeIndex(f[0]);
        }
    }

    // Adapt loads (all cases)
    pb.extLoads = ExternalLoads{};
    pb.extLoads->cases.clear();
    for (auto &lc : allLoads) {
        std::vector<std::array<double,4>> adapted;
        adapted.reserve(lc.size());
        for (auto rec : lc) {
            int n1 = adaptNodeIndex((int)rec[0]);
            if (n1 > 0) adapted.push_back({(double)n1, rec[1], rec[2], rec[3]});
        }
        pb.extLoads->cases.push_back(std::move(adapted));
    }
    pb.extLoads->weights.assign(pb.extLoads->cases.size(), 1.0 / std::max<size_t>(1, pb.extLoads->cases.size()));

    // Validate mesh consistency before allocation
    validate_topvoxel_mesh(pb);

    // Allocate force and DOF vectors
	pb.F.assign(mesh.numDOFs, 0.0);
	pb.isFreeDOF.assign(mesh.numDOFs, 1);
	pb.freeDofIndex.clear();
}

}


