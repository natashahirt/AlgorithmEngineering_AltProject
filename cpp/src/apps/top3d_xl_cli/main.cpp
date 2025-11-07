#include "top3d_xl/core.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>

int main(int argc, char** argv) {
    // Usage:
    // ./top3d_xl_cli GLOBAL        nely nelx nelz V0 nLoop
    // ./top3d_xl_cli LOCAL         nely nelx nelz Ve0 nLoop rMin rHat
    // ./top3d_xl_cli GLOBAL_TV     filePath V0 nLoop rMin
    // ./top3d_xl_cli LOCAL_TV      filePath Ve0 nLoop rMin rHat
    if (argc < 2) { std::cerr << "Usage: GLOBAL/LOCAL/GLOBAL_TV/LOCAL_TV ..." << std::endl; return 1; }
	std::string mode = argv[1];
	if (mode == "GLOBAL") {
		if (argc < 7) { std::cerr << "GLOBAL args: nely nelx nelz V0 nLoop" << std::endl; return 1; }
		int nely = std::atoi(argv[2]);
		int nelx = std::atoi(argv[3]);
		int nelz = std::atoi(argv[4]);
		double V0 = std::atof(argv[5]);
		int nLoop = std::atoi(argv[6]);
		std::cout << "Running GLOBAL topo-opt on cuboid: "
				  << nely << "x" << nelx << "x" << nelz
				  << ", V0=" << V0 << ", iters=" << nLoop << "\n";
		top3d::TOP3D_XL_GLOBAL(nely, nelx, nelz, V0, nLoop, std::sqrt(3.0));
	} else if (mode == "LOCAL") {
		if (argc < 9) { std::cerr << "LOCAL args: nely nelx nelz Ve0 nLoop rMin rHat" << std::endl; return 1; }
		int nely = std::atoi(argv[2]);
		int nelx = std::atoi(argv[3]);
		int nelz = std::atoi(argv[4]);
		double Ve0 = std::atof(argv[5]);
		int nLoop = std::atoi(argv[6]);
		double rMin = std::atof(argv[7]);
		double rHat = std::atof(argv[8]);
		std::cout << "Running LOCAL (PIO) topo-opt on cuboid: "
				  << nely << "x" << nelx << "x" << nelz
				  << ", Ve0=" << Ve0 << ", iters=" << nLoop
				  << ", rMin=" << rMin << ", rHat=" << rHat << "\n";
		top3d::TOP3D_XL_LOCAL(nely, nelx, nelz, Ve0, nLoop, rMin, rHat);
    } else if (mode == "GLOBAL_TV" || mode == "LOCAL_TV") {
        std::cerr << "TopVoxel modes are disabled in this build." << std::endl;
        return 1;
    } else {
		std::cerr << "Unknown mode: " << mode << std::endl; return 1;
	}
	return 0;
}


