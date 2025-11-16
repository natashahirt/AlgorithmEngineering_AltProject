#include "core.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>

int main(int argc, char** argv) {
    // Usage:
    // ./top3d_xl_cli GLOBAL nely nelx nelz V0 nLoop
    if (argc < 7) { std::cerr << "Usage: GLOBAL nely nelx nelz V0 nLoop" << std::endl; return 1; }

    std::string mode = argv[1];
    if (mode != "GLOBAL") {
        std::cerr << "Only GLOBAL mode is supported in this build." << std::endl;
        return 1;
    }

    int nely = std::atoi(argv[2]);
    int nelx = std::atoi(argv[3]);
    int nelz = std::atoi(argv[4]);
    double V0  = std::atof(argv[5]);
    int nLoop  = std::atoi(argv[6]);

    std::cout << "Running GLOBAL topo-opt on cuboid: "
              << nely << "x" << nelx << "x" << nelz
              << ", V0=" << V0 << ", iters=" << nLoop << "\n";
    top3d::TOP3D_XL_GLOBAL(nely, nelx, nelz, V0, nLoop, std::sqrt(3.0));
    return 0;
}