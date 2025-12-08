#include "core.hpp"
#include "driver.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>

int main(int argc, char** argv) {
    // Usage:
    // ./top3d_xl_cli GLOBAL nely nelx nelz V0 nLoop
    if (argc < 8) { std::cerr << "Usage: GLOBAL nely nelx nelz V0 nLoop" << std::endl; return 1; }

    // Force unbuffered output to ensure logs appear immediately in SLURM/CI environments
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    std::string mode = argv[1];
    if (mode != "GLOBAL") {
        std::cerr << "Only GLOBAL mode is supported in this build." << std::endl;
        return 1;
    }

    int nely = std::atoi(argv[2]);
    int nelx = std::atoi(argv[3]);
    int nelz = std::atoi(argv[4]);
    float V0  = std::atof(argv[5]);
    int nLoop  = std::atoi(argv[6]);
    int simulation_count = std::atoi(argv[7]);
    
    // Optional argument for preconditioner: 1=MG (default), 0=Jacobi
    bool use_mg = true;
    if (argc > 8) {
        use_mg = (std::atoi(argv[8]) != 0);
    }

    std::cout << "Running GLOBAL topo-opt on cuboid: "
              << nely << "x" << nelx << "x" << nelz
              << ", V0=" << V0 << ", iters=" << nLoop << "\n";
    std::cout << "Preconditioner: " << (use_mg ? "Multigrid" : "Jacobi") << "\n";

    top3d::TOP3D_XL_GLOBAL(nely, nelx, nelz, V0, nLoop, std::sqrt(3.0f), simulation_count, use_mg);
    return 0;
}
