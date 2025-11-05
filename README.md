# AlgorithmEngineering_AltProject

This is a minimal, matrix-free, finest-level-only C++ port of the MATLAB code provided in `Efficient large-scale 3D topology optimization with matrix-free MATLAB code
 (Wang, Aage, Wu, Sigmund, and Westermann, 2025)`. 
 
The codebase provides:

- Cartesian voxel mesh generation for a solid cuboid
- Built-in boundary conditions (fixed x=0 face, -Z load on x=max face lower slice)
- Matrix-free PCG solve on the finest level using a placeholder isotropic 24x24 element stiffness
- SIMPLE SIMP update and OC volume constraint handling (no PDE filter, no multigrid)
- CLI to run a demo

Build

- Requirements: CMake 3.15+, a C++17 compiler

```
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

Run

```
./top3d_xl_cli [nely nelx nelz V0 nLoop]
# example
./top3d_xl_cli 30 60 30 0.12 30
```