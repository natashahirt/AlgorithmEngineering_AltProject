# AlgorithmEngineering_AltProject

This is a minimal, matrix-free, finest-level-only C++ port of the MATLAB code provided in `Efficient large-scale 3D topology optimization with matrix-free MATLAB code
 (Wang, Aage, Wu, Sigmund, and Westermann, 2025)`. 
 
The codebase provides:

- Cartesian voxel mesh generation for a solid cuboid
- Built-in boundary conditions (fixed x=0 face, -Z load on x=max face lower slice)
- Matrix-free PCG solve on the finest level using a placeholder isotropic 24x24 element stiffness
- SIMPLE SIMP update and OC volume constraint handling
- CLI to run a demo

## Repository structure

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── apps
│   │   └── top3d_cli.cpp
│   ├── include
│   │   ├── core.hpp
│   │   ├── driver.hpp
│   │   ├── fea.hpp
│   │   ├── filter.hpp
│   │   ├── io.hpp
│   │   ├── solver.hpp
│   │   ├── geometry_out
│   │   │   ├── export_stl.hpp
│   │   │   └── voxel_surface.hpp
│   │   └── multigrid
│   │       ├── multigrid.hpp
│   │       ├── padding.hpp
│   │       └── detail/
│   └── src
│       ├── core.cpp
│       ├── fea.cpp
│       ├── solver.cpp
│       ├── filter.cpp
│       ├── io.cpp
│       ├── driver.cpp
│       └── multigrid
│           ├── coarsest.cpp
│           ├── diagonal.cpp
│           ├── hierarchy.cpp
│           ├── masks.cpp
│           ├── preconditioner.cpp
│           └── transfers.cpp
├── inputs
│   ├── Femur.TopVoxel
│   ├── GEbracket.TopVoxel
│   ├── Molar.TopVoxel
│   └── README.md
├── matlab
│   ├── LICENSE
│   ├── README.md
│   └── TOP3D_XL.m
├── out
│   ├── comments/
│   ├── log/
│   └── stl/
├── run_cubic_GLOBAL.sbatch
├── run_cubic_test_top3d_GLOBAL.sbatch
├── run_MATLAB_TOP3D_XL_GLOBAL.sbatch
└── README.md
```

Build

- Requirements: CMake 3.15+, a C++17 compiler

```
cd cpp
mkdir -p build && cd build
cmake ..
cmake --build . -j
```

Run

```
# from cpp/build
./top3d_xl_cli GLOBAL <nely> <nelx> <nelz> <V0> <nLoop> <simulation_count>
# example
./top3d_xl_cli GLOBAL 30 60 30 0.12 30 5
```