# TOP3D_XL (C++) - Minimal Runnable Port

This is a minimal, matrix-free, finest-level-only C++ port of the MATLAB `TOP3D_XL` GLOBAL topology optimization path. It provides:

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

Notes

- This is a deliberately simplified starting point. The LOCAL (PIO) path, PDE filtering, multigrid, NIfTI/STL exports, and visualization are not included in this minimal port.
- The element stiffness matrix is approximated with a symmetric placeholder to keep the example lightweight and dependency-free. Replace `ComputeVoxelKe` with the full 24x24 table if desired.