#!/bin/bash

# --- 1. PREPARATION ---
echo "Setting up output directories..."
OUT_DIR="test_scaling"
mkdir -p "$OUT_DIR/log" "$OUT_DIR/stl"

# Note: The global 'out' symlink is no longer needed/modified here
# because the worker scripts now handle output isolation per-job
# using scratch directories and local symlinks.

# --- 2. BUILD STEP (Do this ONCE before submitting) ---
# Load required modules for building
module load cmake/3.27.9
module load intel-hpc/2025.2.1.44

echo "Building Multigrid executable..."
rm -rf build_mg
cmake -S cpp -B build_mg -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/orcd/software/core/001/pkg/intel-hpc/2025.2.1.44/compiler/2025.2/bin/icpx \
      -DCMAKE_CXX_FLAGS="-march=native -qopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++" > /dev/null
cmake --build build_mg -j > /dev/null || { echo "MG Build failed"; exit 1; }

echo "Building Jacobi executable..."
rm -rf build_jac
cmake -S cpp -B build_jac -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/orcd/software/core/001/pkg/intel-hpc/2025.2.1.44/compiler/2025.2/bin/icpx \
      -DCMAKE_CXX_FLAGS="-march=native -qopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++" > /dev/null
cmake --build build_jac -j > /dev/null || { echo "Jacobi Build failed"; exit 1; }

# --- 3. SUBMIT JOBS ---
DOMAINS=(
"30 60 30"
"50 100 50"
"100 200 100"
"150 300 150"
"200 400 200"
"250 500 250"
)

# Explicitly set standard CPU count for domain tests
STD_CPU=16

for D in "${DOMAINS[@]}"; do
    # Extract dimensions for naming
    read -r NY NX NZ <<< "$D"
    DIM_TAG="${NY}x${NX}x${NZ}"
    
    echo "Submitting jobs for domain: $D (Output: $OUT_DIR)"

#     # 1. Multigrid
#     sbatch --job-name="MG_${DIM_TAG}" \
#            --cpus-per-task=$STD_CPU \
#            --output="${OUT_DIR}/log/MG_${DIM_TAG}_%j.out" \
#            --error="${OUT_DIR}/log/MG_${DIM_TAG}_%j.err" \
#            job_ours_multigrid.sbatch $NY $NX $NZ $OUT_DIR

    # 1b. Multigrid with forced Jacobi on coarsest (NlimitDofs=0) for comparison
    TOP3D_JACOBI_COARSEST=1 sbatch --job-name="JACCOARSE_${DIM_TAG}" \
           --cpus-per-task=$STD_CPU \
           --output="${OUT_DIR}/log/JACCOARSE_${DIM_TAG}_%j.out" \
           --error="${OUT_DIR}/log/JACCOARSE_${DIM_TAG}_%j.err" \
           job_ours_multigrid.sbatch $NY $NX $NZ $OUT_DIR

    # 2. Jacobi
#     sbatch --job-name="JAC_FINEST_${DIM_TAG}" \
#            --cpus-per-task=$STD_CPU \
#            --output="${OUT_DIR}/log/JAC_FINEST_${DIM_TAG}_%j.out" \
#            --error="${OUT_DIR}/log/JAC_FINEST_${DIM_TAG}_%j.err" \
#            job_ours_jacobi.sbatch $NY $NX $NZ $OUT_DIR

#     # 3. MATLAB
#     sbatch --job-name="MAT_${DIM_TAG}" \
#            --cpus-per-task=$STD_CPU \
#            --output="${OUT_DIR}/log/MAT_${DIM_TAG}_%j.out" \
#            --error="${OUT_DIR}/log/MAT_${DIM_TAG}_%j.err" \
#            job_matlab.sbatch $NY $NX $NZ $OUT_DIR
done

echo "All jobs submitted to ${OUT_DIR}."
