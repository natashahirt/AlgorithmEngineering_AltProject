#!/bin/bash

# --- 1. PREPARATION ---
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "Setting up output directories..."
OUT_DIR="${SCRIPT_DIR}/test_parallel"
mkdir -p "$OUT_DIR/log" "$OUT_DIR/stl"

# Note: The global 'out' symlink is no longer needed/modified here
# because the worker scripts now handle output isolation per-job
# using scratch directories and local symlinks.

# --- 2. BUILD STEP (Do this ONCE before submitting) ---
# Load required modules for building
module load cmake/3.27.9
module load intel-hpc/2025.2.1.44

echo "Building Multigrid executable..."
rm -rf "${ROOT_DIR}/build_mg"
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/build_mg" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/orcd/software/core/001/pkg/intel-hpc/2025.2.1.44/compiler/2025.2/bin/icpx \
      -DCMAKE_CXX_FLAGS="-march=native -qopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++" > /dev/null
cmake --build "${ROOT_DIR}/build_mg" -j > /dev/null || { echo "MG Build failed"; exit 1; }

echo "Building Jacobi executable..."
rm -rf "${ROOT_DIR}/build_jac"
cmake -S "${ROOT_DIR}/cpp" -B "${ROOT_DIR}/build_jac" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=/orcd/software/core/001/pkg/intel-hpc/2025.2.1.44/compiler/2025.2/bin/icpx \
      -DCMAKE_CXX_FLAGS="-march=native -qopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++" > /dev/null
cmake --build "${ROOT_DIR}/build_jac" -j > /dev/null || { echo "Jacobi Build failed"; exit 1; }

# --- 3. SUBMIT SCALING JOBS ---
# Fixed resolution
NY=64
NX=128
NZ=64
DIM_TAG="${NY}x${NX}x${NZ}"

# Sweep CPU counts (1..32); default C++ behavior uses 2x threads per CPU when no override is set
CPU_COUNTS=(1 2 4 8 16 32)
for CPU in "${CPU_COUNTS[@]}"; do
    TAG="${DIM_TAG}_${CPU}cpu"
    echo "Submitting parallelism tests for $DIM_TAG with ${CPU}cpu (default OMP threads) (Output: $OUT_DIR)..."

    sbatch --job-name="MG_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/MG_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/MG_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_ours_multigrid.sbatch" $NY $NX $NZ $OUT_DIR

    TOP3D_PRECOND=jacobi_coarsest \
    sbatch --job-name="JACCOARSE_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/JACCOARSE_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/JACCOARSE_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_ours_multigrid.sbatch" $NY $NX $NZ $OUT_DIR

    TOP3D_PRECOND=jacobi_finest \
    sbatch --job-name="JAC_FINEST_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/JAC_FINEST_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/JAC_FINEST_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_ours_jacobi.sbatch" $NY $NX $NZ $OUT_DIR

    sbatch --job-name="MAT_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/MAT_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/MAT_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_matlab.sbatch" $NY $NX $NZ $OUT_DIR
done

# Additional single-core, single-thread runs for C++ variants (override OMP to 1)
CPU=1
TAG="${DIM_TAG}_${CPU}cpu_1thread"
TOP3D_OMP_THREADS=1 \
    sbatch --job-name="MG_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/MG_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/MG_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_ours_multigrid.sbatch" $NY $NX $NZ $OUT_DIR

TOP3D_PRECOND=jacobi_coarsest TOP3D_OMP_THREADS=1 \
    sbatch --job-name="JACCOARSE_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/JACCOARSE_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/JACCOARSE_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_ours_multigrid.sbatch" $NY $NX $NZ $OUT_DIR

TOP3D_PRECOND=jacobi_finest TOP3D_OMP_THREADS=1 \
    sbatch --job-name="JAC_FINEST_${TAG}" \
           --cpus-per-task=$CPU \
           --output="${OUT_DIR}/log/JAC_FINEST_SCALING_${TAG}_%j.out" \
           --error="${OUT_DIR}/log/JAC_FINEST_SCALING_${TAG}_%j.err" \
           "${SCRIPT_DIR}/job_ours_jacobi.sbatch" $NY $NX $NZ $OUT_DIR

echo "All parallelism jobs submitted to ${OUT_DIR}."

