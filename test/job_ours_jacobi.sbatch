#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodelist=node[1600-1619]

# Arguments: $1=NELY $2=NELX $3=NELZ $4=OUTPUT_DIR_NAME (optional)
NELY=$1
NELX=$2
NELZ=$3
OUT_DIR_NAME=${4:-test}

module load gcc/12.2.0
module load cmake/3.27.9
module load intel-hpc/2025.2.1.44

cd /home/nhirt/AlgorithmEngineering_AltProject

# --- Setup Output Directory (Safety Check) ---
mkdir -p "$OUT_DIR_NAME/log" "$OUT_DIR_NAME/stl"

# Create a job-specific scratch directory to isolate output
WORK_DIR="scratch_jac_${SLURM_JOB_ID}"
mkdir -p "$WORK_DIR"
# Symlink the target output directory to 'out' inside the scratch dir
ln -sfn "../$OUT_DIR_NAME" "$WORK_DIR/out"

# Env setup: Use 2x threads per core for C++
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK * 2))
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Running Jacobi for $NELY $NELX $NELZ with OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Output directed to: $OUT_DIR_NAME (via $WORK_DIR/out)"

# Run from within the scratch directory so app finds ./out
cd "$WORK_DIR"
../build_jac/top3d_xl_cli GLOBAL $NELY $NELX $NELZ 0.12 50 1 0
cd ..

# Cleanup scratch directory
rm -rf "$WORK_DIR"

# Cleanup empty error file
ERR_FILE="${OUT_DIR_NAME}/log/JAC_${NELY}x${NELX}x${NELZ}_${SLURM_JOB_ID}.err"
if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
    rm "$ERR_FILE"
fi
