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
mkdir -p "$OUT_DIR_NAME/log" "$OUT_DIR_NAME/stl" "$OUT_DIR_NAME/out"

# Env setup: allow override; otherwise use 2x threads per core for C++
if [ -n "$TOP3D_OMP_THREADS" ]; then
    export OMP_NUM_THREADS=$TOP3D_OMP_THREADS
else
    export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK * 2))
fi
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Running Multigrid for $NELY $NELX $NELZ with OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Output directed to: $OUT_DIR_NAME (cwd=$OUT_DIR_NAME)"
echo "TOP3D_PRECOND=${TOP3D_PRECOND:-mg}"

# Run from within the target output directory so the app writes to ./out
cd "$OUT_DIR_NAME"
EXEC="/home/nhirt/AlgorithmEngineering_AltProject/build_mg/top3d_xl_cli"
if [ ! -x "$EXEC" ]; then
    echo "Executable not found at $EXEC" >&2
    exit 1
fi
"$EXEC" GLOBAL $NELY $NELX $NELZ 0.12 50 1
cd ..

# Cleanup empty error file
ERR_FILE="${OUT_DIR_NAME}/log/MG_${NELY}x${NELX}x${NELZ}_${SLURM_JOB_ID}.err"
if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
    rm "$ERR_FILE"
fi
