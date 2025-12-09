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

module purge
module load matlab/matlab-2024a

cd /home/nhirt/AlgorithmEngineering_AltProject

MATLAB_BIN="$(command -v matlab)"
if [ -z "$MATLAB_BIN" ]; then
    echo "matlab binary not found in PATH after module load" >&2
    exit 1
fi

# --- Setup Output Directory (Safety Check) ---
mkdir -p "$OUT_DIR_NAME/log" "$OUT_DIR_NAME/stl" "$OUT_DIR_NAME/out"

echo "Running MATLAB for $NELY $NELX $NELZ"
echo "Output directed to: $OUT_DIR_NAME (via $WORK_DIR/out)"

# Run from within the target output directory so MATLAB writes under ./out
cd "$OUT_DIR_NAME"

"$MATLAB_BIN" -batch "try; \
  maxNumCompThreads(str2double(getenv('SLURM_CPUS_PER_TASK'))); \
  addpath('/home/nhirt/AlgorithmEngineering_AltProject/matlab'); \
  TOP3D_XL(true($NELY,$NELX,$NELZ), 'GLOBAL', 0.12, 50, sqrt(3)); \
catch e; disp(getReport(e,'extended')); exit(1); end"

cd ..

# Cleanup empty error file
ERR_FILE="${OUT_DIR_NAME}/log/MAT_${NELY}x${NELX}x${NELZ}_${SLURM_JOB_ID}.err"
if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
    rm "$ERR_FILE"
fi
