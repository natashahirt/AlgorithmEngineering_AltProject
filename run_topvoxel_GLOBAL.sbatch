#!/bin/bash
# Job Flags
#SBATCH -p mit_normal
#SBATCH --job-name=top_GLOBAL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# Set up environment
module load gcc/12.2.0
module load cmake/3.27.9

# Change to project directory
cd /home/nhirt/AlgorithmEngineering_AltProject

# Mode and parameters
MODE="GLOBAL_TV"
FILE="/home/nhirt/AlgorithmEngineering_AltProject/inputs/Femur.TopVoxel"
# GLOBAL_FromTopVoxel(file, V0, nLoop, rMin); use rMin = sqrt(3)
PARAMS="0.40 50 1.7320508075688772"

# Datetime and logs
DATETIME=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.out"
ERROR_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.err"
LOG_FILE="out/log/${MODE}_latest.log"

# Ensure out directories
mkdir -p out/log
mkdir -p out/stl

# Redirect output
exec > >(tee "$OUTPUT_FILE" | tee "$LOG_FILE")
exec 2> >(tee "$ERROR_FILE" >&2)

echo "Job started at $(date)"
echo "Mode: $MODE"
echo "File: $FILE"
echo "Parameters: $PARAMS"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo "Latest log: $LOG_FILE (tail -f $LOG_FILE)"
echo "STL out: out/stl/"
echo "NODELIST: ${SLURM_JOB_NODELIST:-$SLURM_NODELIST}"
echo ""

# Build if needed
if [ ! -f build/top3d_xl_cli ]; then
    echo "Building top3d_xl_cli..."
    cd build
    cmake ../cpp
    cmake --build . -j
    cd ..
fi

# Run
./build/top3d_xl_cli "$MODE" "$FILE" $PARAMS

echo "Job completed at $(date)"