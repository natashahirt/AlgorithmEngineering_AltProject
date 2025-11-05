#!/bin/bash
# Job Flags
#SBATCH -p mit_normal
#SBATCH --job-name=top3d_GLOBAL
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

# Set mode and parameters (modify these as needed)
MODE="GLOBAL"
# For GLOBAL mode: nely nelx nelz V0 nLoop (rMin is hardcoded to sqrt(3))
# For LOCAL mode: nely nelx nelz Ve0 nLoop rMin rHat
PARAMS="50 100 50 0.12 50"

# Generate datetime string (YYYYMMDD_HHMMSS)
DATETIME=$(date +"%Y%m%d_%H%M%S")
# Create unique output filenames in log subdirectory
OUTPUT_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.out"
ERROR_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.err"
LOG_FILE="out/log/${MODE}_latest.log"

# Ensure out directory structure exists
mkdir -p out/log
mkdir -p out/stl

# Redirect all output to unique files (tee allows monitoring with tail -f)
exec > >(tee "$OUTPUT_FILE" | tee "$LOG_FILE")
exec 2> >(tee "$ERROR_FILE" >&2)

echo "Job started at $(date)"
echo "Mode: $MODE"
echo "Parameters: $PARAMS"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo "Latest log: $LOG_FILE (use 'tail -f $LOG_FILE' to monitor)"
echo "STL files will be saved to: out/stl/"
echo ""

# Build if executable doesn't exist
if [ ! -f build/top3d_xl_cli ]; then
    echo "Building top3d_xl_cli..."
    cd build
    cmake ../cpp
    cmake --build . -j
    cd ..
fi

# Run the topology optimization
./build/top3d_xl_cli $MODE $PARAMS

echo "Job completed at $(date)"