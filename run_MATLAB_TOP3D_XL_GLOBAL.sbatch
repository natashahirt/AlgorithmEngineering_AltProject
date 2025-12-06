#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=matlab_TOP3D_GLOBAL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodelist=node[1600-1619]

# Optional: license server (uncomment and set if required)
# export MLM_LICENSE_FILE=27000@your-license-hostname

# Load MATLAB
module purge
module load matlab/R2023a

# Project directory
cd /home/nhirt/AlgorithmEngineering_AltProject

# Logs
DATETIME=$(date +"%Y%m%d_%H%M%S")
MODE="MATLAB_GLOBAL"
OUTPUT_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.out"
ERROR_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.err"
LOG_FILE="out/log/${MODE}_latest.log"

# Ensure output directories
mkdir -p out/log out/stl

# Redirect output
exec > >(tee "$OUTPUT_FILE" | tee "$LOG_FILE")
exec 2> >(tee "$ERROR_FILE" >&2)

echo "Job started at $(date)"
echo "Mode: $MODE"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo "Latest log: $LOG_FILE (tail -f $LOG_FILE)"
echo "NODELIST: ${SLURM_JOB_NODELIST:-$SLURM_NODELIST}"
echo ""

# Run MATLAB non-interactively
# Limit MATLAB threads to the allocated CPUs, add path, and call TOP3D_XL

module purge
module load matlab/matlab-2024a
which matlab || { echo "MATLAB not on PATH"; exit 1; }

matlab -batch "try; \
  maxNumCompThreads(str2double(getenv('SLURM_CPUS_PER_TASK'))); \
  addpath('/home/nhirt/AlgorithmEngineering_AltProject/matlab'); \
  TOP3D_XL(true(50,100,50), 'GLOBAL', 0.12, 50, sqrt(3)); \
catch e; disp(getReport(e,'extended')); exit(1); end"

echo "Job completed at $(date)"