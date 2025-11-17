#!/bin/bash
#SBATCH -p mit_quicktest
#SBATCH --job-name=test_GLOBAL
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --nodelist=node[1600-1619]

# using 2x48AMD EPYC 9474F 48-Core Processors 

module load gcc/12.2.0
module load cmake/3.27.9

cd /home/nhirt/AlgorithmEngineering_AltProject

MODE="GLOBAL"
PARAMS="12 24 12 0.12 30"

DATETIME=$(date +"%Y%m%d_%H%M%S")
mkdir -p out/log out/stl
OUTPUT_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.out"
ERROR_FILE="out/log/${MODE}_${DATETIME}_${SLURM_JOB_ID}.err"
LOG_FILE="out/log/${MODE}_latest.log"

exec > >(tee "$OUTPUT_FILE" | tee "$LOG_FILE")
exec 2> >(tee "$ERROR_FILE" >&2)

echo "Job started at $(date)"
echo "Mode: $MODE"
echo "Parameters: $PARAMS"
echo "Output file: $OUTPUT_FILE"
echo "Error file: $ERROR_FILE"
echo "Latest log: $LOG_FILE"
echo "STL dir: out/stl/"
echo "NODELIST: ${SLURM_JOB_NODELIST:-$SLURM_NODELIST}"

if [ ! -f build/top3d_xl_cli ]; then
  echo "Building top3d_xl_cli..."
  mkdir -p build
  cd build
  cmake ../cpp
  cmake --build . -j
  cd ..
fi

./build/top3d_xl_cli $MODE $PARAMS

echo "Job completed at $(date)"
