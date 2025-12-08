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
PARAMS="50 100 50 0.12 30"
SIMULATION_COUNT="1"

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
echo "Simulation count: $SIMULATION_COUNT"
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

# ====== Profiling with perf: baseline stats + attribution ======
# Build perf output paths
PERF_DIR="out/perf"
mkdir -p "$PERF_DIR"
PERF_TAG="${MODE}_${DATETIME}_${SLURM_JOB_ID}"
PERF_RUN_DIR="${PERF_DIR}/${PERF_TAG}"
mkdir -p "$PERF_RUN_DIR"
PERF_STAT_FILE="${PERF_RUN_DIR}/stat.txt"
PERF_DATA_FILE="${PERF_RUN_DIR}/perf.data"
PERF_REPORT_FILE="${PERF_RUN_DIR}/report.txt"
PERF_ANNOT_FILE="${PERF_RUN_DIR}/annotate_K_times_u_finest.txt"

# Keep runs comparable (multithread; bind places)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Helper: wrap executable and args
APP="./build/top3d_xl_cli"
APP_ARGS="$MODE $PARAMS $SIMULATION_COUNT"


echo ""

if command -v perf >/dev/null 2>&1; then
  echo "perf found. Checking hardware PMU availability..."
  HW_OK=1
  perf stat -e cycles -- true >/dev/null 2>"${PERF_RUN_DIR}/hw_check.err" || HW_OK=0
  if [ "$HW_OK" -eq 1 ]; then
    echo "Hardware PMU accessible; using hardware events and DWARF call stacks."
    # Simplified baseline counters
    STAT_EVENTS="cycles,instructions,cache-references,cache-misses"
    # Simple, robust record event for attribution
    REC_EVENTS="cycles"
    CALLGRAPH="dwarf"
  else
    echo "Hardware PMU blocked; falling back to software events and frame-pointer call stacks."
    # Simplified software counters
    STAT_EVENTS="task-clock,cpu-clock"
    # Simple software clock for attribution
    REC_EVENTS="cpu-clock"
    CALLGRAPH="fp"
  fi
  # Verify record events are supported; if not, fall back further
  if ! perf stat -e "$REC_EVENTS" -- true >/dev/null 2>"${PERF_RUN_DIR}/rec_check.err"; then
    echo "Record event set '$REC_EVENTS' not supported; falling back to cpu-clock." | tee -a "$LOG_FILE"
    REC_EVENTS="cpu-clock"
  fi
  # Avoid taskset on Slurm-managed CPUs (can conflict with cgroups); rely on OMP_* for consistency
  TASKSET_PREFIX=()
  # Optional: set PERF_ENABLE_THREAD_STAT=1 to run a separate per-thread counters pass
  PERF_ENABLE_THREAD_STAT="${PERF_ENABLE_THREAD_STAT:-0}"
  if [ "$PERF_ENABLE_THREAD_STAT" = "1" ]; then
    echo "Running perf stat --per-thread for per-thread counters..."
    "${TASKSET_PREFIX[@]}" perf stat --per-thread -d -d -d \
      -e "$STAT_EVENTS" \
      -o "$PERF_STAT_FILE" -- $APP $APP_ARGS
    echo "perf stat saved to: $PERF_STAT_FILE"
  fi
  echo "Running perf record for call stacks across all threads..."
  "${TASKSET_PREFIX[@]}" perf record -g --call-graph "$CALLGRAPH" -F 99 \
    -e "$REC_EVENTS" \
    -o "$PERF_DATA_FILE" -- $APP $APP_ARGS
  echo "perf record data saved to: $PERF_DATA_FILE"

  echo "Generating text report (perf report --stdio)..."
  if [ -s "$PERF_DATA_FILE" ]; then
    perf report --stdio -i "$PERF_DATA_FILE" > "$PERF_REPORT_FILE" || true
    echo "perf report saved to: $PERF_REPORT_FILE"
  else
    echo "perf.data is empty; skipping perf report." | tee -a "$LOG_FILE"
  fi

  echo "Annotating hotspot functions (if symbols available)..."
  if [ -s "$PERF_DATA_FILE" ]; then
    perf annotate --stdio -i "$PERF_DATA_FILE" -s 'top3d::K_times_u_finest' > "$PERF_ANNOT_FILE" || true
    perf annotate --stdio -i "$PERF_DATA_FILE" -s 'top3d::PCG_free' > "${PERF_RUN_DIR}/annotate_PCG_free.txt" || true
  fi
  # Quick combined summary filtered from report
  if [ -s "$PERF_REPORT_FILE" ]; then
    grep -E 'top3d::K_times_u_finest|top3d::PCG_free' "$PERF_REPORT_FILE" > "${PERF_RUN_DIR}/summary.txt" || true
  fi
  echo "perf outputs saved under: $PERF_RUN_DIR"
else
  echo "perf not found on PATH; running application without profiling."
  ./build/top3d_xl_cli $MODE $PARAMS $SIMULATION_COUNT
fi

# After job is done, delete error file if it's empty/job was successful
if [ ! -s "$ERROR_FILE" ]; then
  rm -f "$ERROR_FILE"
fi

echo "Job completed at $(date)"