#!/usr/bin/env bash
#
# Portable benchmarking script for TOP3D_XL
# Works on both macOS (Apple Silicon/Intel) and Linux
#
# Usage:
#   ./benchmark.sh [OPTIONS]
#
# Options:
#   -n TRIALS       Number of trials to run (default: 5)
#   -s SIZE         Problem size preset: small/medium/large (default: medium)
#   -c CUSTOM       Custom parameters: "nely nelx nelz V0 nLoop" (overrides -s)
#   -b BASELINE     Path to baseline executable for comparison
#   -o OUTPUT       Output CSV file for results (default: benchmark_results.csv)
#   -h              Show this help message
#
# Examples:
#   ./benchmark.sh -n 10                           # Run 10 trials with medium problem
#   ./benchmark.sh -s large -n 5                   # Run 5 trials with large problem
#   ./benchmark.sh -c "30 60 30 0.12 20"          # Custom problem size
#   ./benchmark.sh -b ./top3d_xl_cli.baseline     # Compare with baseline version
#

set -e  # Exit on error

# Default parameters
TRIALS=5
SIZE="medium"
CUSTOM_PARAMS=""
BASELINE_EXE=""
OUTPUT_CSV="benchmark_results.csv"
CURRENT_EXE="./top3d_xl_cli"

# Color output (works on both macOS and Linux)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# Parse command line arguments
while getopts "n:s:c:b:o:h" opt; do
    case $opt in
        n) TRIALS=$OPTARG ;;
        s) SIZE=$OPTARG ;;
        c) CUSTOM_PARAMS=$OPTARG ;;
        b) BASELINE_EXE=$OPTARG ;;
        o) OUTPUT_CSV=$OPTARG ;;
        h)
            head -n 20 "$0" | tail -n +3 | sed 's/^# \?//'
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Problem size presets
declare -A PRESETS
PRESETS[small]="10 20 10 0.12 10"
PRESETS[medium]="30 60 30 0.12 30"
PRESETS[large]="50 100 50 0.12 50"
PRESETS[xlarge]="80 160 80 0.12 80"

# Determine parameters
if [[ -n "$CUSTOM_PARAMS" ]]; then
    PARAMS="$CUSTOM_PARAMS"
else
    if [[ ! -v PRESETS[$SIZE] ]]; then
        echo -e "${RED}Error: Unknown size preset '$SIZE'${NC}" >&2
        echo "Available presets: ${!PRESETS[@]}" >&2
        exit 1
    fi
    PARAMS="${PRESETS[$SIZE]}"
fi

# Parse parameters
read -r NELY NELX NELZ V0 NLOOP <<< "$PARAMS"

# Detect platform
PLATFORM="$(uname -s)"
case "$PLATFORM" in
    Darwin*)
        PLATFORM_NAME="macOS"
        # Check if Apple Silicon
        if [[ "$(uname -m)" == "arm64" ]]; then
            ARCH="Apple Silicon (M1/M2/M3)"
        else
            ARCH="Intel"
        fi
        ;;
    Linux*)
        PLATFORM_NAME="Linux"
        ARCH="$(uname -m)"
        ;;
    *)
        PLATFORM_NAME="$PLATFORM"
        ARCH="$(uname -m)"
        ;;
esac

# Function to extract timing from output
extract_timing() {
    local output="$1"

    # Extract key metrics using grep and awk
    local total_solver_time=$(echo "$output" | grep "total solver time:" | awk '{print $4}')
    local time_per_iter=$(echo "$output" | grep "time per iter:" | awk '{print $4}')
    local pct_cg=$(echo "$output" | grep "percentage time spent on cg:" | awk '{print $6}' | tr -d '%')
    local total_runtime=$(echo "$output" | grep "Total runtime time for" | grep -oE '[0-9]+\.[0-9]+e[+-][0-9]+' | tail -1)
    local num_threads=$(echo "$output" | grep -A1 "THREADS" | tail -1)
    local iterations=$(echo "$output" | grep "iterations:" | awk '{print $2}')

    echo "$total_solver_time|$time_per_iter|$pct_cg|$total_runtime|$num_threads|$iterations"
}

# Function to run benchmark
run_benchmark() {
    local exe_path="$1"
    local label="$2"

    echo -e "${BOLD}Running benchmark: $label${NC}"
    echo "  Executable: $exe_path"
    echo "  Parameters: nely=$NELY nelx=$NELX nelz=$NELZ V0=$V0 nLoop=$NLOOP"
    echo "  Trials: $TRIALS"
    echo ""

    # Arrays to store results
    local -a solver_times
    local -a iter_times
    local -a total_times

    # Run trials
    for trial in $(seq 1 $TRIALS); do
        echo -ne "  Trial $trial/$TRIALS... "

        # Run the program and capture output
        local output=$("$exe_path" GLOBAL $NELY $NELX $NELZ $V0 $NLOOP 1 2>&1)

        # Extract timing
        local timing=$(extract_timing "$output")
        IFS='|' read -r solver_time iter_time pct_cg total_time num_threads iterations <<< "$timing"

        # Store results
        solver_times+=("$solver_time")
        iter_times+=("$iter_time")
        total_times+=("$total_time")

        echo -e "${GREEN}✓${NC} solver=${solver_time}s, per_iter=${iter_time}s, total=${total_time}s"
    done

    echo ""

    # Compute statistics using awk (portable across macOS and Linux)
    local solver_stats=$(printf '%s\n' "${solver_times[@]}" | awk '
        {sum+=$1; sumsq+=$1*$1; if(NR==1){min=$1;max=$1} if($1<min){min=$1} if($1>max){max=$1}}
        END {
            mean=sum/NR;
            stddev=sqrt(sumsq/NR - mean*mean);
            printf "%.6f|%.6f|%.6f|%.6f", mean, stddev, min, max
        }')

    local iter_stats=$(printf '%s\n' "${iter_times[@]}" | awk '
        {sum+=$1; sumsq+=$1*$1; if(NR==1){min=$1;max=$1} if($1<min){min=$1} if($1>max){max=$1}}
        END {
            mean=sum/NR;
            stddev=sqrt(sumsq/NR - mean*mean);
            printf "%.6f|%.6f|%.6f|%.6f", mean, stddev, min, max
        }')

    local total_stats=$(printf '%s\n' "${total_times[@]}" | awk '
        {sum+=$1; sumsq+=$1*$1; if(NR==1){min=$1;max=$1} if($1<min){min=$1} if($1>max){max=$1}}
        END {
            mean=sum/NR;
            stddev=sqrt(sumsq/NR - mean*mean);
            printf "%.6f|%.6f|%.6f|%.6f", mean, stddev, min, max
        }')

    # Parse stats
    IFS='|' read -r solver_mean solver_std solver_min solver_max <<< "$solver_stats"
    IFS='|' read -r iter_mean iter_std iter_min iter_max <<< "$iter_stats"
    IFS='|' read -r total_mean total_std total_min total_max <<< "$total_stats"

    # Display results
    echo -e "${BOLD}Results for $label:${NC}"
    echo "  Total Solver Time (CG only):"
    printf "    Mean:   %9.4f s  (±%.4f s)\n" "$solver_mean" "$solver_std"
    printf "    Range:  %9.4f s  to %.4f s\n" "$solver_min" "$solver_max"
    echo ""
    echo "  Time per Iteration:"
    printf "    Mean:   %9.4f s  (±%.4f s)\n" "$iter_mean" "$iter_std"
    printf "    Range:  %9.4f s  to %.4f s\n" "$iter_min" "$iter_max"
    echo ""
    echo "  Total Runtime:"
    printf "    Mean:   %9.4f s  (±%.4f s)\n" "$total_mean" "$total_std"
    printf "    Range:  %9.4f s  to %.4f s\n" "$total_min" "$total_max"
    echo ""

    # Return mean values for comparison
    echo "$solver_mean|$iter_mean|$total_mean|$num_threads|$iterations"
}

# Print header
echo "========================================"
echo "  TOP3D_XL Benchmark Suite"
echo "========================================"
echo "Platform: $PLATFORM_NAME ($ARCH)"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run current version
CURRENT_RESULTS=$(run_benchmark "$CURRENT_EXE" "Current Version")
IFS='|' read -r current_solver current_iter current_total num_threads iterations <<< "$CURRENT_RESULTS"

# Run baseline if provided
if [[ -n "$BASELINE_EXE" ]]; then
    if [[ ! -x "$BASELINE_EXE" ]]; then
        echo -e "${RED}Error: Baseline executable '$BASELINE_EXE' not found or not executable${NC}" >&2
        exit 1
    fi

    BASELINE_RESULTS=$(run_benchmark "$BASELINE_EXE" "Baseline Version")
    IFS='|' read -r baseline_solver baseline_iter baseline_total _ _ <<< "$BASELINE_RESULTS"

    # Compute speedup
    solver_speedup=$(echo "$baseline_solver $current_solver" | awk '{printf "%.2f", $1/$2}')
    iter_speedup=$(echo "$baseline_iter $current_iter" | awk '{printf "%.2f", $1/$2}')
    total_speedup=$(echo "$baseline_total $current_total" | awk '{printf "%.2f", $1/$2}')

    # Display comparison
    echo "========================================"
    echo -e "${BOLD}  COMPARISON${NC}"
    echo "========================================"
    echo ""
    printf "Metric                  Baseline      Current       Speedup\n"
    echo "------------------------------------------------------------------------"
    printf "Solver Time (CG)     %9.4f s  %9.4f s    ${GREEN}%.2fx${NC}\n" "$baseline_solver" "$current_solver" "$solver_speedup"
    printf "Time per Iteration   %9.4f s  %9.4f s    ${GREEN}%.2fx${NC}\n" "$baseline_iter" "$current_iter" "$iter_speedup"
    printf "Total Runtime        %9.4f s  %9.4f s    ${GREEN}%.2fx${NC}\n" "$baseline_total" "$current_total" "$total_speedup"
    echo ""

    if (( $(echo "$solver_speedup >= 1.1" | bc -l) )); then
        echo -e "${GREEN}✓ Significant speedup achieved! (${solver_speedup}x faster)${NC}"
    elif (( $(echo "$solver_speedup >= 0.95" | bc -l) )); then
        echo -e "${YELLOW}⚠ Performance is comparable (within 5%)${NC}"
    else
        echo -e "${RED}✗ Performance regression detected! (${solver_speedup}x)${NC}"
    fi
    echo ""
fi

# Save to CSV
{
    # Write header if file doesn't exist
    if [[ ! -f "$OUTPUT_CSV" ]]; then
        echo "timestamp,platform,arch,nely,nelx,nelz,V0,nLoop,threads,iterations,trials,solver_mean,solver_std,iter_mean,iter_std,total_mean,total_std,version"
    fi

    # Compute stats again for CSV (we already have them but this is cleaner)
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$PLATFORM_NAME,$ARCH,$NELY,$NELX,$NELZ,$V0,$NLOOP,$num_threads,$iterations,$TRIALS,$current_solver,0.0,$current_iter,0.0,$current_total,0.0,current"
} >> "$OUTPUT_CSV"

echo "Results saved to: $OUTPUT_CSV"
echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
