#!/usr/bin/env bash
#
# Quick performance test - runs once and shows timing
# Useful for rapid iteration during development
#

set -e

# Default to small problem
NELY=${1:-10}
NELX=${2:-20}
NELZ=${3:-10}
V0=${4:-0.12}
NLOOP=${5:-10}

echo "========================================="
echo "  Quick Performance Test"
echo "========================================="
echo "Problem: ${NELY}x${NELX}x${NELZ}, V0=${V0}, ${NLOOP} iterations"
echo ""

# Run and capture output
OUTPUT=$(./top3d_xl_cli GLOBAL $NELY $NELX $NELZ $V0 $NLOOP 1 2>&1)

# Extract timing info
SOLVER_TIME=$(echo "$OUTPUT" | grep "total solver time:" | awk '{print $4}')
ITER_TIME=$(echo "$OUTPUT" | grep "time per iter:" | awk '{print $4}')
PCT_CG=$(echo "$OUTPUT" | grep "percentage time spent on cg:" | awk '{print $6}')
TOTAL_TIME=$(echo "$OUTPUT" | grep "Total runtime time for" | grep -oE '[0-9]+\.[0-9]+e[+-][0-9]+' | tail -1)
THREADS=$(echo "$OUTPUT" | grep -A1 "THREADS" | tail -1 | xargs)

# Display results
echo "Results:"
echo "  Total Solver Time (CG): ${SOLVER_TIME} s"
echo "  Time per Iteration:     ${ITER_TIME} s"
echo "  Total Runtime:          ${TOTAL_TIME} s"
echo "  Threads Used:           ${THREADS}"
echo "  % Time in CG:           ${PCT_CG}%"
echo ""
echo "✓ Test complete"
