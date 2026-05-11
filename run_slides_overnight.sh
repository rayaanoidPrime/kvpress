#!/usr/bin/env bash
# Run ALL remaining slides sequentially. Use: bash run_slides_overnight.sh
set -euo pipefail

VENV_PYTHON=".venv/Scripts/python.exe"
SLIDE_DIR="experimental/slide_scripts"
RESULTS=""
FAILURES=""

run_slide() {
    local name="$1"
    local script="${SLIDE_DIR}/${name}.py"
    echo ""
    echo "===== $(date '+%H:%M:%S') SLIDE: $name ====="
    if "$VENV_PYTHON" "$script"; then
        echo "===== $(date '+%H:%M:%S') DONE: $name ====="
        RESULTS="$RESULTS $name"
    else
        echo "===== $(date '+%H:%M:%S') FAILED: $name ====="
        FAILURES="$FAILURES $name"
    fi
}

echo "==========================================="
echo "KVPRESS SLIDE GENERATION - OVERNIGHT RUNNER"
echo "Started: $(date)"
echo "==========================================="


# # Phase 2: Heavy model-dependent slides
# run_slide s07_attention_kl
# run_slide s08_eviction

# Phase 3: PPL sweeps (heaviest)
run_slide s15_ppl_sweep
run_slide s16_ppl_vs_bits

# Phase 4: More model runs
run_slide s17_crossover
run_slide s18_needle

echo ""
echo "==========================================="
echo "COMPLETED: $(date)"
echo "==========================================="
echo "Success: $RESULTS"
echo "Failed:  $FAILURES"
