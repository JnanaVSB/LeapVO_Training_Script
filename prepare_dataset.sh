#!/bin/bash
# Download and annotate MOVI-F sequences.
# Each sequence runs in a separate Python process to prevent TF memory leaks.
# Resume-safe: skips sequences that already exist.
#
# Usage:
#   ./prepare_dataset.sh              # default 2000 sequences
#   ./prepare_dataset.sh 3000         # process 3000 sequences
#   ./prepare_dataset.sh 3000 2000    # process indices 2000-2999

NUM_SEQ=${1:-2000}
START_ID=${2:-0}
OUTPUT_DIR="/home/jnana/ASU/PIR/leapvo_training/training_dataset"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

echo "=== MOVI-F Dataset Preparation ==="
echo "Sequences: $START_ID to $((START_ID + NUM_SEQ - 1))"
echo "Output: $OUTPUT_DIR"
echo ""

DONE=0
SKIP=0
FAIL=0
START_TIME=$(date +%s)

for i in $(seq $START_ID $((START_ID + NUM_SEQ - 1))); do
    SEQ_ID=$(printf "%06d" $i)
    SEQ_DIR="$OUTPUT_DIR/$SEQ_ID"
    NPY="$SEQ_DIR/$SEQ_ID.npy"

    # Skip if already done
    if [ -f "$NPY" ] && [ -d "$SEQ_DIR/frames" ]; then
        SKIP=$((SKIP + 1))
        continue
    fi

    T0=$(date +%s)

    python3 "$SCRIPT_DIR/process_one.py" --idx $i --output_dir "$OUTPUT_DIR" 2>/dev/null
    STATUS=$?

    T1=$(date +%s)
    DT=$((T1 - T0))

    if [ $STATUS -eq 0 ] && [ -f "$NPY" ]; then
        DONE=$((DONE + 1))
        TOTAL=$((DONE + SKIP))
        ELAPSED=$((T1 - START_TIME))
        if [ $DONE -gt 0 ] && [ $ELAPSED -gt 0 ]; then
            RATE=$(echo "scale=2; $DONE / $ELAPSED" | bc)
            REMAINING=$((NUM_SEQ - TOTAL))
            ETA=$(echo "scale=0; $REMAINING / $RATE / 60" | bc 2>/dev/null || echo "?")
        else
            RATE="0"
            ETA="?"
        fi
        echo "[$TOTAL/$NUM_SEQ] $SEQ_ID | done | ${DT}s | ${RATE}/s | ETA ${ETA}m"
    else
        FAIL=$((FAIL + 1))
        echo "[$((DONE + SKIP))/$NUM_SEQ] $SEQ_ID | FAILED"
        # Clean up partial
        rm -rf "$OUTPUT_DIR/${SEQ_ID}_tmp" "$SEQ_DIR"
    fi
done

echo ""
echo "========================================="
echo "Complete: $((DONE + SKIP)) sequences"
echo "  New: $DONE"
echo "  Skipped: $SKIP"
echo "  Failed: $FAIL"
echo "  Output: $OUTPUT_DIR"