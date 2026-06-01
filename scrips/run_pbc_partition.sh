#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <molecule-list> <max-parallel> <partition-label>"
    exit 1
fi

MOLECULE_LIST="$1"
MAX_PARALLEL="$2"
PARTITION_LABEL="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_OUT_DIR="$SCRIPT_DIR/../outputs_PBC2/cluster"

if [ ! -f "$MOLECULE_LIST" ]; then
    echo "Molecule list not found: $MOLECULE_LIST"
    exit 1
fi

mkdir -p "$CLUSTER_OUT_DIR"

running=0
index=0

while IFS= read -r mol; do
    [ -n "$mol" ] || continue

    index=$((index + 1))
    echo "[$PARTITION_LABEL] starting $index: $mol"

    (
        set -euo pipefail
        python "$SCRIPT_DIR/deltaG_PBC_TIP3P-ok.py" -m "$mol" -s SOL.TIP3P.itp \
            > "$CLUSTER_OUT_DIR/sortid_${SLURM_JOB_ID}_${PARTITION_LABEL}_${index}_${mol}.txt" \
            2> "$CLUSTER_OUT_DIR/error_${SLURM_JOB_ID}_${PARTITION_LABEL}_${index}_${mol}.txt"
    ) &

    running=$((running + 1))

    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait -n || true
        running=$((running - 1))
    fi
done < "$MOLECULE_LIST"

wait || true

echo "[$PARTITION_LABEL] completed $index molecules"
