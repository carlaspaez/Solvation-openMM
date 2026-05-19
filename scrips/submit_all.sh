#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPGRO_DIR="$SCRIPT_DIR/../DADES/topgro_actu"

if [ ! -d "$TOPGRO_DIR" ]; then
    echo "Directory not found: $TOPGRO_DIR"
    exit 1
fi

for f in "$TOPGRO_DIR"/*_GMX.gro; do
    [ -f "$f" ] || continue
    mol=$(basename "$f" _GMX.gro)
    echo "Submitting: $mol"
    sbatch --job-name="$mol" \
           --output="sortid_%j_${mol}.txt" \
           --error="error_%j_${mol}.txt" \
           "$SCRIPT_DIR/enviar.sh" "$mol"
done

echo "All SLURM jobs submitted."
