#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPGRO_DIR="$SCRIPT_DIR/../DADES/topgro_actu"
PARTITIONS=(highmem normal1 normal2 normal3 normal4 normal5)

if [ ! -d "$TOPGRO_DIR" ]; then
    echo "Directory not found: $TOPGRO_DIR"
    exit 1
fi

partition_idle_cpus() {
    local partition="$1"
    sinfo -h -p "$partition" -o "%C" 2>/dev/null | awk -F/ '{idle += $2} END {print idle + 0}'
}

partition_slots=()
for partition in "${PARTITIONS[@]}"; do
    idle_cpus="$(partition_idle_cpus "$partition")"
    echo "Partition $partition: $idle_cpus idle CPUs"

    for ((i = 0; i < idle_cpus; i++)); do
        partition_slots+=("$partition")
    done
done

if [ "${#partition_slots[@]}" -eq 0 ]; then
    echo "No idle CPUs detected with sinfo; falling back to one slot per partition."
    partition_slots=("${PARTITIONS[@]}")
fi

job_index=0
for f in "$TOPGRO_DIR"/*_GMX.gro; do
    [ -f "$f" ] || continue

    mol="$(basename "$f" _GMX.gro)"
    partition="${partition_slots[$((job_index % ${#partition_slots[@]}))]}"

    echo "Submitting: $mol -> $partition"
    sbatch --partition="$partition" \
        --cpus-per-task=1 \
        --job-name="$mol" \
        --output="sortid_%j_${mol}.txt" \
        --error="error_%j_${mol}.txt" \
        "$SCRIPT_DIR/enviar.sh" "$mol"

    job_index=$((job_index + 1))
done

echo "All SLURM jobs submitted."