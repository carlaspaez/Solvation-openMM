#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPGRO_DIR="$SCRIPT_DIR/../DADES/topgro_SBC"
CLUSTER_OUT_DIR="$SCRIPT_DIR/../outputs_sbc/cluster"
MOLECULE_LIST="$CLUSTER_OUT_DIR/molecules_sbc.txt"

PARTITIONS=(highmem normal1 normal2 normal3 normal4 normal5)
SLOTS=(42 14 14 14 14 14)
TOTAL_SLOTS=112

if [ ! -d "$TOPGRO_DIR" ]; then
    echo "Directory not found: $TOPGRO_DIR"
    exit 1
fi

mkdir -p "$CLUSTER_OUT_DIR"

find "$TOPGRO_DIR" -maxdepth 1 -type f -name "*_GMX.gro" \
    -printf "%f\n" \
    | sed 's/_GMX\.gro$//' \
    | sort > "$MOLECULE_LIST"

num_molecules="$(wc -l < "$MOLECULE_LIST")"

if [ "$num_molecules" -eq 0 ]; then
    echo "No *_GMX.gro files found in $TOPGRO_DIR"
    exit 1
fi

for partition in "${PARTITIONS[@]}"; do
    : > "$CLUSTER_OUT_DIR/molecules_sbc_${partition}.txt"
done

line_number=0
while IFS= read -r mol; do
    slot=$((line_number % TOTAL_SLOTS))
    offset=0

    for i in "${!PARTITIONS[@]}"; do
        next_offset=$((offset + SLOTS[i]))

        if [ "$slot" -lt "$next_offset" ]; then
            echo "$mol" >> "$CLUSTER_OUT_DIR/molecules_sbc_${PARTITIONS[i]}.txt"
            break
        fi

        offset=$next_offset
    done

    line_number=$((line_number + 1))
done < "$MOLECULE_LIST"

echo "Molecules: $num_molecules"
echo "Molecule list: $MOLECULE_LIST"
echo "Submitting one dispatcher job per partition/node."

for i in "${!PARTITIONS[@]}"; do
    partition="${PARTITIONS[i]}"
    slots="${SLOTS[i]}"
    partition_list="$CLUSTER_OUT_DIR/molecules_sbc_${partition}.txt"
    partition_count="$(wc -l < "$partition_list")"

    if [ "$partition_count" -eq 0 ]; then
        echo "Skipping $partition: no molecules assigned."
        continue
    fi

    echo "Submitting $partition: $partition_count molecules, $slots parallel processes inside one job"

    sbatch \
        --partition="$partition" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$slots" \
        --job-name="SBC_${partition}" \
        --chdir="$SCRIPT_DIR" \
        --output="$CLUSTER_OUT_DIR/sortid_%j_${partition}.txt" \
        --error="$CLUSTER_OUT_DIR/error_%j_${partition}.txt" \
        --wrap="bash \"$SCRIPT_DIR/run_sbc_partition.sh\" \"$partition_list\" \"$slots\" \"$partition\""
done

echo "Submitted SBC dispatcher jobs for all partitions."
