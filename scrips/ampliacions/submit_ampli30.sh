#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PBC_SCRIPT="$SCRIPT_DIR/deltaG_PBC_TIP3P-ampli30.py"
SBC_SCRIPT="$SCRIPT_DIR/deltaG_SBC_TIP3P_ampli30.py"
PBC_CLUSTER_OUT="$ROOT_DIR/outputs_PBC_ampli30/cluster"
SBC_CLUSTER_OUT="$ROOT_DIR/outputs_SBC_ampli30/cluster"
PYTHON_BIN="/home/10031954/miniconda3/envs/md-openmm/bin/python"

PARTITIONS=(gpu highmem normal1 normal2 normal3 normal4 normal5)
MOLECULES=(mobley_1723043 mobley_9534740 mobley_1893937)

mkdir -p "$PBC_CLUSTER_OUT" "$SBC_CLUSTER_OUT"

partition_slots=("${PARTITIONS[@]}")

submit_job() {
    local method="$1"
    local mol="$2"
    local partition="$3"
    local script="$4"
    local out_dir="$5"

    echo "Submitting ampli30 $method: $mol -> $partition"

    local gpu_args=()
    if [ "$partition" = "gpu" ]; then
        gpu_args=(--gres=gpu:1)
    fi

    sbatch \
        --partition="$partition" \
        --cpus-per-task=1 \
        "${gpu_args[@]}" \
        --job-name="A30_${method}_${mol}" \
        --chdir="$SCRIPT_DIR" \
        --output="$out_dir/sortid_%j_${method}_${mol}.txt" \
        --error="$out_dir/error_%j_${method}_${mol}.txt" \
        --wrap="\"$PYTHON_BIN\" \"$script\" -m \"$mol\" -s SOL.TIP3P.itp"
}

job_index=0
for mol in "${MOLECULES[@]}"; do
    partition="${partition_slots[$((job_index % ${#partition_slots[@]}))]}"
    submit_job "PBC" "$mol" "$partition" "$PBC_SCRIPT" "$PBC_CLUSTER_OUT"
    job_index=$((job_index + 1))

    partition="${partition_slots[$((job_index % ${#partition_slots[@]}))]}"
    submit_job "SBC" "$mol" "$partition" "$SBC_SCRIPT" "$SBC_CLUSTER_OUT"
    job_index=$((job_index + 1))
done

echo "Submitted ampli30 PBC/SBC jobs."
