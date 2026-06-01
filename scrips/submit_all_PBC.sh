#!/bin/bash
set -euo pipefail

# Directori on viu aquest script, independentment d'on s'executi.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Entrada: fitxers .gro de les molecules PBC.
TOPGRO_DIR="$SCRIPT_DIR/../DADES/topgro_actu"

# Sortida: llistes de molècules per partició i logs dels jobs de SLURM.
CLUSTER_OUT_DIR="$SCRIPT_DIR/../outputs_PBC2/cluster"
MOLECULE_LIST="$CLUSTER_OUT_DIR/molecules_pbc.txt"

# Particions/nodes disponibles i nombre de processos paral·lels que volem
# executar dins de cada job dispatcher.
PARTITIONS=(highmem normal1 normal2 normal3 normal4 normal5) #numero de particions que te el cluster, sense comptar les GPU
SLOTS=(42 14 14 14 14 14) # els nodes de la partició highmem tenen 42 cpus, els altres 14. S'ha de multiplicar per 1 perquè cada job dispatcher llança un procés que a dins en paral·lel llança fins a "slots" processos.
TOTAL_SLOTS=112

# Comprovem que existeixi el directori d'entrada abans de generar cap fitxer.
if [ ! -d "$TOPGRO_DIR" ]; then
    echo "Directory not found: $TOPGRO_DIR"
    exit 1
fi

mkdir -p "$CLUSTER_OUT_DIR"

# Construïm la llista global de molècules a partir dels fitxers *_GMX.gro.
# El nom de la molècula queda sense el sufix _GMX.gro.
find "$TOPGRO_DIR" -maxdepth 1 -type f -name "*_GMX.gro" \
    -printf "%f\n" \
    | sed 's/_GMX\.gro$//' \
    | sort > "$MOLECULE_LIST"

num_molecules="$(wc -l < "$MOLECULE_LIST")"

if [ "$num_molecules" -eq 0 ]; then
    echo "No *_GMX.gro files found in $TOPGRO_DIR"
    exit 1
fi

# Reiniciem les llistes per partició perquè cada execució sigui neta i no es repeteixil la feina.
for partition in "${PARTITIONS[@]}"; do
    : > "$CLUSTER_OUT_DIR/molecules_pbc_${partition}.txt"
done

# Repartim les molècules en cicles de TOTAL_SLOTS.
# Exemple: els primers 42 noms van a highmem, els següents 14 a normal1, etc.
# Després de 112 molècules, el patró torna a començar.
line_number=0
while IFS= read -r mol; do
    slot=$((line_number % TOTAL_SLOTS))
    offset=0

    # Busquem a quin rang de slots cau aquesta molècula per assignar-la a la partició corresponent.
    for i in "${!PARTITIONS[@]}"; do
        next_offset=$((offset + SLOTS[i]))

        if [ "$slot" -lt "$next_offset" ]; then
            echo "$mol" >> "$CLUSTER_OUT_DIR/molecules_pbc_${PARTITIONS[i]}.txt"
            break
        fi

        offset=$next_offset
    done

    line_number=$((line_number + 1))
done < "$MOLECULE_LIST"

echo "Molecules: $num_molecules"
echo "Molecule list: $MOLECULE_LIST"
echo "Submitting one dispatcher job per partition/node."

# Enviem un únic job dispatcher  per partició. Cada dispatcher llegeix la seva  llista de molècules i llança fins a "$slots" processos en paral·lel.
#job dispatcher= una feina enviada al clúster que s’encarrega de gestionar i llançar els càlculs de moltes molècules dins d’una partició.
for i in "${!PARTITIONS[@]}"; do
    partition="${PARTITIONS[i]}"
    slots="${SLOTS[i]}"
    partition_list="$CLUSTER_OUT_DIR/molecules_pbc_${partition}.txt"
    partition_count="$(wc -l < "$partition_list")"

    if [ "$partition_count" -eq 0 ]; then
        echo "Skipping $partition: no molecules assigned."
        continue
    fi

    echo "Submitting $partition: $partition_count molecules, $slots parallel processes inside one job"

    # SLURM executarà run_pbc_partition.sh dins del directori d'aquest script.
    # Els logs queden separats per job id i partició.
    sbatch \
        --partition="$partition" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="$slots" \
        --job-name="PBC_${partition}" \
        --chdir="$SCRIPT_DIR" \
        --output="$CLUSTER_OUT_DIR/sortid_%j_${partition}.txt" \
        --error="$CLUSTER_OUT_DIR/error_%j_${partition}.txt" \
        --wrap="bash \"$SCRIPT_DIR/run_pbc_partition.sh\" \"$partition_list\" \"$slots\" \"$partition\""
done

echo "Submitted PBC dispatcher jobs for all partitions."
