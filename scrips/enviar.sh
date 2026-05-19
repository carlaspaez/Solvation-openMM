#!/bin/bash
#SBATCH --job-name=calcul
#SBATCH --output=sortid_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=2:00:00
##SBATCH --cpus-per-task=4
##SBATCH --mem=4GB
#SBATCH --partition=normal2                 # canvieu de node, tenim highmem, normal1,2,3,4,5 i gpu
set -euo pipefail

if [ -z "${1:-}" ]; then
        echo "Usage: $0 <molecule>"
        exit 1
fi

python deltaG_PBC_TIP3P-ok.py -m "$1" -s SOL.TIP3P.itp
