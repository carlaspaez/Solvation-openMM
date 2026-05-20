#!/bin/bash
#SBATCH --job-name=calcul
#SBATCH --output=sortid_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=2:00:00
##SBATCH --cpus-per-task=1
##SBATCH --mem=4GB

set -euo pipefail


if [ -z "${1:-}" ]; then
        echo "Usage: $0 <molecule>"
        exit 1
fi

python deltaG_SBC_TIP3P.py -m "$1" -s SOL.TIP3P.itp
