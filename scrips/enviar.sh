#!/bin/bash
#SBATCH --job-name=calcul
#SBATCH --output=../outputs-OK/cluster/sortid_%j.txt
#SBATCH --error=../outputs-OK/cluster/error_%j.txt
#SBATCH --time=2:00:00
##SBATCH --cpus-per-task=1
##SBATCH --mem=4GB

set -euo pipefail
mkdir -p ../outputs-OK/cluster

if [ -z "${1:-}" ]; then
        echo "Usage: $0 <molecule>"
        exit 1
fi

python deltaG_PBC_TIP3P-ok.py -m "$1" -s SOL.TIP3P.itp
