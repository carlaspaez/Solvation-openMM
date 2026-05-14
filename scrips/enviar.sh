#!/bin/bash
#SBATCH --job-name=calcul          # Nom del job
#SBATCH --output=sortid%j.txt        # Fitxer de sortida
#SBATCH --error=error%j.txt           # Fitxer d’errors
#SBATCH --time=10:00:00             # Temps màxim (hh:mm:ss)
##SBATCH --cpus-per-task=4           # Nombre de CPUs per tasca
##SBATCH --mem=4GB                   # Memòria assignada

echo "Inici del job: $(date)"
python deltaG.py -m mobley_5857 -s SOL.TIP3P.itp          # Execució del vostre codi
echo "Final del job: $(date)"
