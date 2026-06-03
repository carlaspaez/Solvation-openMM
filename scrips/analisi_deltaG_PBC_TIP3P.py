import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

csv_file = ROOT / "resultats/database_carla.csv"
output_dir = ROOT / "outputs_PBC2"

col1 = "metode 1 (kcal/mol)"
col2 = "incertesa metode 1 (kcal/mol)"

with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    cols = reader.fieldnames

if col1 not in cols:
    cols.append(col1)

if col2 not in cols:
    cols.append(col2)


def busca_results_json(mol):
    json_file = output_dir / mol / "TIP3P" / "results.json"
    if json_file.exists():
        return json_file
    return None


actualitzades = 0
pendents = 0

for row in rows:

    mol = row["identificador"]

    json_file = busca_results_json(mol)
    if json_file is None:
        pendents += 1
        row[col1] = "NA"
        row[col2] = "NA"
        print("Encara no trobo results.json:", mol)
        continue

    with open(json_file, "r") as f:
        data = json.load(f)

    row[col1] = data["DG_hyd_kcal_mol"]
    row[col2] = data["err_hyd_kcal_mol"]
    actualitzades += 1

    print("Actualitzat:", mol, "des de", json_file)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print("CSV actualitzat")
print("Resultats actualitzats:", actualitzades)
print("Molecules encara pendents:", pendents)
