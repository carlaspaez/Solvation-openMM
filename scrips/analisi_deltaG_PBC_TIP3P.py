import csv
import json
from pathlib import Path

csv_file = Path("resultats/database_carla.csv")
output_dir = Path("outputs")

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

for row in rows:

    mol = row["identificador"]

 ######   json_file = output_dir / mol / "cvbn.json"

    if not json_file.exists():
        print("No trobo:", mol)
        continue

    with open(json_file, "r") as f:
        data = json.load(f)

    row[col1] = data["DG_hyd_kcal_mol"]
    row[col2] = data["err_hyd_kcal_mol"]

    print("OK:", mol)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print("CSV actualitzat")