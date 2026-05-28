import csv
import json
from pathlib import Path

csv_file = Path("resultats/database_carla.csv")
output_dir = Path("outputs_PBC-ok")

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


def falta_resultat(row):
    valor = row.get(col1, "").strip()
    incertesa = row.get(col2, "").strip()
    return valor in {"", "NA"} or incertesa in {"", "NA"}


def busca_results_json(mol):
    json_file = output_dir / mol / "TIP3P" / "results.json"
    if json_file.exists():
        return json_file
    return None


afegides = 0
ja_existents = 0
pendents = 0

for row in rows:

    mol = row["identificador"]

    if not falta_resultat(row):
        ja_existents += 1
        print("Ja tenia resultat, salto:", mol)
        continue

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
    afegides += 1

    print("Afegit:", mol, "des de", json_file)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print("CSV actualitzat")
print("Resultats nous afegits:", afegides)
print("Molecules que ja tenien resultat:", ja_existents)
print("Molecules encara pendents:", pendents)
