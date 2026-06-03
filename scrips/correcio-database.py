import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

database_inicial = ROOT / "resultats" / "database_carla.csv"
database_final = ROOT / "resultats" / "database_final.csv"

columnes_resultats = [
    "metode 1 (kcal/mol)",
    "incertesa metode 1 (kcal/mol)",
    "metode 2 (kcal/mol)",
    "incertesa metode 2 (kcal/mol)",
]


def te_resultats_complets(row):
    for columna in columnes_resultats:
        valor = row.get(columna, "").strip()
        if valor in {"", "NA"}:
            return False
    return True


with open(database_inicial, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    cols = reader.fieldnames

rows_final = [row for row in rows if te_resultats_complets(row)]

with open(database_final, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows_final)

print("Database final creada:", database_final)
print("Molecules inicials:", len(rows))
print("Molecules amb metode 1 i metode 2:", len(rows_final))
print("Molecules descartades:", len(rows) - len(rows_final))
