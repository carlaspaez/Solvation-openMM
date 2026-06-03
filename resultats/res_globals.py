#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTATS_DIR = ROOT / "resultats"
DATABASE_PATH = RESULTATS_DIR / "database_carla.csv"
OUT_CSV_PATH = RESULTATS_DIR / "res_globals_taula.csv"

PBC_ERRORS_PATH = ROOT / "outputs_PBC2" / "metadata_error_PBC.json"
SBC_ERRORS_PATH = ROOT / "outputs_SBC2" / "metadata_error_SBC.json"

BIB_COL = "resultats a partir de bibliografia (kcal/mol)"
METHOD_COLUMNS = {
    "1=PBC": "metode 1 (kcal/mol)",
    "2=SBC": "metode 2 (kcal/mol)",
}


def carregar_database():
    with DATABASE_PATH.open(newline="", encoding="utf-8") as fitxer:
        return list(csv.DictReader(fitxer))


def carregar_fallides(path):
    with path.open(encoding="utf-8") as fitxer:
        errors = json.load(fitxer)

    return sum(not error.get("te_results_json", False) for error in errors)


def valor_float(row, col):
    try:
        valor = float(row[col])
    except (KeyError, TypeError, ValueError):
        return None

    if math.isnan(valor):
        return None
    return valor


def calcular_r2_com_graf_taula(rows, method_col):
    x_vals = []
    y_vals = []

    for row in rows:
        x = valor_float(row, BIB_COL)
        y = valor_float(row, method_col)
        if x is None or y is None:
            continue

        x_vals.append(x)
        y_vals.append(y)

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    r = np.corrcoef(x_arr, y_arr)[0, 1]
    return r**2


def calcular_metriques(rows, method_col):
    refs = []
    preds = []

    for row in rows:
        ref = valor_float(row, BIB_COL)
        pred = valor_float(row, method_col)
        if ref is None or pred is None:
            continue

        refs.append(ref)
        preds.append(pred)

    ref_arr = np.array(refs)
    pred_arr = np.array(preds)
    residuals = pred_arr - ref_arr

    return {
        "n": len(pred_arr),
        "R2 vs bibliografic": calcular_r2_com_graf_taula(rows, method_col),
        "RMSE": np.sqrt(np.mean(residuals**2)),
        "MAE": np.mean(np.abs(residuals)),
        "biaix": np.mean(residuals),
    }


def format_float(valor):
    return f"{valor:.4f}"


def format_row(row):
    return [
        row["Mètode"],
        str(row["n"]),
        format_float(row["R2 vs bibliografic"]),
        format_float(row["RMSE"]),
        format_float(row["MAE"]),
        format_float(row["biaix"]),
        str(row["simulacions fallides"]),
    ]


def imprimir_taula(rows):
    headers = [
        "Mètode",
        "n",
        "R2 vs bibliogràfic",
        "RMSE",
        "MAE",
        "biaix",
        "simulacions fallides",
    ]
    formatted_rows = [format_row(row) for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in formatted_rows))
        for i in range(len(headers))
    ]

    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(" | ".join("-" * width for width in widths))
    for row in formatted_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def guardar_csv(rows):
    headers = [
        "Mètode",
        "n",
        "R2 vs bibliografic",
        "RMSE",
        "MAE",
        "biaix",
        "simulacions fallides",
    ]

    with OUT_CSV_PATH.open("w", newline="", encoding="utf-8") as fitxer:
        writer = csv.DictWriter(fitxer, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    rows = carregar_database()
    fallides = {
        "1=PBC": carregar_fallides(PBC_ERRORS_PATH),
        "2=SBC": carregar_fallides(SBC_ERRORS_PATH),
    }

    resultats = []
    for metode, method_col in METHOD_COLUMNS.items():
        metriques = calcular_metriques(rows, method_col)
        resultats.append({
            "Mètode": metode,
            **metriques,
            "simulacions fallides": fallides[metode],
        })

    imprimir_taula(resultats)
    guardar_csv(resultats)
    print(f"\nTaula guardada a: {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()
