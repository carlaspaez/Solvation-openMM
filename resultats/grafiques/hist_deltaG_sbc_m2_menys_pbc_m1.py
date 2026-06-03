#!/usr/bin/env python3
import csv
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


RESULTATS_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = RESULTATS_DIR / "database_final.csv"
OUT_PATH = Path(__file__).resolve().parent / "hist_deltaG_sbc_m2_menys_pbc_m1.png"


def llegir_diferencies():
    x_col = "metode 1 (kcal/mol)"
    y_col = "metode 2 (kcal/mol)"

    diferencies = []
    with CSV_PATH.open(newline="", encoding="utf-8") as fitxer:
        reader = csv.DictReader(fitxer)
        fields = reader.fieldnames or []
        missing = [col for col in (x_col, y_col) if col not in fields]
        if missing:
            raise ValueError("Falten columnes al CSV: " + ", ".join(missing))

        for row in reader:
            try:
                dg_pbc_m1 = float(row[x_col])
                dg_sbc_m2 = float(row[y_col])
            except (TypeError, ValueError):
                continue

            diferencies.append(dg_sbc_m2 - dg_pbc_m1)

    return np.array(diferencies)


def main():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("Falta matplotlib. Instal·la'l amb: pip install matplotlib") from exc

    diferencies = llegir_diferencies()
    if diferencies.size == 0:
        raise ValueError("No hi ha dades numèriques vàlides per fer l'histograma.")

    mitjana = diferencies.mean()
    mediana = np.median(diferencies)
    desviacio = diferencies.std(ddof=0)
    sem = diferencies.std(ddof=1) / np.sqrt(diferencies.size)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        diferencies,
        bins="auto",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=1.1, linestyle="--")
    ax.axvline(mitjana, color="#C44E52", linewidth=1.3, label="mitjana")

    ax.set_xlabel("ΔG SBC (mètode 2) - ΔG PBC (mètode 1) [kcal/mol]")
    ax.set_ylabel("Nombre de molècules")
    ax.set_title("Distribució de les diferències entre ΔG SBC i ΔG PBC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    ax.text(
        0.98,
        0.97,
        (
            f"n = {diferencies.size}\n"
            f"mitjana = {mitjana:.3f}\n"
            f"mediana = {mediana:.3f}\n"
            f"desv. est. = {desviacio:.3f}\n"
            f"SEM = {sem:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.8},
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)
    print(f"Histograma guardat a: {OUT_PATH}")
    print(f"Nombre de molècules: {diferencies.size}")
    print(f"Mitjana: {mitjana:.6f} kcal/mol")
    print(f"Mediana: {mediana:.6f} kcal/mol")
    print(f"Desviació estàndard: {desviacio:.6f} kcal/mol")
    print(f"SEM: {sem:.6f} kcal/mol")


if __name__ == "__main__":
    main()
