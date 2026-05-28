#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np


RESULTATS_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = RESULTATS_DIR / 'database_carla.csv'
OUT_PATH = Path(__file__).resolve().parent / 'zoomM1.png'
ZOOM_MIN = -30.0
ZOOM_MAX = 30.0


def main():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Falta matplotlib. Instal.la'l amb: pip install matplotlib"
        ) from exc

    x_col = 'resultats a partir de bibliografia (kcal/mol)'
    y_col = 'metode 1 (kcal/mol)'
    x_unc_col = 'incertesa calculada (kcal/mol)'
    y_unc_col = 'incertesa metode 1 (kcal/mol)'
    id_col = 'identificador'

    x_vals = []
    y_vals = []
    x_unc_vals = []
    y_unc_vals = []
    ids = []

    with CSV_PATH.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = [x_col, y_col, x_unc_col, y_unc_col, id_col]
        missing = [col for col in required if col not in fields]
        if missing:
            raise ValueError(
                "Falten columnes requerides al CSV: " + ", ".join(missing)
            )

        for row in reader:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                x_unc = float(row[x_unc_col])
                y_unc = float(row[y_unc_col])
            except (ValueError, TypeError, KeyError):
                continue

            if not (ZOOM_MIN <= x <= ZOOM_MAX and ZOOM_MIN <= y <= ZOOM_MAX):
                continue

            x_vals.append(x)
            y_vals.append(y)
            x_unc_vals.append(x_unc)
            y_unc_vals.append(y_unc)
            ids.append(row[id_col])

    if not x_vals:
        raise ValueError('No hi ha dades numèriques dins del rang del zoom.')

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    x_unc_arr = np.array(x_unc_vals)
    y_unc_arr = np.array(y_unc_vals)

    r = np.corrcoef(x_arr, y_arr)[0, 1]
    r2 = r ** 2
    n = len(x_arr)

    residuals = y_arr - x_arr
    abs_errors = np.abs(residuals)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(abs_errors)
    bias = np.mean(residuals)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        x_arr,
        y_arr,
        xerr=x_unc_arr,
        yerr=y_unc_arr,
        fmt='o',
        markersize=4,
        color='blue',
        ecolor='blue',
        elinewidth=0.7,
        capsize=2,
        alpha=0.85,
        markeredgecolor='black',
        markeredgewidth=0.4,
    )

    plt.plot(
        [ZOOM_MIN, ZOOM_MAX],
        [ZOOM_MIN, ZOOM_MAX],
        '-',
        linewidth=1.2,
        color='black',
        label='y = x',
    )

    x_line = np.array([ZOOM_MIN, ZOOM_MAX])
    plt.plot(
        x_line,
        slope * x_line + intercept,
        '--',
        linewidth=1.2,
        color='red',
        label='regressió',
    )

    plt.xlim(ZOOM_MIN, ZOOM_MAX)
    plt.ylim(ZOOM_MIN, ZOOM_MAX)
    plt.xlabel('resultats a partir de la bibliografia (kcal/mol)')
    plt.ylabel('resultats a partir de mètode 1 (kcal/mol)')
    plt.title('Zoom ΔG d’hidratació (-30 a 30 kcal/mol): bibliografia vs mètode 1')
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    plt.text(
        0.03,
        0.97,
        (
            f'n = {n}\n'
            f'r = {r:.4f}\n'
            f'$R^2$ = {r2:.4f}\n'
            f'RMSE = {rmse:.3f} kcal/mol\n'
            f'MAE = {mae:.3f} kcal/mol\n'
            f'biaix = {bias:.3f} kcal/mol\n'
            f'pendent = {slope:.3f}\n'
            f'intercepció = {intercept:.3f}'
        ),
        transform=plt.gca().transAxes,
        va='top',
        ha='left',
        fontsize=11,
        bbox={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.8},
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)

    print(f'Gràfica zoom guardada a: {OUT_PATH}')
    print(f'Rang del zoom: [{ZOOM_MIN}, {ZOOM_MAX}] kcal/mol en X i Y')
    print(f'Punts representats: {n}')
    print(f'Correlació de Pearson (r): {r:.4f}')
    print(f'Coeficient de determinació (R^2): {r2:.4f}')
    print(f'RMSE: {rmse:.4f} kcal/mol')
    print(f'MAE: {mae:.4f} kcal/mol')
    print(f'Biaix mitjà (mètode 1 - bibliografia): {bias:.4f} kcal/mol')
    print(f'Regressió y = pendent*x + intercepció: pendent={slope:.4f}, intercepció={intercept:.4f}')


if __name__ == '__main__':
    main()
