#!/usr/bin/env python3
# Llibreria estàndard per llegir CSV amb capçalera.
import csv
# Eines de camins de fitxer independents del sistema operatiu.
from pathlib import Path

# NumPy per càlcul numèric (arrays i correlació).
import numpy as np


# Carpeta de resultats del projecte.
RESULTATS_DIR = Path(__file__).resolve().parents[1]
# Fitxer CSV d'entrada amb valors i incerteses.
CSV_PATH = RESULTATS_DIR / 'database_final.csv'
# Fitxer PNG de sortida on guardem la figura.
OUT_PATH = Path(__file__).resolve().parent / 'graf_taula_m1vsm2.png'


# Punt d'entrada principal de l'script.
def main():
    # Import local de matplotlib per mostrar un error net si falta el paquet.
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        # Sortida controlada amb instrucció clara d'instal·lació.
        raise SystemExit(
            "Falta matplotlib. Instal·la'l amb: pip install matplotlib"
        ) from exc

    # Obrim el CSV en mode lectura.
    with CSV_PATH.open(newline='', encoding='utf-8') as f:
        # DictReader: cada fila serà un diccionari {nom_columna: valor}.
        reader = csv.DictReader(f)
        # Llista de noms de columna del fitxer.
        fields = reader.fieldnames or []

        # Definició explícita de columnes per al gràfic:
        # Eix X: metode 1.
        x_col = 'metode 1 (kcal/mol)'
        # Eix Y: mètode 2.
        y_col = 'metode 2 (kcal/mol)'
        # Error en X: incertesa del mètode 1.
        x_unc_col = 'incertesa metode 1 (kcal/mol)'
        # Error en Y: incertesa del mètode 2.
        y_unc_col = 'incertesa metode 2 (kcal/mol)'
        # Identificador de la molècula, útil per etiquetar outliers.
        id_col = 'identificador'

        # Validem que existeixin totes les columnes imprescindibles.
        required = [x_col, y_col, x_unc_col, y_unc_col, id_col]
        missing = [col for col in required if col not in fields]
        if missing:
            # Si falta alguna columna, parem amb un missatge informatiu.
            raise ValueError(
                "Falten columnes requerides al CSV: " + ", ".join(missing)
            )

        # Llistes temporals per acumular dades numèriques vàlides.
        x_vals = []
        y_vals = []
        x_unc_vals = []
        y_unc_vals = []
        ids = []
        # Recorrem totes les files del CSV.
        for row in reader:
            try:
                # Convertim cada camp a float.
                x = float(row[x_col])
                y = float(row[y_col])
                x_unc = float(row[x_unc_col])
                y_unc = float(row[y_unc_col])
            except (ValueError, TypeError, KeyError):
                # Si la fila té valors buits/no numèrics, la descartem.
                continue
            # Guardem la fila vàlida.
            x_vals.append(x)
            y_vals.append(y)
            x_unc_vals.append(x_unc)
            y_unc_vals.append(y_unc)
            ids.append(row[id_col])

    # Protecció: no intentem pintar si no hi ha cap punt vàlid.
    if not x_vals:
        raise ValueError('No hi ha dades numèriques vàlides per representar.')

    # Convertim a arrays NumPy per facilitar càlculs vectorials.
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    x_unc_arr = np.array(x_unc_vals)
    y_unc_arr = np.array(y_unc_vals)
    ids_arr = np.array(ids)

    # Coeficient de correlació de Pearson entre X i Y.
    r = np.corrcoef(x_arr, y_arr)[0, 1]
    r2 = r ** 2
    n = len(x_arr)

    # Errors respecte a la diagonal y=x: positiu vol dir que mètode 2 és més alt.
    residuals = y_arr - x_arr
    abs_errors = np.abs(residuals)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(abs_errors)
    bias = np.mean(residuals)

    # Regressió lineal simple y = pendent*x + intercept.
    slope, intercept = np.polyfit(x_arr, y_arr, 1)

    # Outliers estadístics: errors absoluts clarament grans respecte al conjunt.
    outlier_threshold = abs_errors.mean() + 2 * abs_errors.std(ddof=0)
    outlier_mask = abs_errors > outlier_threshold
    # Creem la figura de mida 8x6 polzades.
    plt.figure(figsize=(8, 6))
    # Dibuix principal: punts + barres d'error horitzontals i verticals.
    plt.errorbar(
        # Coordenades X.
        x_arr,
        # Coordenades Y.
        y_arr,
        # Incertesa de cada punt en X.
        xerr=x_unc_arr,
        # Incertesa de cada punt en Y.
        yerr=y_unc_arr,
        # Format de marcador circular.
        fmt='o',
        # Mida del marcador.
        markersize=4,
        # Color dels punts.
        color='blue',
        # Color de les barres d'error.
        ecolor='blue',
        # Gruix de les barres d'error.
        elinewidth=0.7,
        # Mida dels "caps" de les barres.
        capsize=2,
        # Transparència general.
        alpha=0.85,
        # Vora negra del punt per millorar visibilitat.
        markeredgecolor='black',
        # Gruix de la vora del marcador.
        markeredgewidth=0.4,
    )

    # Límits comuns per dibuixar la diagonal y = x.
    lim_min = min(x_arr.min(), y_arr.min())
    lim_max = max(x_arr.max(), y_arr.max())
    # Línia de referència d'acord perfecte.
    plt.plot([lim_min, lim_max], [lim_min, lim_max], '-', linewidth=1.2, color='black', label='y = x')

    # Recta de regressió ajustada.
    x_line = np.array([lim_min, lim_max])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, '--', linewidth=1.2, color='red', label='regressió')

    # Etiquetes i títol.
    plt.xlabel('resultats a partir de mètode 1 (kcal/mol)')
    plt.ylabel('resultats a partir de mètode 2 (kcal/mol)')
    plt.title('Comparació de ΔG lliure d’hidratació: mètode 1 vs mètode 2')
    # Graella suau per facilitar lectura.
    plt.grid(alpha=0.25)
    # Llegenda amb la línia y=x.
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
    # Ajust automàtic d'espais perquè no es talli text.
    plt.tight_layout()
    # Guardem la figura a PNG.
    plt.savefig(OUT_PATH, dpi=300)

    # Resum per terminal.
    print(f'Gràfica guardada a: {OUT_PATH}')
    print(f'Columna X usada: {x_col} (+ {x_unc_col})')
    print(f'Columna Y usada: {y_col} (+ {y_unc_col})')
    print(f'Punts representats: {len(x_arr)}')
    print(f'Correlació de Pearson (r): {r:.4f}')
    print(f'Coeficient de determinació (R^2): {r2:.4f}')
    print(f'n: {n}')
    print(f'RMSE: {rmse:.4f} kcal/mol')
    print(f'MAE: {mae:.4f} kcal/mol')
    print(f'Biaix mitjà (mètode 2 - mètode 1): {bias:.4f} kcal/mol')
    print(f'Regressió y = pendent*x + intercepció: pendent={slope:.4f}, intercepció={intercept:.4f}')
    print(f'Llindar outlier |error|: {outlier_threshold:.4f} kcal/mol')
    print(f'Outliers estadístics: {int(outlier_mask.sum())}')
    print(f'Incerteses X representades: {len(x_unc_arr)}')
    print(f'Incerteses Y representades: {len(y_unc_arr)}')


# Execució directa de l'script des de terminal.
if __name__ == '__main__':
    main()
