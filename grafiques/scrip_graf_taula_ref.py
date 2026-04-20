#!/usr/bin/env python3
# Llibreria estàndard per llegir CSV amb capçalera.
import csv
# Eines de camins de fitxer independents del sistema operatiu.
from pathlib import Path

# NumPy per càlcul numèric (arrays i correlació).
import numpy as np


# Carpeta arrel del projecte (pare de la carpeta `grafiques`).
ROOT = Path(__file__).resolve().parents[1]
# Fitxer CSV d'entrada amb valors i incerteses.
CSV_PATH = ROOT / 'TAULA REF.csv'
# Fitxer PNG de sortida on guardem la figura.
OUT_PATH = Path(__file__).resolve().parent / 'graf_taula_ref.png'


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
        # Eix X: bibliogràfic.
        x_col = 'resultats a partir de bibliografia (kcal/mol)'
        # Eix Y: experimental (admet diverses capçaleres).
        exp_candidates = [
            'resultats experimentals (kcal/mol)',
            'resultats experimentals (AMBER/GAFF) (kcal/mol)',
        ]
        y_col = next((c for c in exp_candidates if c in fields), exp_candidates[0])
        # Error en X: incertesa calculada/bibliogràfica.
        x_unc_col = 'uncertesa calculada (kcal/mol)'
        # Error en Y: incertesa experimental.
        y_unc_col = 'uncertesa experimental (kcal/mol)'

        # Validem que existeixin totes les columnes imprescindibles.
        required = [x_col, y_col, x_unc_col, y_unc_col]
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

    # Protecció: no intentem pintar si no hi ha cap punt vàlid.
    if not x_vals:
        raise ValueError('No hi ha dades numèriques vàlides per representar.')

    # Convertim a arrays NumPy per facilitar càlculs vectorials.
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    x_unc_arr = np.array(x_unc_vals)
    y_unc_arr = np.array(y_unc_vals)

    # Coeficient de correlació de Pearson entre X i Y.
    r = np.corrcoef(x_arr, y_arr)[0, 1]

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

    # Etiquetes i títol.
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Correlacio: bibliografics (X) vs experimentals (Y)')
    # Graella suau per facilitar lectura.
    plt.grid(alpha=0.25)
    # Llegenda amb la línia y=x.
    plt.legend(loc='best')
    # Ajust automàtic d'espais perquè no es talli text.
    plt.tight_layout()
    # Guardem la figura a PNG.
    plt.savefig(OUT_PATH, dpi=300)

    # Resum per terminal.
    print(f'Grafica guardada a: {OUT_PATH}')
    print(f'Columna X usada: {x_col} (+ {x_unc_col})')
    print(f'Columna Y usada: {y_col} (+ {y_unc_col})')
    print(f'Punts representats: {len(x_arr)}')
    print(f'Correlacio de Pearson (r): {r:.4f}')


# Execució directa de l'script des de terminal.
if __name__ == '__main__':
    main()
