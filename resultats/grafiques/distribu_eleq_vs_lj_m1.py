
import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib.pyplot as plt
import numpy as np


def find_value(d, keys):
    for k in keys:
        if k in d:
            try:
                v = d[k]
                if v is None:
                    continue
                return float(v)
            except Exception:
                continue
    return None


def main():
    base = Path(__file__).resolve().parents[2] / 'outputs_PBC-ok'
    if not base.exists():
        base = Path('outputs_PBC-ok')

    elec_diffs = []
    lj_diffs = []
    n_json = 0

    # key variants to be tolerant
    elec_aq_keys = ['DG_elec_aq_kcal_mol', 'ΔG_elec,aq', 'dG_elec_aq', 'dG_elec,aq', 'DG_elec_aq']
    elec_gas_keys = ['DG_elec_gas_kcal_mol', 'ΔG_elec,gas', 'dG_elec_gas', 'dG_elec,gas', 'DG_elec_gas']
    lj_aq_keys = ['DG_lj_aq_kcal_mol', 'ΔG_LJ,aq', 'dG_LJ_aq', 'dG_LJ,aq', 'DG_LJ_aq', 'ΔG_vdW,aq', 'dG_vdW_aq']
    lj_gas_keys = ['DG_lj_gas_kcal_mol', 'ΔG_LJ,gas', 'dG_LJ_gas', 'dG_LJ,gas', 'DG_LJ_gas', 'ΔG_vdW,gas', 'dG_vdW_gas']

    for moldir in sorted(base.iterdir() if base.exists() else []):
        if not moldir.is_dir():
            continue
        rj = moldir / 'TIP3P' / 'results.json'
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text(encoding='utf-8'))
        except Exception:
            continue
        n_json += 1

        elec_aq = find_value(data, elec_aq_keys)
        elec_gas = find_value(data, elec_gas_keys)
        lj_aq = find_value(data, lj_aq_keys)
        lj_gas = find_value(data, lj_gas_keys)

        if elec_aq is not None and elec_gas is not None:
            elec_diffs.append(elec_aq - elec_gas)
        if lj_aq is not None and lj_gas is not None:
            lj_diffs.append(lj_aq - lj_gas)

    # require some data
    if len(elec_diffs) == 0 and len(lj_diffs) == 0:
        print('No he trobat dades a outputs_PBC-ok/*/TIP3P/results.json')
        return

    # Make arrays and remove NaNs
    elec = np.array([x for x in elec_diffs if not math.isnan(x)])
    lj = np.array([x for x in lj_diffs if not math.isnan(x)])

    # Plot violin plot for comparison
    plt.figure(figsize=(7, 5))

    colors = ['#4C72B0', '#DD8452']  # blau = electrostàtica, taronja = LJ

    datasets = []
    labels = []
    used_colors = []
    if len(elec):
        datasets.append(elec)
        labels.append('electrostàtica')
        used_colors.append(colors[0])
    if len(lj):
        datasets.append(lj)
        labels.append('LJ / vdW')
        used_colors.append(colors[1])

    if len(datasets) == 0:
        print('No hi ha dades numèriques per representar')
        return

    positions = list(range(1, len(datasets) + 1))
    parts = plt.violinplot(datasets, positions=positions, showmeans=True, showextrema=True, widths=0.7)
    for pc, color in zip(parts['bodies'], used_colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    plt.xticks(positions, labels)
    plt.xlabel("Tipus d'interacció")
    plt.ylabel('Contribució a ΔG_hyd (kcal/mol)')
    plt.title('Distribució de les contribucions a ΔG_hyd')
    plt.tight_layout()

    out = Path(__file__).with_suffix('.png')
    plt.savefig(out, dpi=150)
    print(f'Gràfica guardada a: {out}')
    print(f'Fitxers JSON llegits: {n_json}')
    print(f'Punts electrostàtica: {len(elec)}')
    print(f'Punts LJ / vdW: {len(lj)}')


if __name__ == '__main__':
    main()
