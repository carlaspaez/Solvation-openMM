import json
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / 'outputs_PBC-ok'

MOLES = ['mobley_2518989', 'mobley_8705848', 'mobley_2923700']


def get_float(data, key):
    value = float(data[key])
    if math.isnan(value):
        raise ValueError(f'Valor NaN a {key}')
    return value


def read_components(mol_id):
    path = OUTPUTS / mol_id / 'TIP3P' / 'results.json'
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding='utf-8') as f:
        data = json.load(f)

    elec = get_float(data, 'DG_elec_aq_kcal_mol') - get_float(data, 'DG_elec_gas_kcal_mol')
    lj = get_float(data, 'DG_lj_aq_kcal_mol') - get_float(data, 'DG_lj_gas_kcal_mol')
    total = get_float(data, 'DG_hyd_kcal_mol')

    return elec, lj, total


def plot():
    labels = MOLES
    elecs, ljs = [], []
    for mol_id in MOLES:
        elec, lj, total = read_components(mol_id)
        elecs.append(elec)
        ljs.append(lj)

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ['#4C72B0', '#DD8452']  # blau = electrostàtica, taronja = LJ
    ax.bar([i - width/2 for i in x], elecs, width, label='Electrostàtica', color=colors[0])
    ax.bar([i + width/2 for i in x], ljs, width, label='LJ / vdW', color=colors[1])

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Contribució a ΔG_hyd (kcal/mol)')
    ax.set_title('Contribució a ΔG_hyd per a tres molècules representatives')
    ax.legend()
    fig.tight_layout()

    out_png = Path(__file__).with_suffix('.png')
    fig.savefig(out_png, dpi=200)
    print(f'Gràfica guardada a: {out_png}')


if __name__ == '__main__':
    plot()
