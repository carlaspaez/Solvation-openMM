import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

ROOT = Path(__file__).resolve().parents[2]
DATABASE_FINAL = ROOT / "resultats" / "database_final.csv"
OUT_PNG = Path(__file__).with_suffix(".png")
OUT_ELEC_PNG = Path(__file__).with_name("hist_delta_delta_electrostatica_pbc_sbc.png")
OUT_LJ_PNG = Path(__file__).with_name("hist_delta_delta_lj_pbc_sbc.png")
OUT_CSV = Path(__file__).with_name("delta_delta_components_pbc_sbc.csv")


def find_value(data, keys):
    for key in keys:
        if key not in data:
            continue
        try:
            value = data[key]
            if value is None:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def load_database_ids():
    with DATABASE_FINAL.open(newline="", encoding="utf-8") as f:
        return [row["identificador"] for row in csv.DictReader(f)]


def load_results(path):
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def component_values(data):
    elec_aq = find_value(data, ["DG_elec_aq_kcal_mol", "dG_elec_aq", "DG_elec_aq"])
    elec_gas = find_value(data, ["DG_elec_gas_kcal_mol", "dG_elec_gas", "DG_elec_gas"])
    lj_aq = find_value(data, ["DG_lj_aq_kcal_mol", "dG_LJ_aq", "DG_LJ_aq", "dG_vdW_aq"])
    lj_gas = find_value(data, ["DG_lj_gas_kcal_mol", "dG_LJ_gas", "DG_LJ_gas", "dG_vdW_gas"])

    if None in (elec_aq, elec_gas, lj_aq, lj_gas):
        return None

    return {
        "dg_elec": elec_aq - elec_gas,
        "dg_lj": lj_aq - lj_gas,
    }


def collect_rows():
    rows = []

    for mol_id in load_database_ids():
        pbc_path = ROOT / "outputs_PBC2" / mol_id / "TIP3P" / "results.json"
        sbc_path = ROOT / "outputs_SBC2" / mol_id / "results.json"

        pbc_data = load_results(pbc_path)
        sbc_data = load_results(sbc_path)
        if pbc_data is None or sbc_data is None:
            continue

        pbc = component_values(pbc_data)
        sbc = component_values(sbc_data)
        if pbc is None or sbc is None:
            continue

        rows.append(
            {
                "molecula": mol_id,
                "dg_elec_pbc": pbc["dg_elec"],
                "dg_elec_sbc": sbc["dg_elec"],
                "dg_lj_pbc": pbc["dg_lj"],
                "dg_lj_sbc": sbc["dg_lj"],
                "delta_delta_elec_sbc_menys_pbc": sbc["dg_elec"] - pbc["dg_elec"],
                "delta_delta_lj_sbc_menys_pbc": sbc["dg_lj"] - pbc["dg_lj"],
            }
        )

    return rows


def save_csv(rows):
    fieldnames = [
        "molecula",
        "dg_elec_pbc",
        "dg_elec_sbc",
        "dg_lj_pbc",
        "dg_lj_sbc",
        "delta_delta_elec_sbc_menys_pbc",
        "delta_delta_lj_sbc_menys_pbc",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = np.array(values, dtype=float)
    return {
        "n": len(values),
        "mitjana": values.mean(),
        "mediana": np.median(values),
        "desviacio": values.std(ddof=0),
        "sem": values.std(ddof=1) / np.sqrt(len(values)),
        "minim": values.min(),
        "maxim": values.max(),
    }


def stats_box(label, values):
    s = stats(values)
    return (
        f"{label}\n"
        f"n = {s['n']}\n"
        f"mitjana = {s['mitjana']:.2f}\n"
        f"mediana = {s['mediana']:.2f}\n"
        f"desv. est. = {s['desviacio']:.2f}\n"
        f"SEM = {s['sem']:.2f}"
    )


def plot_one_histogram(axis, values, title, color, xlabel):
    axis.hist(values, bins="auto", color=color, edgecolor="black", linewidth=0.7, alpha=0.85)
    axis.axvline(0, color="black", linestyle="--", linewidth=1.1, label="0")
    axis.axvline(values.mean(), color="#C44E52", linewidth=1.3, label="mitjana")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Nombre de molècules")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    axis.text(
        0.97,
        0.83,
        stats_box(title, values),
        transform=axis.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.82},
    )


def plot_histograms(rows):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("Falta matplotlib. Instal·la'l amb: pip install matplotlib") from exc

    delta_elec = np.array([row["delta_delta_elec_sbc_menys_pbc"] for row in rows])
    delta_lj = np.array([row["delta_delta_lj_sbc_menys_pbc"] for row in rows])

    configs = [
        (
            delta_elec,
            "ΔΔG electroestàtica",
            "#4C72B0",
            "ΔG electroestàtica SBC - ΔG electroestàtica PBC [kcal/mol]",
            OUT_ELEC_PNG,
        ),
        (
            delta_lj,
            "ΔΔG LJ/vdW",
            "#DD8452",
            "ΔG LJ/vdW SBC - ΔG LJ/vdW PBC [kcal/mol]",
            OUT_LJ_PNG,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, (values, title, color, xlabel, _out_path) in zip(axes, configs):
        plot_one_histogram(axis, values, title, color, xlabel)

    fig.suptitle("Canvi dels components finals de ΔG en passar de PBC a SBC")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    for values, title, color, xlabel, out_path in configs:
        fig, axis = plt.subplots(figsize=(7, 5))
        plot_one_histogram(axis, values, title, color, xlabel)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def main():
    rows = collect_rows()
    if not rows:
        raise SystemExit("No he trobat dades finals PBC/SBC per comparar.")

    save_csv(rows)
    plot_histograms(rows)

    delta_elec = [row["delta_delta_elec_sbc_menys_pbc"] for row in rows]
    delta_lj = [row["delta_delta_lj_sbc_menys_pbc"] for row in rows]

    print("Gràfica guardada a:", OUT_PNG)
    print("Gràfica electroestàtica guardada a:", OUT_ELEC_PNG)
    print("Gràfica LJ/vdW guardada a:", OUT_LJ_PNG)
    print("CSV guardat a:", OUT_CSV)
    print("Molècules comparades:", len(rows))
    for label, values in [
        ("ΔΔG electroestàtica", delta_elec),
        ("ΔΔG LJ/vdW", delta_lj),
    ]:
        s = stats(values)
        print(
            f"{label}: n={s['n']}, mitjana={s['mitjana']:.4f}, "
            f"mediana={s['mediana']:.4f}, desv. est.={s['desviacio']:.4f}, "
            f"SEM={s['sem']:.4f}, mínim={s['minim']:.4f}, "
            f"màxim={s['maxim']:.4f}"
        )


if __name__ == "__main__":
    main()
