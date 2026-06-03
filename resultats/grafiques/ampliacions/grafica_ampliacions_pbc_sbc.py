import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

ROOT = Path(__file__).resolve().parents[3]
DATABASE_PATH = ROOT / "resultats" / "database_carla.csv"
OUT_DIR = ROOT / "resultats" / "grafiques" / "ampliacions"
OUT_PNG = OUT_DIR / "barres_error_ampliacions_pbc_sbc.png"
OUT_CSV = OUT_DIR / "dades_error_ampliacions_pbc_sbc.csv"

REF_COL = "resultats a partir de bibliografia (kcal/mol)"
REF_ERR_COL = "incertesa calculada (kcal/mol)"
AMPLIACIONS = [10, 30, 50, 150]
EXCLUDED_MOLECULES = {"mobley_9534740"}

METHODS = {
    "PBC": {
        "results_path": lambda ampli, mol: (
            ROOT / f"outputs_PBC_ampli{ampli}" / mol / "TIP3P" / "results.json"
        ),
    },
    "SBC": {
        "results_path": lambda ampli, mol: (
            ROOT / f"outputs_SBC_ampli{ampli}" / mol / "results.json"
        ),
    },
}

AMPLI_COLORS = {
    10: "#1f77b4",
    30: "#2ca02c",
    50: "#ff7f0e",
    150: "#d62728",
}


def float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_database():
    with DATABASE_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_result(path):
    if not path.exists():
        return None

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    dg = float_or_none(data.get("DG_hyd_kcal_mol"))
    err = float_or_none(data.get("err_hyd_kcal_mol"))
    if dg is None or err is None:
        return None

    return dg, err


def collect_points(rows):
    points = []

    for row in rows:
        mol = row["identificador"]
        if mol in EXCLUDED_MOLECULES:
            continue

        ref = float_or_none(row.get(REF_COL))
        ref_err = float_or_none(row.get(REF_ERR_COL))
        if ref is None or ref_err is None:
            continue

        for ampli in AMPLIACIONS:
            for method, config in METHODS.items():
                result_path = config["results_path"](ampli, mol)
                result = load_result(result_path)
                if result is None:
                    continue

                dg, err = result
                points.append(
                    {
                        "molecula": mol,
                        "nom iupac molec": row["nom iupac molec"],
                        "metode": method,
                        "ampliacio_percent": ampli,
                        "bibliografia_kcal_mol": ref,
                        "incertesa_bibliografia_kcal_mol": ref_err,
                        "deltaG_simulacio_kcal_mol": dg,
                        "incertesa_simulacio_kcal_mol": err,
                        "error_vs_bibliografia_kcal_mol": dg - ref,
                    }
                )

    return points


def short_label(name):
    if len(name) <= 46:
        return name
    return name[:43] + "..."


def save_points_csv(points):
    fieldnames = [
        "molecula",
        "nom iupac molec",
        "metode",
        "ampliacio_percent",
        "bibliografia_kcal_mol",
        "incertesa_bibliografia_kcal_mol",
        "deltaG_simulacio_kcal_mol",
        "incertesa_simulacio_kcal_mol",
        "error_vs_bibliografia_kcal_mol",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def grouped_by_molecule(points):
    grouped = {}
    for point in points:
        grouped.setdefault(point["molecula"], []).append(point)
    return dict(sorted(grouped.items()))


def plot_error_bars(points):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise SystemExit("Falta matplotlib. Instal.la'l amb: pip install matplotlib") from exc

    grouped = grouped_by_molecule(points)
    if not grouped:
        raise SystemExit("No he trobat cap results.json d'ampliacions per dibuixar.")

    n_molecules = len(grouped)
    fig, axes = plt.subplots(
        n_molecules,
        1,
        figsize=(8.5, 3.2 * n_molecules),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    all_errors = [point["error_vs_bibliografia_kcal_mol"] for point in points]
    max_abs_error = max(abs(value) for value in all_errors)
    y_margin = max(1.0, 0.08 * max_abs_error)
    y_lim = max_abs_error + y_margin

    for axis, (mol, mol_points) in zip(axes, grouped.items()):
        name = mol_points[0]["nom iupac molec"]
        ref = mol_points[0]["bibliografia_kcal_mol"]
        ref_err = mol_points[0]["incertesa_bibliografia_kcal_mol"]

        axis.axhline(0, color="black", linewidth=1.0, linestyle="-")
        axis.fill_between(
            [-0.55, 1.55],
            [-ref_err, -ref_err],
            [ref_err, ref_err],
            color="black",
            alpha=0.08,
            linewidth=0,
        )

        group_centers = {"PBC": 0, "SBC": 1}
        bar_width = 0.16
        offsets = {
            10: -1.5 * bar_width,
            30: -0.5 * bar_width,
            50: 0.5 * bar_width,
            150: 1.5 * bar_width,
        }

        for method in METHODS:
            for ampli in AMPLIACIONS:
                matches = [
                    point
                    for point in mol_points
                    if point["metode"] == method
                    and point["ampliacio_percent"] == ampli
                ]
                if not matches:
                    continue

                point = matches[0]
                x = group_centers[method] + offsets[ampli]
                axis.bar(
                    x,
                    point["error_vs_bibliografia_kcal_mol"],
                    width=bar_width,
                    color=AMPLI_COLORS[ampli],
                    edgecolor="black",
                    linewidth=0.5,
                    yerr=point["incertesa_simulacio_kcal_mol"],
                    capsize=3,
                    label=f"{ampli}%",
                )

        axis.set_title(f"{short_label(name)} ({mol}) | bibliografia = {ref:.2f} kcal/mol")
        axis.set_ylabel("Error (kcal/mol)")
        axis.set_ylim(-y_lim, y_lim)
        axis.set_xlim(-0.55, 1.55)
        axis.grid(alpha=0.25)

    axes[-1].set_xticks([0, 1])
    axes[-1].set_xticklabels(["PBC", "SBC"])
    axes[-1].set_xlabel("Mètode de simulació")

    legend_handles = [
        Patch(
            facecolor=AMPLI_COLORS[ampli],
            edgecolor="black",
            linewidth=0.5,
            label=f"{ampli}%",
        )
        for ampli in AMPLIACIONS
    ]
    fig.legend(
        legend_handles,
        [f"{ampli}%" for ampli in AMPLIACIONS],
        title="Ampliació",
        loc="lower center",
        ncol=len(AMPLIACIONS),
    )

    fig.suptitle(
        "Convergència de ΔG lliure d'hidratació respecte el valor bibliogràfic",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)


def main():
    rows = load_database()
    points = collect_points(rows)
    save_points_csv(points)
    plot_error_bars(points)

    print("Punts trobats:", len(points))
    for mol, mol_points in grouped_by_molecule(points).items():
        name = mol_points[0]["nom iupac molec"]
        print(f"{mol} - {name}")
        for method in METHODS:
            n = sum(point["metode"] == method for point in mol_points)
            print(f"  {method}: {n}")
    print("Gràfica guardada a:", OUT_PNG)
    print("Dades guardades a:", OUT_CSV)


if __name__ == "__main__":
    main()
