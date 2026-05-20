from pathlib import Path
import math
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "DADES" / "v0.31" / "mol2files_gaff"
WORK = ROOT / "DADES" / "pretractament_work_SBC"
OUTPUT = ROOT / "DADES" / "topgro_SBC"
COMMON = ROOT / "DADES" / "common"

WATER_ITP = COMMON / "SOL.TIP3P.itp"
RADIUS_NM = 1.8
BOX_PADDING_NM = 2.0

WORK.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

for programa in ["acpype", "gmx"]:
    if shutil.which(programa) is None:
        raise RuntimeError(
            f"No trobo la comanda '{programa}'. Instal.la-la o activa l'entorn/module que la contingui."
        )

if not WATER_ITP.exists():
    raise FileNotFoundError(f"No trobo el fitxer: {WATER_ITP}")


def xyz(line):
    return (
        float(line[20:28]),
        float(line[28:36]),
        float(line[36:44]),
    )


def retallar_esfera(gro_in, top_in, gro_out, top_out, water_itp, radius):
    with open(gro_in) as f:
        lines = f.readlines()

    title = lines[0]
    atoms = lines[2:-1]
    box = lines[-1]

    solute_atoms = []
    waters = {}

    for line in atoms:
        resid = line[:5].strip()
        resname = line[5:10].strip()

        if resname in ["SOL", "WAT", "HOH"]:
            waters.setdefault(resid, []).append(line)
        else:
            solute_atoms.append(line)

    if not solute_atoms:
        raise RuntimeError("No he trobat atoms de solut.")

    cx = sum(xyz(a)[0] for a in solute_atoms) / len(solute_atoms)
    cy = sum(xyz(a)[1] for a in solute_atoms) / len(solute_atoms)
    cz = sum(xyz(a)[2] for a in solute_atoms) / len(solute_atoms)

    new_atoms = solute_atoms[:]
    n_water = 0

    for wat in waters.values():
        ox, oy, oz = xyz(wat[0])
        d = math.sqrt((ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2)

        if d <= radius:
            new_atoms.extend(wat)
            n_water += 1

    with open(gro_out, "w") as f:
        f.write(title)
        f.write(f"{len(new_atoms):5d}\n")

        for i, line in enumerate(new_atoms, start=1):
            f.write(line[:15] + f"{i:5d}" + line[20:])

        f.write(box)

    with open(top_in) as f:
        top_lines = f.readlines()

    include_line = f'#include "{water_itp.as_posix()}"\n'

    new_top = []
    added_include = False
    in_molecules = False

    for line in top_lines:
        clean = line.strip()

        if clean.startswith("[ system ]") and not added_include:
            new_top.append("\n")
            new_top.append(include_line)
            new_top.append("\n")
            added_include = True

        if clean.startswith("[ molecules ]"):
            in_molecules = True
            new_top.append(line)
            continue

        if in_molecules and clean.startswith("SOL"):
            new_top.append(f"SOL {n_water}\n")
        else:
            new_top.append(line)

    with open(top_out, "w") as f:
        f.writelines(new_top)

    return n_water


for mol2 in INPUT.glob("*.mol2"):

    nom = mol2.stem
    print(f"\nProcessant {nom}")

    carpeta = WORK / nom
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir()

    subprocess.run(["acpype", "-i", str(mol2), "-b", nom, "-o", "gmx"], cwd=carpeta, check=True)

    acpype_dir = carpeta / f"{nom}.acpype"

    subprocess.run([
        "gmx", "editconf",
        "-f", str(acpype_dir / f"{nom}_GMX.gro"),
        "-o", str(carpeta / f"{nom}_box.gro"),
        "-c",
        "-d", str(BOX_PADDING_NM),
        "-bt", "cubic",
    ], check=True)

    subprocess.run([
        "gmx", "solvate",
        "-cp", str(carpeta / f"{nom}_box.gro"),
        "-cs", "spc216.gro",
        "-o", str(carpeta / f"{nom}_water.gro"),
        "-p", str(acpype_dir / f"{nom}_GMX.top"),
    ], check=True)

    retallar_esfera(
        carpeta / f"{nom}_water.gro",
        acpype_dir / f"{nom}_GMX.top",
        carpeta / f"{nom}_GMX.gro",
        carpeta / f"{nom}_GMX.top",
        WATER_ITP,
        RADIUS_NM,
    )

    shutil.copy2(carpeta / f"{nom}_GMX.gro", OUTPUT / f"{nom}_GMX.gro")
    shutil.copy2(carpeta / f"{nom}_GMX.top", OUTPUT / f"{nom}_GMX.top")
    shutil.copy2(acpype_dir / f"{nom}_GMX.itp", OUTPUT / f"{nom}_GMX.itp")

    posre = acpype_dir / f"posre_{nom}.itp"
    if posre.exists():
        shutil.copy2(posre, OUTPUT / posre.name)

    print(f"{nom} completat")
