from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "DADES" / "v0.31" / "mol2files_gaff"
WORK = ROOT / "DADES" / "pretractament_work"
OUTPUT = ROOT / "DADES" / "topgro_actu"

WORK.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)


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
        "-d", "1.0",
        "-bt", "cubic",
    ], check=True)

    subprocess.run([
        "gmx", "solvate",
        "-cp", str(carpeta / f"{nom}_box.gro"),
        "-cs", "spc216.gro",
        "-o", str(carpeta / f"{nom}_GMX.gro"),
        "-p", str(acpype_dir / f"{nom}_GMX.top"),
    ], check=True)

    shutil.copy2(carpeta / f"{nom}_GMX.gro", OUTPUT / f"{nom}_GMX.gro")
    shutil.copy2(acpype_dir / f"{nom}_GMX.top", OUTPUT / f"{nom}_GMX.top")
    shutil.copy2(acpype_dir / f"{nom}_GMX.itp", OUTPUT / f"{nom}_GMX.itp")

    posre = acpype_dir / f"posre_{nom}.itp"
    if posre.exists():
        shutil.copy2(posre, OUTPUT / posre.name)

    print(f"{nom} completat")
