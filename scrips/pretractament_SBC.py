from pathlib import Path
import subprocess
import shutil
import math

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V031_DIR = PROJECT_ROOT / "DADES/v0.31"

# ============================================================
# CONFIGURACIÓ
# ============================================================

INPUT_DIR = V031_DIR / "mol2files_gaff"
OUT_DIR = PROJECT_ROOT / "DADES/topgro_sbc"
COMMON_DIR = PROJECT_ROOT / "DADES/common"

WATER_ITP = COMMON_DIR / "SOL.TIP3P.itp"

RADIUS_NM = 1.8
BOX_PADDING_NM = 2.0

# None = totes les molècules
TEST_MOLECULE = None

ERROR_FILE = PROJECT_ROOT / "errors_sbc.txt"


# ============================================================
# FUNCIONS
# ============================================================

def run(cmd, cwd=None):
    subprocess.run(
        cmd,
        check=True,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def xyz(line):
    return (
        float(line[20:28]),
        float(line[28:36]),
        float(line[36:44])
    )


def is_inside(path, folder):
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


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
        raise RuntimeError("No he trobat àtoms de solut.")

    cx = sum(xyz(a)[0] for a in solute_atoms) / len(solute_atoms)
    cy = sum(xyz(a)[1] for a in solute_atoms) / len(solute_atoms)
    cz = sum(xyz(a)[2] for a in solute_atoms) / len(solute_atoms)

    new_atoms = solute_atoms[:]
    n_water = 0

    for wat in waters.values():
        ox, oy, oz = xyz(wat[0])
        d = math.sqrt((ox - cx)**2 + (oy - cy)**2 + (oz - cz)**2)

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


# ============================================================
# COMPROVACIONS INICIALS
# ============================================================

if is_inside(OUT_DIR, V031_DIR):
    raise RuntimeError(f"No escric dins {V031_DIR}. Canvia OUT_DIR: {OUT_DIR}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

if ERROR_FILE.exists():
    ERROR_FILE.unlink()

if not INPUT_DIR.exists():
    raise FileNotFoundError(f"No existeix la carpeta: {INPUT_DIR}")

if not WATER_ITP.exists():
    raise FileNotFoundError(f"No trobo el fitxer: {WATER_ITP}")


# ============================================================
# TRIAR MOLÈCULES
# ============================================================

if TEST_MOLECULE is not None:
    mol2_files = [INPUT_DIR / f"{TEST_MOLECULE}.mol2"]
else:
    mol2_files = sorted(INPUT_DIR.glob("*.mol2"))

if not mol2_files:
    raise RuntimeError(f"No he trobat cap .mol2 dins {INPUT_DIR}")


# ============================================================
# LOOP PRINCIPAL
# ============================================================

total = len(mol2_files)
ok = 0
failed = 0

for mol2_file in mol2_files:
    name = mol2_file.stem
    tmp_dir = Path(f"tmp_sbc_{name}")

    try:
        if not mol2_file.exists():
            raise FileNotFoundError(f"No existeix: {mol2_file}")

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tmp_dir.mkdir()
        shutil.copy(mol2_file, tmp_dir / mol2_file.name)

        tmp_abs = tmp_dir.resolve()

        run([
            "acpype",
            "-i", str(tmp_abs / mol2_file.name),
            "-b", name,
            "-a", "gaff"
        ], cwd=tmp_abs)

        acpype_dir = tmp_dir / f"{name}.acpype"
        gmx_top = acpype_dir / f"{name}_GMX.top"
        gmx_gro = acpype_dir / f"{name}_GMX.gro"
        gmx_itp = acpype_dir / f"{name}_GMX.itp"
        posre_itp = acpype_dir / f"posre_{name}.itp"

        if not gmx_top.exists():
            raise FileNotFoundError(gmx_top)

        if not gmx_gro.exists():
            raise FileNotFoundError(gmx_gro)

        if not gmx_itp.exists():
            raise FileNotFoundError(gmx_itp)

        centered_gro = tmp_dir / f"{name}_center.gro"

        run([
            "gmx", "editconf",
            "-f", str(gmx_gro),
            "-o", str(centered_gro),
            "-c",
            "-d", str(BOX_PADDING_NM),
            "-bt", "cubic"
        ])

        water_gro = tmp_dir / f"{name}_water.gro"

        run([
            "gmx", "solvate",
            "-cp", str(centered_gro),
            "-cs", "spc216.gro",
            "-o", str(water_gro),
            "-p", str(gmx_top)
        ])

        final_gro = OUT_DIR / f"{name}_sbc.gro"
        final_top = OUT_DIR / f"{name}_sbc.top"

        retallar_esfera(
            water_gro,
            gmx_top,
            final_gro,
            final_top,
            WATER_ITP,
            RADIUS_NM
        )

        shutil.copy(gmx_itp, OUT_DIR / gmx_itp.name)

        if posre_itp.exists():
            shutil.copy(posre_itp, OUT_DIR / posre_itp.name)

        ok += 1

    except Exception as e:
        failed += 1

        with open(ERROR_FILE, "a") as f:
            f.write(f"{name}: {repr(e)}\n")

        print(f"ERROR en {name}. Mira {ERROR_FILE}")

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


# ============================================================
# FINAL
# ============================================================

print("\nPROCÉS ACABAT")
print(f"Molècules totals: {total}")
print(f"Correctes: {ok}")
print(f"Errors: {failed}")
print(f"Fitxers finals a: {OUT_DIR}")

if failed > 0:
    print(f"Errors guardats a: {ERROR_FILE}")
