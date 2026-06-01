from pathlib import Path
import argparse
import builtins
import json
import math
import shutil
import time

TARGET_MOLECULES = {
    "mobley_1723043",  # mes positiva: octafluorocyclobutane
    "mobley_9534740",  # mes negativa
    "mobley_1893937",  # mes propera a 0: 1-chlorohexane
}


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--molecule",
    required=True,
    help="Nom de la molècula, per exemple mobley_5857"
)

parser.add_argument(
    "-s",
    "--solvent-file",
    "--solvent-itp",
    dest="solvent_file",
    required=True,
    help="Nom del fitxer .itp del solvent dins DADES/common, per exemple SOL.TIP3P.itp"
)

args = parser.parse_args()

if args.molecule not in TARGET_MOLECULES:
    allowed = ", ".join(sorted(TARGET_MOLECULES))
    raise SystemExit(
        f"Molècula descartada: {args.molecule}. "
        f"Aquesta ampliació només calcula: {allowed}"
    )

from openmm import *
from openmm.app import *
from openmm.unit import *

import mdtraj as md
import numpy as np


# ============================================================
# RUTES GENERALS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

data_dir = ROOT / "DADES"

# CANVI PBC -> SBC:
# abans: DADES/topgro_actu
# ara:   DADES/topgro_SBC
topgro_dir = data_dir / "topgro_SBC"

common_dir = data_dir / "common"


# ============================================================
# LOCALITZAR FITXERS .TOP I .GRO
# ============================================================

def file_candidates(directory, molecule, extension):
    patterns = [
        f"{molecule}_sbc.{extension}",
        f"{molecule}_GMX.{extension}",
        f"{molecule}*.{extension}",
    ]

    candidates = []
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path not in candidates:
                candidates.append(path)

    return candidates


def choose_top_file(candidates):
    non_opls = [path for path in candidates if "OPLS" not in path.stem.upper()]
    return (non_opls or candidates)[0]


top_candidates = file_candidates(topgro_dir, args.molecule, "top")
gro_candidates = file_candidates(topgro_dir, args.molecule, "gro")

if not top_candidates:
    raise FileNotFoundError(f"No trobo cap .top per {args.molecule} dins {topgro_dir}")

if not gro_candidates:
    raise FileNotFoundError(f"No trobo cap .gro per {args.molecule} dins {topgro_dir}")

top_file = choose_top_file(top_candidates)
gro_file = gro_candidates[0]


# ============================================================
# SOLVENT DEFINIT EN UN .ITP
# ============================================================

def strip_gromacs_comment(line):
    return line.split(";", 1)[0].strip()


def gromacs_section(line):
    clean = strip_gromacs_comment(line)
    if clean.startswith("[") and "]" in clean:
        return clean[1:clean.index("]")].strip().lower()
    return None


def resolve_input_file(path_text):
    path = Path(path_text).expanduser()
    path_variants = [path]

    if path.suffix != ".itp":
        path_variants.append(Path(f"{path}.itp"))

    candidates = []

    if path.is_absolute():
        candidates.extend(path_variants)
    else:
        for variant in path_variants:
            candidates.extend([
                common_dir / variant,
                Path.cwd() / variant,
                ROOT / variant,
                topgro_dir / variant,
            ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"No trobo el fitxer: {path_text}")


def read_moleculetypes(path):
    moleculetypes = set()
    section = None

    with open(path) as f:
        for line in f:
            new_section = gromacs_section(line)
            if new_section is not None:
                section = new_section
                continue

            clean = strip_gromacs_comment(line)
            if not clean:
                continue

            if section == "moleculetype":
                moleculetypes.add(clean.split()[0])

    return moleculetypes


def read_solvent_names_from_itp(path):
    names = set()
    section = None

    with open(path) as f:
        for line in f:
            new_section = gromacs_section(line)
            if new_section is not None:
                section = new_section
                continue

            clean = strip_gromacs_comment(line)
            if not clean:
                continue

            parts = clean.split()

            if section == "moleculetype":
                names.add(parts[0])

            elif section == "atoms" and len(parts) >= 4:
                names.add(parts[3])

    if not names:
        raise RuntimeError(f"No he pogut llegir el nom del solvent dins {path}")

    return names


def expand_water_aliases(names):
    expanded = set(names)

    known_water_names = {"SOL", "WAT", "HOH", "TIP3P", "TP3", "T3P"}

    if expanded & known_water_names:
        expanded.update({"SOL", "WAT", "HOH"})

    return expanded


def solvent_output_label(path):
    stem = path.stem

    if stem.upper().startswith("SOL."):
        stem = stem.split(".", 1)[1]

    return stem.replace("/", "_").replace("\\", "_")


solvent_itp_file = resolve_input_file(args.solvent_file)
solvent_label = solvent_output_label(solvent_itp_file)

# CANVI PBC -> SBC:
# guardem els resultats en outputs_SBC_ampli30 per no barrejar-los amb PBC.
# A SBC, cada molecula te una carpeta propia sense subcarpeta de solvent.
output_root = ROOT / "outputs_SBC_ampli30" / args.molecule
output_root.mkdir(parents=True, exist_ok=True)

solvent_names = read_solvent_names_from_itp(solvent_itp_file)
solvent_names = expand_water_aliases(solvent_names)

if not solvent_names:
    raise RuntimeError(
        f"No he pogut llegir el solvent dins {solvent_itp_file}"
    )


def prepare_topology_file():
    top_moleculetypes = read_moleculetypes(top_file)
    itp_moleculetypes = read_moleculetypes(solvent_itp_file)

    prepared_top = output_root / f"{top_file.stem}_with_solvent.top"
    include_line = f'#include "{solvent_itp_file.as_posix()}"\n'

    with open(top_file) as f:
        lines = f.readlines()

    has_solvent_include = False
    new_lines = []

    for line in lines:
        if line.lstrip().startswith("#include") and solvent_itp_file.name in line:
            if not has_solvent_include:
                new_lines.append(include_line)
                has_solvent_include = True
            continue

        new_lines.append(line)

    if itp_moleculetypes and itp_moleculetypes.issubset(top_moleculetypes):
        with open(prepared_top, "w") as f:
            f.writelines(new_lines)

        return prepared_top

    if has_solvent_include:
        with open(prepared_top, "w") as f:
            f.writelines(new_lines)

        return prepared_top

    insert_at = next(
        ( i for i, line in enumerate(new_lines)
            if gromacs_section(line) in {"system", "molecules"}
        ),
        len(new_lines),
    )

    prepared_lines = new_lines[:insert_at]

    if prepared_lines and prepared_lines[-1].strip():
        prepared_lines.append("\n")

    prepared_lines.append(include_line)
    prepared_lines.append("\n")
    prepared_lines.extend(new_lines[insert_at:])

    with open(prepared_top, "w") as f:
        f.writelines(prepared_lines)

    return prepared_top


openmm_top_file = prepare_topology_file()

# ============================================================
# PARÀMETRES DE SIMULACIÓ
# ============================================================
temperature = 300 * kelvin
pressure = 1 * atmosphere
friction = 10 / picosecond
timestep = 0.002 * picoseconds

# Mateixos valors que el teu script de prova PBC
nvt_steps = 650
npt_steps = 650
prod_steps = 3250

report_interval = 100

lambdas_elec = [0.0, 0.25, 0.5, 0.75, 1.0]

lambdas_lj = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

kB = 0.008314462618
beta = 1.0 / (kB * 300.0)

n_bootstrap = 40
block_size = 2

# ============================================================
# FUNCIONS PETITES
# ============================================================

def lambda_tag(lam):
    return str(lam).replace(".", "p")

def load_gromacs():
    gro = GromacsGroFile(str(gro_file))

    # CANVI PBC -> SBC:
    # abans passàvem periodicBoxVectors=gro.getPeriodicBoxVectors()
    # ara NO ho fem perquè SBC no utilitza caixa periòdica.
    top = GromacsTopFile(
        str(openmm_top_file),
        includeDir=str(topgro_dir)
    )

    return gro, top


def get_solute_atoms():
    gro, top = load_gromacs()

    solute = [
        atom.index
        for atom in top.topology.atoms()
        if atom.residue.name not in solvent_names
    ]

    if not solute:
        raise RuntimeError("No he trobat àtoms del solut. Revisa el fitxer de solvent.")

    return solute


solute_atoms = get_solute_atoms()

# ============================================================
# PARET ESFÈRICA SBC
# ============================================================

# Ha de coincidir amb el radi utilitzat per crear els .gro SBC
sbc_radius = 1.8 * nanometer

# Constant harmònica de la paret
sbc_k = 1000.0 * kilojoule_per_mole / nanometer**2


def sbc_center_from_solute(positions):
    coords = [
        positions[i].value_in_unit(nanometer)
        for i in solute_atoms
    ]

    return (
        builtins.sum(p[0] for p in coords) / len(coords),
        builtins.sum(p[1] for p in coords) / len(coords),
        builtins.sum(p[2] for p in coords) / len(coords),
    )


def add_sbc_wall(system, positions):
    """
    SBC necessita una paret perquè no hi ha condicions periòdiques.
    PBC:        l'aigua es manté en una caixa periòdica.
    SBC:        l'aigua és una gota/esfera finita i cal contenir-la.
    """

    x0, y0, z0 = sbc_center_from_solute(positions)

    wall = CustomExternalForce(
        """
        step(r-r0)*0.5*k*(r-r0)^2;
        r=sqrt((x-x0)^2+(y-y0)^2+(z-z0)^2)
        """
    )

    wall.addGlobalParameter("r0", sbc_radius)
    wall.addGlobalParameter("k", sbc_k)
    wall.addGlobalParameter("x0", x0)
    wall.addGlobalParameter("y0", y0)
    wall.addGlobalParameter("z0", z0)

    for i in range(system.getNumParticles()):
        wall.addParticle(i, [])

    system.addForce(wall)

# ============================================================
# BAR I BOOTSTRAP
# ============================================================

def bar_delta_f(w_forward, w_reverse):
    def f(df):
        left = np.sum(1.0 / (1.0 + np.exp(w_forward - df)))
        right = np.sum(1.0 / (1.0 + np.exp(w_reverse + df)))
        return left - right

    lo = -100.0
    hi = 100.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)

        if f(mid) > 0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def make_blocks(x, block_size):
    n_blocks = len(x) // block_size

    if n_blocks < 1:
        raise ValueError("Massa pocs frames per fer block bootstrap")

    trimmed = x[: n_blocks * block_size]

    return trimmed.reshape(n_blocks, block_size)


def resample_blocks(x, block_size):
    blocks = make_blocks(x, block_size)
    choices = np.random.randint(0, len(blocks), size=len(blocks))

    return blocks[choices].reshape(-1)


def block_bootstrap_bar(wf, wr):
    values = []

    for _ in range(n_bootstrap):
        wf_bs = resample_blocks(wf, block_size)
        wr_bs = resample_blocks(wr, block_size)
        values.append(bar_delta_f(wf_bs, wr_bs))

    return np.std(values, ddof=1)


# ============================================================
# SISTEMA ELECTROSTÀTIC EN AIGUA
# ============================================================

def build_aq_elec_system(lambda_elec):
    gro, top = load_gromacs()

    system = top.createSystem(
        # CANVI PBC -> SBC:
        # abans: PME
        # ara: CutoffNonPeriodic
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=0.9 * nanometer,
        constraints=HBonds,
        rigidWater=True,
    )

    nb = None

    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nb = force
            break

    if nb is None:
        raise RuntimeError("No he trobat NonbondedForce")

    for i in solute_atoms:
        q, sigma, epsilon = nb.getParticleParameters(i)

        nb.setParticleParameters(
            i,
            (1.0 - lambda_elec) * q,
            sigma,
            epsilon
        )

    # CANVI PBC -> SBC:
    # afegim paret esfèrica a la part aquosa
    add_sbc_wall(system, gro.positions)

    return system, top.topology, gro.positions

# ============================================================
# SISTEMA LJ EN AIGUA
# ============================================================

def build_aq_lj_system(lambda_lj):
    gro, top = load_gromacs()

    system = top.createSystem(
        # CANVI PBC -> SBC:
        # abans: PME
        # ara: CutoffNonPeriodic
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=0.9 * nanometer,
        constraints=HBonds,
        rigidWater=True,
    )

    nb = None

    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nb = force
            break

    if nb is None:
        raise RuntimeError("No he trobat NonbondedForce")

    n_particles = system.getNumParticles()
    solute_atom_set = set(solute_atoms)

    params = [
        nb.getParticleParameters(i)
        for i in range(n_particles)
    ]

    lj_atoms = [
        i for i, (_, sigma, epsilon) in enumerate(params)
        if epsilon.value_in_unit(kilojoule_per_mole) > 0.0
    ]

    # Mateixa lògica que el script PBC:
    # per la pota LJ, el solut ja està sense càrregues.
    for i in range(n_particles):
        q, sigma, epsilon = nb.getParticleParameters(i)

        q_new = 0.0 * elementary_charge if i in solute_atom_set else q

        nb.setParticleParameters(
            i,
            q_new,
            sigma,
            0.0 * kilojoule_per_mole
        )

    original_exceptions = nb.getNumExceptions()

    for k in range(original_exceptions):
        i, j, chargeprod, sigma, epsilon = nb.getExceptionParameters(k)

        chargeprod_new = chargeprod

        if i in solute_atom_set or j in solute_atom_set:
            chargeprod_new = 0.0 * elementary_charge**2

        nb.setExceptionParameters(
            k,
            i,
            j,
            chargeprod_new,
            sigma,
            epsilon
        )

    softcore = CustomNonbondedForce(
        """
        4*epsilon*((1-solute_solvent)*(y*y - y) + solute_solvent*(1-lambda_lj)*(x*x - x));
        solute_solvent = solute1 + solute2 - 2*solute1*solute2;
        x = 1/(alpha*lambda_lj + (r/sigma)^6);
        y = (sigma/r)^6;
        sigma = 0.5*(sigma1 + sigma2);
        epsilon = sqrt(epsilon1*epsilon2);
        """
    )

    softcore.addGlobalParameter("lambda_lj", lambda_lj)
    softcore.addGlobalParameter("alpha", 0.5)

    softcore.addPerParticleParameter("sigma")
    softcore.addPerParticleParameter("epsilon")
    softcore.addPerParticleParameter("solute")

    for i, (q, sigma, epsilon) in enumerate(params):
        is_solute = 1.0 if i in solute_atom_set else 0.0
        softcore.addParticle([sigma, epsilon, is_solute])

    for k in range(original_exceptions):
        i, j, chargeprod, sigma, epsilon = nb.getExceptionParameters(k)
        softcore.addExclusion(i, j)

    # CANVI PBC -> SBC:
    # abans: CustomNonbondedForce.CutoffPeriodic
    # ara:   CustomNonbondedForce.CutoffNonPeriodic
    softcore.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    softcore.setCutoffDistance(0.9 * nanometer)
    softcore.setUseSwitchingFunction(True)
    softcore.setSwitchingDistance(0.8 * nanometer)
    softcore.addInteractionGroup(lj_atoms, lj_atoms)

    system.addForce(softcore)

    # CANVI PBC -> SBC:
    # afegim paret esfèrica a la part aquosa
    add_sbc_wall(system, gro.positions)

    return system, top.topology, gro.positions


# ============================================================
# CREAR SISTEMA GAS A PARTIR DEL .TOP/.GRO
# ============================================================

def build_gas_base_system(lambda_elec=0.0, scale_lj=False, lambda_lj=0.0):
    gro, top = load_gromacs()

    full_system = top.createSystem(
        nonbondedMethod=NoCutoff,
        constraints=HBonds
    )

    old_to_new = {
        old: new
        for new, old in enumerate(solute_atoms)
    }

    gas_top = Topology()
    chain = gas_top.addChain()
    residue = gas_top.addResidue("MOL", chain)

    old_atom_to_new_atom = {}

    for atom in top.topology.atoms():
        if atom.index in old_to_new:
            new_atom = gas_top.addAtom(
                atom.name,
                atom.element,
                residue
            )
            old_atom_to_new_atom[atom.index] = new_atom

    for bond in top.topology.bonds():
        i = bond[0].index
        j = bond[1].index

        if i in old_atom_to_new_atom and j in old_atom_to_new_atom:
            gas_top.addBond(
                old_atom_to_new_atom[i],
                old_atom_to_new_atom[j]
            )

    gas_positions = Quantity(
        [
            gro.positions[i].value_in_unit(nanometer)
            for i in solute_atoms
        ],
        nanometer,
    )

    gas_system = System()

    for old_i in solute_atoms:
        gas_system.addParticle(
            full_system.getParticleMass(old_i)
        )

    for force in full_system.getForces():

        if isinstance(force, HarmonicBondForce):
            new_force = HarmonicBondForce()

            for k in range(force.getNumBonds()):
                i, j, length, kspring = force.getBondParameters(k)

                if i in old_to_new and j in old_to_new:
                    new_force.addBond(
                        old_to_new[i],
                        old_to_new[j],
                        length,
                        kspring
                    )

            gas_system.addForce(new_force)

        elif isinstance(force, HarmonicAngleForce):
            new_force = HarmonicAngleForce()

            for k in range(force.getNumAngles()):
                i, j, l, angle, kspring = force.getAngleParameters(k)

                if i in old_to_new and j in old_to_new and l in old_to_new:
                    new_force.addAngle(
                        old_to_new[i],
                        old_to_new[j],
                        old_to_new[l],
                        angle,
                        kspring
                    )

            gas_system.addForce(new_force)

        elif isinstance(force, PeriodicTorsionForce):
            new_force = PeriodicTorsionForce()

            for k in range(force.getNumTorsions()):
                i, j, l, m, periodicity, phase, kspring = force.getTorsionParameters(k)

                if (
                    i in old_to_new
                    and j in old_to_new
                    and l in old_to_new
                    and m in old_to_new
                ):
                    new_force.addTorsion(
                        old_to_new[i],
                        old_to_new[j],
                        old_to_new[l],
                        old_to_new[m],
                        periodicity,
                        phase,
                        kspring
                    )

            gas_system.addForce(new_force)

        elif isinstance(force, RBTorsionForce):
            new_force = RBTorsionForce()

            for k in range(force.getNumTorsions()):
                params = force.getTorsionParameters(k)

                i, j, l, m = params[0], params[1], params[2], params[3]
                coeffs = params[4:]

                if (
                    i in old_to_new
                    and j in old_to_new
                    and l in old_to_new
                    and m in old_to_new
                ):
                    new_force.addTorsion(
                        old_to_new[i],
                        old_to_new[j],
                        old_to_new[l],
                        old_to_new[m],
                        *coeffs
                    )

            gas_system.addForce(new_force)

        elif isinstance(force, NonbondedForce):
            new_force = NonbondedForce()
            new_force.setNonbondedMethod(
                NonbondedForce.NoCutoff
            )

            for old_i in solute_atoms:
                q, sigma, epsilon = force.getParticleParameters(old_i)

                q_new = (1.0 - lambda_elec) * q

                epsilon_new = (
                    (1.0 - lambda_lj) * epsilon
                    if scale_lj
                    else epsilon
                )

                new_force.addParticle(
                    q_new,
                    sigma,
                    epsilon_new
                )

            for k in range(force.getNumExceptions()):
                i, j, chargeprod, sigma, epsilon = force.getExceptionParameters(k)

                if i in old_to_new and j in old_to_new:
                    eps_new = (
                        (1.0 - lambda_lj) * epsilon
                        if scale_lj
                        else epsilon
                    )

                    new_force.addException(
                        old_to_new[i],
                        old_to_new[j],
                        chargeprod,
                        sigma,
                        eps_new
                    )

            gas_system.addForce(new_force)

    return gas_system, gas_top, gas_positions


def build_gas_elec_system(lambda_elec):
    return build_gas_base_system(
        lambda_elec=lambda_elec,
        scale_lj=False,
        lambda_lj=0.0
    )


def build_gas_lj_system(lambda_lj):
    return build_gas_base_system(
        lambda_elec=1.0,
        scale_lj=True,
        lambda_lj=lambda_lj
    )


# ============================================================
# LOG GLOBAL DE LA MOLECULA
# ============================================================

molecule_log_file = output_root / "molecula.log"
molecule_log = {
    "completed_steps": 0,
    "total_steps": 0,
    "start_time": None,
    "window_summaries": []
}


def window_total_steps(is_aq):
    return nvt_steps + prod_steps


def molecule_total_steps():
    windows = 2 * (len(lambdas_elec) + len(lambdas_lj))
    return windows * window_total_steps(True)


def format_seconds(seconds):
    if seconds is None:
        return ""
    return f"{seconds:.1f}"


def init_molecule_log():
    molecule_log["completed_steps"] = 0
    molecule_log["total_steps"] = molecule_total_steps()
    molecule_log["start_time"] = time.perf_counter()
    molecule_log["window_summaries"] = []

    with open(molecule_log_file, "w") as f:
        f.write(f"molecule={args.molecule}\n")
        f.write(f"solvent_label={solvent_label}\n")
        f.write("mode=SBC\n")
        f.write(f"total_steps={molecule_log['total_steps']}\n")
        f.write("\n[progress]\n")
        f.write("event,window,lambda,phase,phase_step,phase_steps,window_step,window_steps,total_step,total_steps,progress_percent,elapsed_seconds,remaining_seconds\n")


def append_molecule_progress(event, name, lam, phase, phase_step, phase_steps, window_step, window_steps):
    elapsed = time.perf_counter() - molecule_log["start_time"]
    completed = molecule_log["completed_steps"]
    total = molecule_log["total_steps"]
    progress = 100.0 * completed / total if total else 0.0
    remaining = None
    if completed > 0:
        remaining = elapsed * (total - completed) / completed

    with open(molecule_log_file, "a") as f:
        f.write(
            f"{event},{name},{lambda_tag(lam)},{phase},{phase_step},{phase_steps},"
            f"{window_step},{window_steps},{completed},{total},{progress:.3f},"
            f"{format_seconds(elapsed)},{format_seconds(remaining)}\n"
        )


def step_with_molecule_log(simulation, steps, phase, name, lam, window_state):
    phase_step = 0
    remaining = steps

    while remaining > 0:
        chunk = min(report_interval, remaining)
        simulation.step(chunk)
        phase_step += chunk
        window_state["completed_steps"] += chunk
        molecule_log["completed_steps"] += chunk
        remaining -= chunk

        append_molecule_progress(
            "progress",
            name,
            lam,
            phase,
            phase_step,
            steps,
            window_state["completed_steps"],
            window_state["total_steps"]
        )


def append_molecule_summary():
    elapsed = time.perf_counter() - molecule_log["start_time"]

    with open(molecule_log_file, "a") as f:
        f.write("\n[summary]\n")
        f.write("window,lambda,steps,elapsed_seconds\n")
        for item in molecule_log["window_summaries"]:
            f.write(
                f"{item['name']},{lambda_tag(item['lambda'])},{item['steps']},{item['elapsed_seconds']:.1f}\n"
            )
        f.write(f"total_steps={molecule_log['completed_steps']}\n")
        f.write(f"total_elapsed_seconds={elapsed:.1f}\n")


# ============================================================
# SIMULACIÓ D'UNA FINESTRA
# ============================================================

def run_window(name, lam, builder, is_aq):
    outdir = output_root / name / f"lambda_{lambda_tag(lam)}"
    outdir.mkdir(parents=True, exist_ok=True)

    system, topology, positions = builder(lam)

    # CANVI PBC -> SBC:
    # abans, si is_aq, afegíem MonteCarloBarostat.
    # ara NO, perquè en SBC no hi ha caixa periòdica ni NPT.
    #
    # if is_aq:
    #     system.addForce(MonteCarloBarostat(pressure, temperature))

    integrator = LangevinMiddleIntegrator(
        temperature,
        friction,
        timestep
    )

    simulation = Simulation(
        topology,
        system,
        integrator
    )

    simulation.context.setPositions(positions)

    simulation.reporters.append(
        StateDataReporter(
            str(outdir / "state.csv"),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            progress=True,
            remainingTime=True,
            elapsedTime=True,
            totalSteps=window_total_steps(is_aq)
        )
    )

    if not is_aq:
        PDBFile.writeFile(
            topology,
            positions,
            open(outdir / "gas_topology.pdb", "w")
        )

    print(f"Simulant {name}, lambda={lam}")

    window_state = {"completed_steps": 0, "total_steps": window_total_steps(is_aq)}
    window_start = time.perf_counter()
    append_molecule_progress("start", name, lam, "setup", 0, 0, 0, window_state["total_steps"])

    simulation.minimizeEnergy(maxIterations=100)

    simulation.context.setVelocitiesToTemperature(
        temperature
    )

    # CANVI PBC -> SBC:
    # abans fèiem NVT + NPT.
    # ara només NVT.
    step_with_molecule_log(simulation, nvt_steps, "nvt", name, lam, window_state)

    simulation.reporters.append(
        DCDReporter(
            str(outdir / "traj.dcd"),
            report_interval
        )
    )

    step_with_molecule_log(simulation, prod_steps, "prod", name, lam, window_state)

    window_elapsed = time.perf_counter() - window_start
    molecule_log["window_summaries"].append({
        "name": name,
        "lambda": lam,
        "steps": window_state["completed_steps"],
        "elapsed_seconds": window_elapsed
    })
    append_molecule_progress("end", name, lam, "done", 0, 0, window_state["completed_steps"], window_state["total_steps"])


# ============================================================
# EXECUTAR TOTES LES SIMULACIONS
# ============================================================

def run_all_simulations():
    for lam in lambdas_elec:
        run_window(
            "elec_aq",
            lam,
            build_aq_elec_system,
            True
        )

    for lam in lambdas_lj:
        run_window(
            "lj_aq",
            lam,
            build_aq_lj_system,
            True
        )

    for lam in lambdas_elec:
        run_window(
            "elec_gas",
            lam,
            build_gas_elec_system,
            False
        )

    for lam in lambdas_lj:
        run_window(
            "lj_gas",
            lam,
            build_gas_lj_system,
            False
        )


# ============================================================
# RECALCULAR ENERGIES PER BAR
# ============================================================

def energies_for_traj(
    name,
    lambda_traj,
    lambda_a,
    lambda_b,
    builder,
    topology_file
):
    system_a, topology, _ = builder(lambda_a)
    system_b, _, _ = builder(lambda_b)

    sim_a = Simulation(
        topology,
        system_a,
        VerletIntegrator(0.001 * picoseconds)
    )

    sim_b = Simulation(
        topology,
        system_b,
        VerletIntegrator(0.001 * picoseconds)
    )

    traj_file = (
        output_root
        / name
        / f"lambda_{lambda_tag(lambda_traj)}"
        / "traj.dcd"
    )

    if name in {"elec_gas", "lj_gas"}:
        top_md = (
            output_root
            / name
            / f"lambda_{lambda_tag(lambda_traj)}"
            / "gas_topology.pdb"
        )
    else:
        top_md = topology_file

    traj = md.load(
        str(traj_file),
        top=str(top_md)
    )

    e_a = []
    e_b = []

    # En SBC normalment no hi haurà vectors periòdics.
    # Aquesta part es deixa per compatibilitat si mdtraj en troba.
    has_box = (
        traj.unitcell_vectors is not None
        and len(traj.unitcell_vectors) == traj.n_frames
        and not np.isnan(traj.unitcell_vectors).any()
    )

    for i in range(traj.n_frames):
        positions = traj.xyz[i] * nanometer

        if has_box:
            box = traj.unitcell_vectors[i]
            box_vectors = [
                Vec3(*box[j]) * nanometer
                for j in range(3)
            ]

            sim_a.context.setPeriodicBoxVectors(*box_vectors)
            sim_b.context.setPeriodicBoxVectors(*box_vectors)

        sim_a.context.setPositions(positions)
        sim_b.context.setPositions(positions)

        e_a.append(
            sim_a.context.getState(
                getEnergy=True
            ).getPotentialEnergy().value_in_unit(
                kilojoule_per_mole
            )
        )

        e_b.append(
            sim_b.context.getState(
                getEnergy=True
            ).getPotentialEnergy().value_in_unit(
                kilojoule_per_mole
            )
        )

    return np.array(e_a), np.array(e_b)


def analyze(name, lambdas, builder, topology_file):
    total_df = 0.0
    errors_df = []

    for lam_a, lam_b in zip(lambdas[:-1], lambdas[1:]):
        print(f"Analitzant {name}: {lam_a} -> {lam_b}")

        e_a_on_a, e_b_on_a = energies_for_traj(
            name,
            lam_a,
            lam_a,
            lam_b,
            builder,
            topology_file
        )

        wf = beta * (e_b_on_a - e_a_on_a)

        e_a_on_b, e_b_on_b = energies_for_traj(
            name,
            lam_b,
            lam_a,
            lam_b,
            builder,
            topology_file
        )

        wr = beta * (e_a_on_b - e_b_on_b)

        df = bar_delta_f(wf, wr)

        err = block_bootstrap_bar(wf, wr)

        total_df += df
        errors_df.append(err)

    dg_kcal = (total_df / beta) / 4.184

    err_kcal = (
        math.sqrt(
            np.sum(
                np.array(errors_df) ** 2
            )
        )
        / beta
    ) / 4.184

    return dg_kcal, err_kcal


# ============================================================
# MAIN
# ============================================================

print("Molècula:", args.molecule)
print("TOP:", top_file)
print("TOP OpenMM:", openmm_top_file)
print("GRO:", gro_file)
print("Fitxer solvent:", solvent_itp_file)
print("Etiqueta solvent:", solvent_label)
print("Solvent:", ", ".join(sorted(solvent_names)))
print("Àtoms solut:", len(solute_atoms))
print("Mode: SBC")
print("Output:", output_root)

shutil.copy(
    top_file,
    output_root / top_file.name
)

shutil.copy(
    gro_file,
    output_root / gro_file.name
)

init_molecule_log()
run_all_simulations()
append_molecule_summary()

DG_elec_aq, err_elec_aq = analyze(
    "elec_aq",
    lambdas_elec,
    build_aq_elec_system,
    gro_file
)

DG_lj_aq, err_lj_aq = analyze(
    "lj_aq",
    lambdas_lj,
    build_aq_lj_system,
    gro_file
)

DG_elec_gas, err_elec_gas = analyze(
    "elec_gas",
    lambdas_elec,
    build_gas_elec_system,
    gro_file
)

DG_lj_gas, err_lj_gas = analyze(
    "lj_gas",
    lambdas_lj,
    build_gas_lj_system,
    gro_file
)

DG_hyd = -(
    (DG_elec_aq + DG_lj_aq)
    - (DG_elec_gas + DG_lj_gas)
)

err_hyd = math.sqrt(
    err_elec_aq**2
    + err_lj_aq**2
    + err_elec_gas**2
    + err_lj_gas**2
)

results = {
    "molecule": args.molecule,
    "mode": "SBC",
    "solvent_label": solvent_label,
    "solvent": sorted(solvent_names),
    "solvent_file": str(solvent_itp_file),

    "DG_elec_aq_kcal_mol": DG_elec_aq,
    "err_elec_aq_kcal_mol": err_elec_aq,

    "DG_lj_aq_kcal_mol": DG_lj_aq,
    "err_lj_aq_kcal_mol": err_lj_aq,

    "DG_elec_gas_kcal_mol": DG_elec_gas,
    "err_elec_gas_kcal_mol": err_elec_gas,

    "DG_lj_gas_kcal_mol": DG_lj_gas,
    "err_lj_gas_kcal_mol": err_lj_gas,

    "DG_hyd_kcal_mol": DG_hyd,
    "err_hyd_kcal_mol": err_hyd,
}

with open(output_root / "results.json", "w") as f:
    json.dump(
        results,
        f,
        indent=2
    )

with open(output_root / "results.txt", "w") as f:
    f.write(
        f"DG_elec_aq  = {DG_elec_aq:.3f} ± {err_elec_aq:.3f} kcal/mol\n"
    )

    f.write(
        f"DG_lj_aq    = {DG_lj_aq:.3f} ± {err_lj_aq:.3f} kcal/mol\n"
    )

    f.write(
        f"DG_elec_gas = {DG_elec_gas:.3f} ± {err_elec_gas:.3f} kcal/mol\n"
    )

    f.write(
        f"DG_lj_gas   = {DG_lj_gas:.3f} ± {err_lj_gas:.3f} kcal/mol\n"
    )

    f.write(
        f"DG_hyd      = {DG_hyd:.3f} ± {err_hyd:.3f} kcal/mol\n"
    )

print("\nRESULTAT FINAL")
print(f"DG_elec_aq  = {DG_elec_aq:.3f} ± {err_elec_aq:.3f} kcal/mol")
print(f"DG_lj_aq    = {DG_lj_aq:.3f} ± {err_lj_aq:.3f} kcal/mol")
print(f"DG_elec_gas = {DG_elec_gas:.3f} ± {err_elec_gas:.3f} kcal/mol")
print(f"DG_lj_gas   = {DG_lj_gas:.3f} ± {err_lj_gas:.3f} kcal/mol")
print(f"DG_hyd      = {DG_hyd:.3f} ± {err_hyd:.3f} kcal/mol")
