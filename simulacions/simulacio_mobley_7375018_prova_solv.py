#!/usr/bin/env python3  # Indica que se ejecute con Python 3.
from pathlib import Path  # Maneja rutas de archivos de forma portable.

import openmm as mm  # Importa OpenMM (núcleo).
from openmm import app, unit  # Importa utilidades de aplicación y unidades.
from openmm.app import (  # Importa clases para leer GROMACS y reportar.
    DCDReporter,
    ForceField,
    GromacsGroFile,
    GromacsTopFile,
    Modeller,
    PDBFile,
    Simulation,
    StateDataReporter,
)

REPO_DIR = Path(__file__).resolve().parents[1]  # Calcula la raíz del repo.
DATA_DIR = REPO_DIR / "DADES" / "v0.1" / "topgro"  # Ruta a los datos por defecto.
TOP_IN = DATA_DIR / "mobley_7375018.top"  # Ruta del archivo .top específico.
GRO_IN = DATA_DIR / "mobley_7375018.gro"  # Ruta del archivo .gro específico.
OUT_DIR = REPO_DIR / "resultats" / "simulacio_mobley_7375018_prova_solv"   # Carpeta de salida. # CARLA: NO ENTENC PQ EM FA UNA SUBCARPETA DE RESULTATS QUE ES DIU SIMPLE
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta de salida si no existe.

IGNORE_SOLVENT = True  # Cambia a False si quieres usar solvente explícito.
USE_SOLVATION = True  # Activa la solvatación en OpenMM con PBC.
# CAMP DE FORCES + MODEL D'AIGUA+ PADDING DE CAIXA: charmm36.xml + charmm36/water.xml, padding 1.0 nm
# primer ho he provat amb amber14-all.xml + amber14/tip3p.xml pero em diu que el residu TMP no esta parametritzat
#amb chamber diu el mateix
SOLVENT_FORCEFIELD = ("charmm36.xml", "charmm36/water.xml")
SOLVENT_MODEL = "tip3p"
SOLVENT_PADDING_NM = 1.0
SOLUTE_FFXML = None  # Si tens un ffxml del solut (TMP), posa el path aquí.

if not TOP_IN.exists():  # Verifica que el .top exista.
    raise FileNotFoundError(f"Missing .top file: {TOP_IN}")  # Lanza error si falta.
if not GRO_IN.exists():  # Verifica que el .gro exista.
    raise FileNotFoundError(f"Missing .gro file: {GRO_IN}")  # Lanza error si falta.


def _filtered_topology(top_path: Path, out_dir: Path) -> Path:  # Crea una topología sin solvente.
    filtered_path = out_dir / "filtered_no_solvent.top"  # Define el archivo filtrado.
    in_molecules = False  # Marca si estamos en la sección [ molecules ].
    with open(top_path, "r", encoding="utf-8") as src, open(filtered_path, "w", encoding="utf-8") as dst:  # Abre archivos.
        for line in src:  # Recorre cada línea del .top original.
            stripped = line.strip()  # Quita espacios para analizar la línea.
            if stripped.startswith("[") and stripped.endswith("]"):  # Detecta inicio de sección.
                in_molecules = stripped.lower() == "[ molecules ]"  # Activa el modo moléculas.
                dst.write(line)  # Escribe la línea de sección.
                continue  # Pasa a la siguiente línea.
            if in_molecules:  # Si estamos en [ molecules ].
                if not stripped or stripped.startswith(";"):  # Conserva líneas vacías o comentarios.
                    dst.write(line)  # Escribe la línea tal cual.
                    continue  # Pasa a la siguiente línea.
                name = stripped.split()[0]  # Obtiene el nombre de la molécula.
                if name == "SOL":  # Detecta solvente.
                    continue  # Omite líneas de SOL.
            dst.write(line)  # Escribe cualquier otra línea.
    return filtered_path  # Devuelve el .top filtrado.


def _filtered_positions(gro_file: GromacsGroFile, gro_path: Path):  # Filtra posiciones para quitar solvente.
    positions = gro_file.getPositions()  # Lista de posiciones originales.
    with open(gro_path, "r", encoding="utf-8") as src:  # Lee el .gro para extraer residuos.
        lines = src.readlines()  # Carga todas las líneas.
    if len(lines) < 3:  # Verifica formato mínimo.
        raise ValueError(f"Invalid .gro file (too short): {gro_path}")  # Error si es inválido.
    try:
        n_atoms = int(lines[1].strip())  # Número de átomos en la cabecera.
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in .gro file: {gro_path}") from exc  # Error si no es entero.
    atom_lines = lines[2 : 2 + n_atoms]  # Líneas con átomos.
    if len(atom_lines) != len(positions):  # Verifica coherencia con posiciones.
        raise ValueError(
            f"Mismatch between .gro atoms ({len(atom_lines)}) and positions ({len(positions)})"
        )  # Error si no coincide.
    filtered = []  # Lista de posiciones filtradas.
    for line, pos in zip(atom_lines, positions):  # Recorre átomos y posiciones.
        resname = line[5:10].strip()  # Nombre del residuo según formato .gro.
        if resname == "SOL":  # Detecta solvente.
            continue  # Omite posiciones de SOL.
        filtered.append(pos)  # Guarda posiciones no solvente.
    return unit.Quantity(filtered, positions.unit)  # Devuelve posiciones filtradas con unidades.


def _as_quantity(value, unit_obj):  # Asegura que el valor tenga unidades.
    return value if hasattr(value, "unit") else value * unit_obj


def _box_size_from_positions(positions, padding_nm):  # Calcula boxSize con padding en nm.
    if hasattr(positions, "unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    min_x = min(p.x for p in pos_nm)
    min_y = min(p.y for p in pos_nm)
    min_z = min(p.z for p in pos_nm)
    max_x = max(p.x for p in pos_nm)
    max_y = max(p.y for p in pos_nm)
    max_z = max(p.z for p in pos_nm)
    size_x = (max_x - min_x) + 2 * padding_nm
    size_y = (max_y - min_y) + 2 * padding_nm
    size_z = (max_z - min_z) + 2 * padding_nm
    return mm.Vec3(size_x, size_y, size_z) * unit.nanometer


gro = GromacsGroFile(str(GRO_IN))  # Lee coordenadas desde el .gro.
TOP_TO_LOAD = _filtered_topology(TOP_IN, OUT_DIR) if IGNORE_SOLVENT else TOP_IN  # Decide qué .top cargar.
top = GromacsTopFile(str(TOP_TO_LOAD), periodicBoxVectors=gro.getPeriodicBoxVectors())  # Lee topología y caja.

integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)  # Define el integrador.

positions = _filtered_positions(gro, GRO_IN) if IGNORE_SOLVENT else gro.getPositions()  # Ajusta posiciones si se filtra solvente.
if USE_SOLVATION:
    ff_files = list(SOLVENT_FORCEFIELD)  # Fuerza de campo per a solut+agua.
    if SOLUTE_FFXML:
        ff_files.insert(0, str(SOLUTE_FFXML))
    forcefield = ForceField(*ff_files)
    modeller = Modeller(top.topology, positions)  # Construye el modelo editable.
    box_size = _box_size_from_positions(positions, SOLVENT_PADDING_NM)
    try:
        modeller.addSolvent(
            forcefield,
            model=SOLVENT_MODEL,
            boxSize=box_size,
        )  # Añade solvente con PBC.
        system = forcefield.createSystem(  # Sistema con PBC y PME.
            modeller.topology,
            nonbondedMethod=app.PME,
            constraints=app.HBonds,
        )
    except ValueError as exc:
        raise RuntimeError(
            "ERROR: la solvatació ha fallat. Detall original: "
            f"{exc}. Si és un problema de plantilla del solut (TMP), "
            "cal un ffxml del solut (p. ex. OpenFF/GAFF) per continuar."
        ) from exc

    solvated_pdb = OUT_DIR / "solvated.pdb"
    with open(solvated_pdb, "w", encoding="utf-8") as pdb_file:
        PDBFile.writeFile(modeller.topology, modeller.positions, pdb_file)
    topology_for_sim = modeller.topology
    positions_for_sim = modeller.positions
else:
    system = top.createSystem(  # Sistema del .top (sense solvent si IGNORE_SOLVENT=True).
        nonbondedMethod=app.PME,
        constraints=app.HBonds,
    )
    topology_for_sim = top.topology
    positions_for_sim = positions

simulation = Simulation(topology_for_sim, system, integrator)  # Crea la simulación.
simulation.context.setPositions(positions_for_sim)  # Asigna posiciones iniciales.

simulation.minimizeEnergy(maxIterations=200)  # Minimiza la energía.
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)  # Inicializa velocidades.

simulation.reporters.append(StateDataReporter(str(OUT_DIR / "log.csv"), 1000, step=True, temperature=True, potentialEnergy=True))  # Añade log.
simulation.reporters.append(DCDReporter(str(OUT_DIR / "traj.dcd"), 1000))  # Añade trayectoria.

simulation.step(5000)  # Ejecuta la producción.
state = simulation.context.getState(getPositions=True)  # Obtiene posiciones finales.
pdb_path = OUT_DIR / "final.pdb"  # Archivo PDB de salida.
with open(pdb_path, "w", encoding="utf-8") as pdb_file:  # Escribe el PDB.
    PDBFile.writeFile(simulation.topology, state.getPositions(), pdb_file)

print("Written", OUT_DIR / "log.csv", "and", OUT_DIR / "traj.dcd", "and", pdb_path)  # Informa salidas.
