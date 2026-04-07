#!/usr/bin/env python3  # Indica que se ejecute con Python 3.
from pathlib import Path  # Maneja rutas de archivos de forma portable.

import openmm as mm  # Importa OpenMM (núcleo).
from openmm import app, unit  # Importa utilidades de aplicación y unidades.
from openmm.app import (  # Importa clases para leer GROMACS y reportar.
    DCDReporter,
    GromacsGroFile,
    GromacsTopFile,
    PDBFile,
    Simulation,
    StateDataReporter,
)

REPO_DIR = Path(__file__).resolve().parents[1]  # Raíz del repositorio.
DATA_DIR = REPO_DIR / "DADES" / "v0.1" / "topgro"  # Carpeta con los .gro/.top de entrada.
OUT_BASE = REPO_DIR / "resultats" / "simulacio_mobley_10p"  # Carpeta de salida del lote.
OUT_BASE.mkdir(parents=True, exist_ok=True)  # Crea la carpeta base si no existe.

IGNORE_SOLVENT = True  # True = filtra el solvente SOL del .top/.gro.
N_MOLECULES = 10  # Número de moléculas a procesar.


def _filtered_topology(top_path: Path, out_dir: Path) -> Path:  # Genera un .top sin solvente.
    # Se elimina cualquier entrada SOL dentro de la sección [ molecules ].
    filtered_path = out_dir / "filtered_no_solvent.top"  # Archivo .top filtrado.
    in_molecules = False  # Controla si estamos dentro de la sección [ molecules ].
    with open(top_path, "r", encoding="utf-8") as src, open(filtered_path, "w", encoding="utf-8") as dst:  # Abre archivos.
        for line in src:  # Recorre cada línea del .top original.
            stripped = line.strip()  # Quita espacios para analizar la línea.
            if stripped.startswith("[") and stripped.endswith("]"):  # Detecta inicio de sección.
                in_molecules = stripped.lower() == "[ molecules ]"  # Activa el modo moléculas.
                dst.write(line)  # Escribe la línea de sección.
                continue  # Pasa a la siguiente línea.
            if in_molecules:  # Solo filtramos dentro de [ molecules ].
                if not stripped or stripped.startswith(";"):  # Conserva líneas vacías o comentarios.
                    dst.write(line)  # Escribe la línea tal cual.
                    continue  # Pasa a la siguiente línea.
                name = stripped.split()[0]  # Obtiene el nombre de la molécula.
                if name == "SOL":  # Detecta solvente.
                    continue  # Omite líneas de SOL.
            dst.write(line)  # Escribe cualquier otra línea.
    return filtered_path  # Devuelve el .top filtrado.


def _filtered_positions(gro_file: GromacsGroFile, gro_path: Path):  # Filtra posiciones (quita SOL).
    # El .gro contiene todas las posiciones, pero aquí eliminamos las del solvente SOL.
    positions = gro_file.getPositions()  # Posiciones originales con unidades.
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
    filtered = []  # Posiciones filtradas (sin SOL).
    for line, pos in zip(atom_lines, positions):  # Recorre átomos y posiciones.
        resname = line[5:10].strip()  # Nombre del residuo según formato .gro.
        if resname == "SOL":  # Detecta solvente.
            continue  # Omite posiciones de SOL.
        filtered.append(pos)  # Guarda posiciones no solvente.
    return unit.Quantity(filtered, positions.unit)  # Mantiene unidades originales.


def _mobley_code(path: Path) -> str:  # Extrae el código del nombre del archivo.
    # Ejemplo: mobley_1234567.gro -> "1234567"
    return path.stem.split("_")[-1]


# Selecciona las primeras N moléculas por orden alfabético de nombre de archivo.
gro_files = sorted(DATA_DIR.glob("mobley_*.gro"))[:N_MOLECULES]
if not gro_files:
    raise FileNotFoundError(f"No .gro files found in {DATA_DIR}")

for gro_path in gro_files:
    code = _mobley_code(gro_path)  # Código de la molécula (usado en nombres de salida).
    top_path = DATA_DIR / f"mobley_{code}.top"
    if not top_path.exists():
        raise FileNotFoundError(f"Missing .top file: {top_path}")

    out_dir = OUT_BASE / f"mobley_{code}"  # Carpeta específica para esta molécula.
    out_dir.mkdir(parents=True, exist_ok=True)

    gro = GromacsGroFile(str(gro_path))  # Lee coordenadas desde el .gro.
    # Si IGNORE_SOLVENT=True, se usa un .top filtrado sin SOL.
    top_to_load = _filtered_topology(top_path, out_dir) if IGNORE_SOLVENT else top_path
    top = GromacsTopFile(str(top_to_load), periodicBoxVectors=gro.getPeriodicBoxVectors())

    # Crea el sistema físico con restricciones en enlaces a H.
    system = top.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    # Integrador estándar Langevin (300 K, fricción 1/ps, dt 2 fs).
    integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    simulation = Simulation(top.topology, system, integrator)
    # Usa posiciones filtradas si se elimina solvente.
    positions = _filtered_positions(gro, gro_path) if IGNORE_SOLVENT else gro.getPositions()
    simulation.context.setPositions(positions)

    simulation.minimizeEnergy(maxIterations=200)  # Minimiza energía inicial.
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)  # Inicializa velocidades.

    # Rutas de salida para esta molécula.
    log_path = out_dir / f"log_{code}.csv"
    traj_path = out_dir / f"traj_{code}.dcd"
    pdb_path = out_dir / f"final_{code}.pdb"

    # Reporters: log CSV y trayectoria DCD cada 1000 pasos.
    simulation.reporters.append(StateDataReporter(str(log_path), 1000, step=True, temperature=True, potentialEnergy=True))
    simulation.reporters.append(DCDReporter(str(traj_path), 1000))

    simulation.step(5000)  # Producción corta.
    state = simulation.context.getState(getPositions=True)  # Captura posiciones finales.
    with open(pdb_path, "w", encoding="utf-8") as pdb_file:
        PDBFile.writeFile(simulation.topology, state.getPositions(), pdb_file)

    print("Written", log_path, "and", traj_path, "and", pdb_path)
