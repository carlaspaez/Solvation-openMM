import json
import re
from pathlib import Path


CLUSTER_PATH = Path("outputs_SBC-ok/cluster")
CLUSTER_PBC_PATH = Path("outputs_PBC-ok/cluster_PBC")
MOLECULES_PATH = CLUSTER_PATH / "molecules_sbc_normal5.txt"
COST_COMPUTACIONAL_PATH = Path("resultats/costcompu.json")
MOLECULES_AMB_ERROR = {
    "mobley_3572203",
    "mobley_3589456",
    "mobley_6854178",
}


def carregar_molecules(path):
    with open(path, "r") as fitxer:
        return [linia.strip() for linia in fitxer if linia.strip()]


def indexar_elapsed_per_job(path):
    with open(path, "r") as fitxer:
        dades = json.load(fitxer)

    jobs = dades.get("jobs", [])
    return {
        str(job.get("JobID")): job.get("Elapsed")
        for job in jobs
        if job.get("JobID") is not None
    }


def extreure_job_id(error_path):
    match = re.match(r"error_(\d+).*_mobley_\d+\.txt$", error_path.name)
    if match is None:
        return None
    return match.group(1)


def elapsed_a_segons(elapsed):
    if elapsed is None:
        return 0

    parts = elapsed.split(":")
    if len(parts) == 3:
        hores, minuts, segons = parts
        return int(hores) * 3600 + int(minuts) * 60 + int(segons)
    if len(parts) == 4:
        dies, hores, minuts, segons = parts
        return (
            int(dies) * 24 * 3600
            + int(hores) * 3600
            + int(minuts) * 60
            + int(segons)
        )

    raise ValueError(f"Format Elapsed desconegut: {elapsed}")


def main():
    noms_molecules = carregar_molecules(MOLECULES_PATH)
    elapsed_per_job = indexar_elapsed_per_job(COST_COMPUTACIONAL_PATH)
    molecules_pbc = [
        nom_molecula
        for nom_molecula in noms_molecules
        if nom_molecula not in MOLECULES_AMB_ERROR
    ]

    elapsed_sbc = []
    detall_elapsed_sbc = []
    elapsed_pbc = []
    detall_elapsed_pbc = []

    for nom_molecula in noms_molecules:
        error_files = sorted(CLUSTER_PATH.glob(f"error_*_{nom_molecula}.txt"))

        if not error_files:
            detall_elapsed_sbc.append({
                "molecula": nom_molecula,
                "job_id_sbc": None,
                "elapsed_sbc": None,
            })
            continue

        job_id_sbc = extreure_job_id(error_files[0])
        elapsed = elapsed_per_job.get(job_id_sbc)

        elapsed_sbc.append(elapsed)
        detall_elapsed_sbc.append({
            "molecula": nom_molecula,
            "job_id_sbc": job_id_sbc,
            "elapsed_sbc": elapsed,
        })

    for nom_molecula in molecules_pbc:
        error_files = sorted(CLUSTER_PBC_PATH.glob(f"error_*_{nom_molecula}.txt"))

        if not error_files:
            detall_elapsed_pbc.append({
                "molecula": nom_molecula,
                "job_id_pbc": None,
                "elapsed_pbc": None,
            })
            continue

        job_id_pbc = extreure_job_id(error_files[0])
        elapsed = elapsed_per_job.get(job_id_pbc)

        elapsed_pbc.append(elapsed)
        detall_elapsed_pbc.append({
            "molecula": nom_molecula,
            "job_id_pbc": job_id_pbc,
            "elapsed_pbc": elapsed,
        })

    print("Llista de molecules:")
    print(noms_molecules)
    print("Elapsed SBC:")
    print(elapsed_sbc)
    print("Suma segons Elapsed SBC:")
    print(sum(elapsed_a_segons(elapsed) for elapsed in elapsed_sbc))
    print("Detall molecule-job-Elapsed SBC:")
    print(detall_elapsed_sbc)
    print("Molecules PBC sense errors exclosos:")
    print(molecules_pbc)
    print("Elapsed PBC:")
    print(elapsed_pbc)
    print("Suma segons Elapsed PBC:")
    print(sum(elapsed_a_segons(elapsed) for elapsed in elapsed_pbc))
    print("Detall molecule-job-Elapsed PBC:")
    print(detall_elapsed_pbc)


if __name__ == "__main__":
    main()
