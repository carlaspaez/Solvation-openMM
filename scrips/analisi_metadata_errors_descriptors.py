import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

ROOT = Path(__file__).resolve().parent.parent

DATABASE_SMILES = ROOT / "DADES" / "v0.31" / "database.txt"
PBC_ERRORS = ROOT / "outputs_PBC2" / "metadata_error_PBC.json"
SBC_ERRORS = ROOT / "outputs_SBC2" / "metadata_error_SBC.json"
OUT_CSV = ROOT / "resultats" / "descriptors_metadata_errors.csv"
OUT_JSON = ROOT / "resultats" / "descriptors_metadata_errors.json"


def carregar_database_smiles():
    molecules = {}
    with DATABASE_SMILES.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(";")]
            if len(parts) < 3:
                continue

            mol_id, smiles, name = parts[:3]
            molecules[mol_id] = {
                "molecula": mol_id,
                "smiles": smiles,
                "nom": name,
            }
    return molecules


def carregar_fallides(path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    return {
        entry["molecula"]
        for entry in data
        if entry.get("molecula") and not entry.get("te_results_json", False)
    }


def descriptors_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = mol.GetAtoms()
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in atoms)
    hetero_atoms = sum(1 for atom in atoms if atom.GetAtomicNum() not in (1, 6))
    halogens = sum(1 for atom in atoms if atom.GetAtomicNum() in (9, 17, 35, 53))

    return {
        "pes_molecular": Descriptors.MolWt(mol),
        "atoms_pesants": heavy_atoms,
        "anells": Lipinski.RingCount(mol),
        "anells_aromatics": Lipinski.NumAromaticRings(mol),
        "atoms_aromatics": aromatic_atoms,
        "heteroatoms": hetero_atoms,
        "halogens": halogens,
        "donadors_h": Lipinski.NumHDonors(mol),
        "acceptors_h": Lipinski.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "enllacos_rotables": Lipinski.NumRotatableBonds(mol),
        "logp": Descriptors.MolLogP(mol),
        "es_aromatica": Lipinski.NumAromaticRings(mol) > 0,
        "es_halogenada": halogens > 0,
        "gran_pes_molecular_gt_200": Descriptors.MolWt(mol) > 200,
        "polar_tpsa_gt_60": Descriptors.TPSA(mol) > 60,
    }


def main():
    molecules = carregar_database_smiles()
    fallides_pbc = carregar_fallides(PBC_ERRORS)
    fallides_sbc = carregar_fallides(SBC_ERRORS)

    rows = []
    for mol_id in sorted(molecules):
        row = molecules[mol_id].copy()
        desc = descriptors_rdkit(row["smiles"])
        if desc is None:
            continue

        row.update(desc)
        row["fallida_pbc"] = mol_id in fallides_pbc
        row["fallida_sbc"] = mol_id in fallides_sbc
        row["fallida_qualsevol"] = row["fallida_pbc"] or row["fallida_sbc"]
        row["fallida_comuna_pbc_sbc"] = row["fallida_pbc"] and row["fallida_sbc"]
        rows.append(row)

    fieldnames = [
        "molecula",
        "nom",
        "smiles",
        "fallida_pbc",
        "fallida_sbc",
        "fallida_qualsevol",
        "fallida_comuna_pbc_sbc",
        "pes_molecular",
        "atoms_pesants",
        "anells",
        "anells_aromatics",
        "atoms_aromatics",
        "heteroatoms",
        "halogens",
        "donadors_h",
        "acceptors_h",
        "tpsa",
        "enllacos_rotables",
        "logp",
        "es_aromatica",
        "es_halogenada",
        "gran_pes_molecular_gt_200",
        "polar_tpsa_gt_60",
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("Fitxer CSV creat:", OUT_CSV)
    print("Fitxer JSON creat:", OUT_JSON)
    print("Molecules totals:", len(rows))
    print("Fallides PBC:", sum(row["fallida_pbc"] for row in rows))
    print("Fallides SBC:", sum(row["fallida_sbc"] for row in rows))
    print("Fallides comunes:", sum(row["fallida_comuna_pbc_sbc"] for row in rows))


if __name__ == "__main__":
    main()
