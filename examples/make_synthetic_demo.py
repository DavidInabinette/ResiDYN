"""Create a deterministic, non-biological dataset for an end-to-end smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
AA1_TO_3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def write_demo(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    centers = np.cumsum(rng.normal(size=(len(SEQUENCE), 3)) + [2.4, 0.0, 0.0], axis=0)

    pdb_lines: list[str] = []
    serial = 1
    atom_offsets = {
        "N": np.array([-1.1, 0.1, 0.0]),
        "CA": np.zeros(3),
        "C": np.array([1.1, -0.1, 0.1]),
        "O": np.array([1.7, 0.3, -0.1]),
    }
    for residue_number, (aa, center) in enumerate(zip(SEQUENCE, centers), start=1):
        for atom_name, offset in atom_offsets.items():
            x, y, z = center + offset
            element = atom_name[0]
            pdb_lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {AA1_TO_3[aa]:>3s} "
                f"A{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.0:6.2f}{20.0:6.2f}          {element:>2s}"
            )
            serial += 1
    (inputs / "demo.pdb").write_text(
        "\n".join(pdb_lines + ["TER", "END"]) + "\n", encoding="utf-8"
    )

    dms_rows = []
    mutants = "VIL"
    for position, wt in enumerate(SEQUENCE, start=1):
        for offset, mutant in enumerate(mutants):
            if mutant == wt:
                mutant = "A" if wt != "A" else "G"
            latent_dfi = position / len(SEQUENCE)
            mpnn_like = -0.3 * offset + np.sin(position / 3)
            score = 0.8 * mpnn_like + 0.5 * latent_dfi + rng.normal(scale=0.15)
            dms_rows.append({"mutant": f"{wt}{position}{mutant}", "DMS_score": score})
    pd.DataFrame(dms_rows).to_csv(inputs / "dms.csv", index=False)

    log_p = rng.normal(-3.0, 0.5, size=(4, len(SEQUENCE), len(ALPHABET)))
    encoded = np.array([[ALPHABET.index(aa) for aa in SEQUENCE]])
    for position, aa in enumerate(SEQUENCE):
        log_p[:, position, ALPHABET.index(aa)] = -0.5
    np.savez(inputs / "proteinmpnn.npz", log_p=log_p, S=encoded)

    grades = [
        "# POS SEQ SCORE COLOR",
        *[
            f"{position} {aa} {np.cos(position / 5):.5f} "
            f"{1 + (position % 9)}"
            for position, aa in enumerate(SEQUENCE, start=1)
        ],
    ]
    (inputs / "consurf.grades").write_text("\n".join(grades) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "target_position": np.arange(1, len(SEQUENCE) + 1),
            "sasa": np.linspace(10.0, 90.0, len(SEQUENCE)),
            "secondary_structure": np.resize(np.array(["H", "E", "C"]), len(SEQUENCE)),
        }
    ).to_csv(inputs / "covariates.csv", index=False)

    config = {
        "project": {
            "target_id": "SYNTHETIC",
            "structure_instance_id": "SYNTHETIC_PDB",
            "output_dir": "outputs",
            "random_seed": 1729,
        },
        "structure": {
            "path": "inputs/demo.pdb",
            "format": "pdb",
            "label_asym_id": "A",
            "target_sequence": SEQUENCE,
            "assembly_id": "synthetic_single_chain",
            "biological_assembly_verified": True,
            "source_model_number": 1,
            "dms_position_offset": 0,
            "contact_cutoff_angstrom": 8.0,
        },
        "dms": {
            "path": "inputs/dms.csv",
            "assay_id": "SYNTHETIC_ASSAY",
            "mutation_column": "mutant",
            "score_column": "DMS_score",
            "reverse_direction": False,
        },
        "dfi": {
            "spring_law": "d6",
            "gamma": 100.0,
            "cutoff_angstrom": None,
            "direction_mode": "fibonacci",
            "n_directions": 128,
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "rotation_trials": 2,
            "rotation_min_spearman": 0.95,
            "rotation_max_quartile_change": 0.2,
            "require_qc_pass": True,
            "functional_target_positions": [5, 10],
        },
        "proteinmpnn": {
            "conditional_npz": "inputs/proteinmpnn.npz",
            "alphabet": ALPHABET,
            "repository_commit": "synthetic-test",
            "checkpoint": "synthetic-test",
            "backbone_noise": 0.0,
            "seed": 1729,
            "command": "synthetic-test; no external command",
        },
        "consurf": {"path": "inputs/consurf.grades", "format": "grades"},
        "features": {"residue_covariates": "inputs/covariates.csv"},
        "validation": {
            "outer_folds": 5,
            "inner_folds": 3,
            "minimum_residues_per_fold": 5,
            "alphas": [0.1, 1.0, 10.0],
            "bootstrap_replicates": 50,
            "numeric_baseline": [
                "mpnn_delta_logp", "sasa", "contact_number",
                "consurf_score", "coordinate_quality", "gap_adjacent",
            ],
            "categorical_baseline": ["secondary_structure", "wt", "mutant"],
            "dynamics_feature": "pct_dfi",
        },
    }
    config_path = root / "target.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(write_demo(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
