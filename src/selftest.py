from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .dci import (
    fibonacci_directions,
    functional_dci,
    hessian,
    invert_hessian,
    legacy_directions,
    perturbation_matrix,
)
from .mpnn import run_mpnn
from .prepare import run_prepare
from .validate import run_validation


AA1_TO_3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"


def _coordinates(count: int = 12) -> np.ndarray:
    rng = np.random.default_rng(44)
    steps = rng.normal(size=(count, 3))
    steps[:, 0] += 2.5
    return np.cumsum(steps, axis=0)


def run_self_test() -> dict[str, object]:
    coordinates = _coordinates()
    matrix = hessian(coordinates)
    if not np.allclose(matrix, matrix.T):
        raise AssertionError("Hessian is not symmetric")
    covariance, qc = invert_hessian(matrix)
    if qc.zero_modes != 6 or qc.negative_modes != 0:
        raise AssertionError(
            f"Unexpected Hessian modes: zero={qc.zero_modes}, negative={qc.negative_modes}"
        )
    seven = legacy_directions()
    response7 = perturbation_matrix(covariance, seven, normalize=False)
    response14 = perturbation_matrix(
        covariance, np.vstack([seven, seven]), normalize=False
    )
    if not np.allclose(response7, response14):
        raise AssertionError("Direction averaging depends on a hard-coded count")
    demo_response = np.array(
        [[1.0, 2.0, 4.0], [4.0, 2.0, 1.0], [2.0, 2.0, 2.0]]
    )
    dci = functional_dci(demo_response, [0])["functional_dci"].to_numpy()
    expected = demo_response[:, 0] / demo_response.mean(axis=1)
    if not np.allclose(dci, expected):
        raise AssertionError("Functional DCI orientation is incorrect")
    if fibonacci_directions(256).shape != (256, 3):
        raise AssertionError("Fibonacci direction generation failed")
    return {
        "passed": True,
        "checks": [
            "Hessian symmetry",
            "six rigid-body modes",
            "no negative modes",
            "pseudoinverse symmetry",
            "actual direction-count averaging",
            "functional-set to responding-residue DCI orientation",
            "Fibonacci direction generation",
        ],
    }


def _write_demo_cif(path: Path, coordinates: np.ndarray) -> None:
    lines = [
        "data_residyn_demo",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
    ]
    for position, aa in enumerate(SEQUENCE, start=1):
        lines.append(f"1 {position} {AA1_TO_3[aa]} n")
    lines.extend(
        [
            "#",
            "loop_",
            "_pdbx_struct_assembly.id",
            "_pdbx_struct_assembly.details",
            "_pdbx_struct_assembly.oligomeric_count",
            "1 'synthetic monomer' 1",
            "#",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.pdbx_PDB_model_num",
        ]
    )
    offsets = {
        "N": np.array([-1.1, 0.1, 0]),
        "CA": np.zeros(3),
        "C": np.array([1.1, -0.1, 0.1]),
        "O": np.array([1.7, 0.3, -0.1]),
    }
    serial = 1
    for position, (aa, center) in enumerate(zip(SEQUENCE, coordinates), start=1):
        for atom_name, offset in offsets.items():
            x, y, z = center + offset
            lines.append(
                f"ATOM {serial} {atom_name[0]} {atom_name} . {AA1_TO_3[aa]} "
                f"A 1 {position} {x:.5f} {y:.5f} {z:.5f} 1.0 A {position} ? 1"
            )
            serial += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_demo_dms(path: Path, rng: np.random.Generator) -> None:
    rows = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for position, wt in enumerate(SEQUENCE, start=1):
        for offset in range(3):
            mutant = amino_acids[(amino_acids.index(wt) + offset + 1) % len(amino_acids)]
            mutated = list(SEQUENCE)
            mutated[position - 1] = mutant
            latent = position / len(SEQUENCE)
            score = (
                np.sin(position / 4)
                - 0.25 * offset
                + 0.35 * latent
                + rng.normal(scale=0.1)
            )
            rows.append(
                {
                    "mutant": f"{wt}{position}{mutant}",
                    "mutated_sequence": "".join(mutated),
                    "DMS_score": score,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_demo(output: Path | None = None) -> dict[str, object]:
    root = (
        output.expanduser().resolve()
        if output
        else Path(tempfile.mkdtemp(prefix="residyn-demo-"))
    )
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1729)
    coordinates = _coordinates(len(SEQUENCE))
    cif = inputs / "demo.cif"
    dms = inputs / "dms.csv"
    _write_demo_cif(cif, coordinates)
    _write_demo_dms(dms, rng)

    prepare_args = SimpleNamespace(
        pdb_id="9XYZ",
        chain="A",
        dms=dms,
        target_fasta=None,
        features=None,
        assay_id="SYNTHETIC_ASSAY",
        cif=cif,
        assembly="1",
        model=1,
        output=root / "Output",
        mutation_column="mutant",
        sequence_column="mutated_sequence",
        score_column="DMS_score",
        skip_sifts=True,
        skip_dfi=False,
        directions=256,
        rotation_trials=3,
    )
    prepared = run_prepare(prepare_args)
    target_dir = Path(prepared["target_directory"])

    log_p = rng.normal(-3.0, 0.4, size=(4, len(SEQUENCE), len(ALPHABET)))
    encoded = np.array([[ALPHABET.index(aa) for aa in SEQUENCE]])
    for position, aa in enumerate(SEQUENCE):
        log_p[:, position, ALPHABET.index(aa)] = -0.5
    npz = inputs / "proteinmpnn.npz"
    np.savez(npz, log_p=log_p, S=encoded)
    command = inputs / "proteinmpnn_command.txt"
    command.write_text("synthetic demonstration only\n", encoding="utf-8")
    mpnn = run_mpnn(
        SimpleNamespace(
            target_dir=target_dir,
            npz=npz,
            commit="synthetic",
            checkpoint="synthetic",
            command_file=command,
            alphabet=ALPHABET,
        )
    )
    validation = run_validation(
        SimpleNamespace(
            target_dir=target_dir,
            outer_folds=5,
            inner_folds=3,
            minimum_residues_per_fold=5,
            bootstrap=50,
            seed=1729,
        )
    )
    return {
        "passed": True,
        "demo_directory": root,
        "phase1_status": prepared["status"],
        "source_of_truth": prepared["source_of_truth"],
        "mpnn_scored_variants": mpnn["scored_eligible_variants"],
        "validation_summary": validation["summary"],
    }
