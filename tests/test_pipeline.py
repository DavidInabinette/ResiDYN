from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mpnn_dfi.dfi import (  # noqa: E402
    build_hessian,
    calculate_dfi,
    fibonacci_directions,
    functional_dci,
    legacy_directions,
    perturbation_matrix,
    pseudoinverse_hessian,
    rotation_qc,
)
from mpnn_dfi.dms import ingest_dms, parse_single_substitution  # noqa: E402
from mpnn_dfi.proteinmpnn import (  # noqa: E402
    ProteinMPNNError,
    load_conditional_log_probs,
    score_variants,
)
from mpnn_dfi.structure import (  # noqa: E402
    build_residue_table,
    parse_mmcif_atom_site,
)
from mpnn_dfi.validation import (  # noqa: E402
    ModelSpec,
    nested_grouped_predictions,
)


def synthetic_coordinates(n: int = 12) -> np.ndarray:
    rng = np.random.default_rng(44)
    steps = rng.normal(size=(n, 3))
    steps[:, 0] += 2.5
    return np.cumsum(steps, axis=0)


class DFITests(unittest.TestCase):
    def test_hessian_has_six_rigid_body_modes_and_symmetric_inverse(self) -> None:
        coordinates = synthetic_coordinates()
        hessian = build_hessian(coordinates, gamma=100.0, spring_law="d6")
        self.assertTrue(np.allclose(hessian, hessian.T))
        covariance, qc = pseudoinverse_hessian(
            hessian, absolute_tolerance=1e-12, relative_tolerance=1e-10
        )
        self.assertEqual(qc.zero_mode_count, 6)
        self.assertLess(np.max(np.abs(covariance - covariance.T)), 1e-8)

    def test_direction_average_uses_actual_direction_count(self) -> None:
        coordinates = synthetic_coordinates(9)
        covariance, _ = pseudoinverse_hessian(build_hessian(coordinates))
        seven = legacy_directions()
        response_seven = perturbation_matrix(covariance, seven, normalize=False)
        response_fourteen = perturbation_matrix(
            covariance, np.vstack([seven, seven]), normalize=False
        )
        np.testing.assert_allclose(response_seven, response_fourteen, rtol=1e-12, atol=1e-12)

    def test_functional_dci_is_functional_set_to_response(self) -> None:
        response = np.array(
            [
                [1.0, 2.0, 4.0],
                [4.0, 2.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        result = functional_dci(response, [0])
        expected = response[:, 0] / response.mean(axis=1)
        np.testing.assert_allclose(result["functional_dci"], expected)

    def test_fibonacci_dfi_is_rotation_stable(self) -> None:
        coordinates = synthetic_coordinates(10)
        directions = fibonacci_directions(512)
        hessian_kwargs = {
            "gamma": 100.0,
            "spring_law": "d6",
            "cutoff_angstrom": None,
        }
        inverse_kwargs = {
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-10,
            "expected_zero_modes": 6,
        }
        covariance, _ = pseudoinverse_hessian(
            build_hessian(coordinates, **hessian_kwargs), **inverse_kwargs
        )
        reference = calculate_dfi(
            perturbation_matrix(covariance, directions)
        )["pct_dfi"].to_numpy()
        qc = rotation_qc(
            coordinates,
            reference,
            trials=3,
            seed=7,
            directions=directions,
            hessian_kwargs=hessian_kwargs,
            inverse_kwargs=inverse_kwargs,
        )
        self.assertGreaterEqual(qc["minimum_spearman"], 0.98)


class StructureTests(unittest.TestCase):
    def test_mmcif_altloc_mapping_and_masks(self) -> None:
        cif = """data_demo
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
ATOM 1 N N . ALA A 1 0 0 0 1 A 10 ?
ATOM 2 C CA A ALA A 1 1 0 0 0.50 A 10 ?
ATOM 3 C CA B ALA A 1 9 0 0 0.50 A 10 ?
ATOM 4 C C . ALA A 1 2 0 0 1 A 10 ?
ATOM 5 O O . ALA A 1 3 0 0 1 A 10 ?
ATOM 6 N N . GLY A 2 4 0 0 1 A 11 ?
ATOM 7 C CA . GLY A 2 5 0 0 1 A 11 ?
ATOM 8 C C . GLY A 2 6 0 0 1 A 11 ?
ATOM 9 O O . GLY A 2 7 0 0 1 A 11 ?
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.cif"
            path.write_text(cif, encoding="utf-8")
            atoms = parse_mmcif_atom_site(path)
        table = build_residue_table(
            atoms,
            structure_instance_id="demo",
            label_asym_id="A",
            target_sequence="AG",
        )
        self.assertEqual(len(table), 2)
        self.assertTrue(table["analysis_mask"].all())
        self.assertEqual(float(table.loc[0, "ca_x"]), 1.0)
        self.assertEqual(table["target_position"].tolist(), [1, 2])
        self.assertEqual(table["dfi_serial"].tolist(), [1, 2])


class IngestionTests(unittest.TestCase):
    def residue_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "structure_instance_id": ["x", "x"],
                "label_asym_id": ["A", "A"],
                "label_seq_id": [10, 11],
                "target_position": [1, 2],
                "dms_position": [1, 2],
                "wt": ["A", "G"],
                "analysis_mask": [True, True],
                "mapping_identity_ok": [True, True],
            }
        )

    def test_single_mutation_parser_and_identity_validation(self) -> None:
        self.assertEqual(parse_single_substitution("A1V"), ("A", 1, "V"))
        self.assertIsNone(parse_single_substitution("A1V:G2D"))
        dms = pd.DataFrame(
            {
                "mutant": ["A1V", "X2D", "A1V:G2D", "G2D"],
                "DMS_score": [1.0, 0.0, -1.0, 0.2],
            }
        )
        result = ingest_dms(
            dms,
            self.residue_table(),
            target_id="demo",
            assay_id="assay",
            mutation_column="mutant",
            score_column="DMS_score",
        )
        self.assertEqual(result["eligible"].tolist(), [True, False, False, True])
        self.assertEqual(result.loc[1, "exclusion_reason"], "wild_type_identity_mismatch")

    def test_proteinmpnn_import_and_delta_logp(self) -> None:
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        sequence = "AG"
        encoded = np.array([[alphabet.index(aa) for aa in sequence]])
        log_p = np.full((2, 2, len(alphabet)), -10.0)
        log_p[:, 0, alphabet.index("A")] = -1.0
        log_p[:, 0, alphabet.index("V")] = -2.5
        log_p[:, 1, alphabet.index("G")] = -0.5
        log_p[:, 1, alphabet.index("D")] = -1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            np.savez(path, log_p=log_p, S=encoded)
            loaded, loaded_sequence = load_conditional_log_probs(path, alphabet=alphabet)
        variants = ingest_dms(
            pd.DataFrame({"mutant": ["A1V", "G2D"], "DMS_score": [1.0, 0.0]}),
            self.residue_table(),
            target_id="demo",
            assay_id="assay",
            mutation_column="mutant",
            score_column="DMS_score",
        )
        scored = score_variants(
            variants, loaded, loaded_sequence,
            target_sequence=sequence, alphabet=alphabet,
        )
        self.assertAlmostEqual(float(scored.loc[0, "mpnn_delta_logp"]), -1.5)
        with self.assertRaises(ProteinMPNNError):
            score_variants(
                variants, loaded, loaded_sequence,
                target_sequence="AA", alphabet=alphabet,
            )


class ValidationTests(unittest.TestCase):
    def test_nested_cv_keeps_residues_together(self) -> None:
        rng = np.random.default_rng(3)
        rows = []
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        for position in range(1, 21):
            pct_dfi = position / 20
            for variant_index in range(3):
                mpnn = rng.normal()
                mutant = amino_acids[(position + variant_index) % len(amino_acids)]
                rows.append(
                    {
                        "eligible": True,
                        "target_position": position,
                        "dms_score_z": 0.7 * mpnn + 0.4 * pct_dfi + rng.normal(scale=0.1),
                        "mpnn_delta_logp": mpnn,
                        "contact_number": float(position % 7),
                        "pct_dfi": pct_dfi,
                        "wt": amino_acids[position % len(amino_acids)],
                        "mutant": mutant,
                    }
                )
        frame = pd.DataFrame(rows)
        specs = {
            "M0": ModelSpec(("mpnn_delta_logp", "contact_number"), ("wt", "mutant")),
            "M1": ModelSpec(
                ("mpnn_delta_logp", "contact_number", "pct_dfi"), ("wt", "mutant")
            ),
        }
        oof, tuning = nested_grouped_predictions(
            frame,
            specs=specs,
            alphas=[0.1, 1.0, 10.0],
            outer_folds=5,
            inner_folds=3,
            minimum_residues_per_fold=3,
            seed=19,
        )
        self.assertFalse(oof[["prediction_m0", "prediction_m1"]].isna().any().any())
        self.assertTrue((oof.groupby("target_position")["outer_fold"].nunique() == 1).all())
        self.assertEqual(len(tuning), 10)


if __name__ == "__main__":
    unittest.main()
