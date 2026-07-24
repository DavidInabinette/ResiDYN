from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TargetConfig, resolve_path
from .io_utils import build_manifest, write_csv, write_json


class ProteinMPNNError(ValueError):
    """Raised when conditional probabilities do not match the target sequence."""


def load_conditional_log_probs(
    path: str | Path,
    *,
    alphabet: str = "ACDEFGHIKLMNPQRSTVWYX",
) -> tuple[np.ndarray, str]:
    archive = np.load(path, allow_pickle=False)
    if "log_p" not in archive:
        raise ProteinMPNNError("ProteinMPNN NPZ lacks 'log_p'")
    log_p = np.asarray(archive["log_p"], dtype=float)
    if log_p.ndim < 2 or log_p.shape[-1] != len(alphabet):
        raise ProteinMPNNError(
            f"log_p must end with (length, {len(alphabet)}); got {log_p.shape}"
        )
    if log_p.ndim > 2:
        log_p = log_p.mean(axis=tuple(range(log_p.ndim - 2)))
    if "S" not in archive:
        raise ProteinMPNNError("ProteinMPNN NPZ lacks encoded WT sequence 'S'")
    encoded = np.asarray(archive["S"])
    if encoded.ndim > 1:
        encoded = encoded.reshape(-1, encoded.shape[-1])[0]
    encoded = encoded.astype(int).ravel()
    if len(encoded) != log_p.shape[0]:
        raise ProteinMPNNError(
            f"Sequence length {len(encoded)} != probability length {log_p.shape[0]}"
        )
    if np.any((encoded < 0) | (encoded >= len(alphabet))):
        raise ProteinMPNNError("Encoded sequence contains an invalid alphabet index")
    sequence = "".join(alphabet[index] for index in encoded)
    return log_p, sequence


def score_variants(
    variants: pd.DataFrame,
    log_p: np.ndarray,
    mpnn_sequence: str,
    *,
    target_sequence: str,
    alphabet: str,
) -> pd.DataFrame:
    if len(mpnn_sequence) != len(target_sequence):
        raise ProteinMPNNError(
            f"ProteinMPNN sequence length {len(mpnn_sequence)} != target length "
            f"{len(target_sequence)}"
        )
    mismatches = [
        index + 1
        for index, (observed, expected) in enumerate(zip(mpnn_sequence, target_sequence))
        if observed != expected
    ]
    if mismatches:
        raise ProteinMPNNError(
            f"ProteinMPNN WT sequence differs from target at positions {mismatches[:20]}"
        )
    alphabet_index = {aa: index for index, aa in enumerate(alphabet)}
    output = variants.copy()
    output["mpnn_wt_logp"] = np.nan
    output["mpnn_mutant_logp"] = np.nan
    output["mpnn_delta_logp"] = np.nan
    for index, row in output.loc[output["eligible"].astype(bool)].iterrows():
        position = int(row["target_position"]) - 1
        wt, mutant = str(row["wt"]), str(row["mutant"])
        if wt not in alphabet_index or mutant not in alphabet_index:
            raise ProteinMPNNError(f"Unsupported amino acid in {wt}{position + 1}{mutant}")
        if mpnn_sequence[position] != wt:
            raise ProteinMPNNError(
                f"Variant WT mismatch at target position {position + 1}: "
                f"{wt} versus {mpnn_sequence[position]}"
            )
        wt_logp = float(log_p[position, alphabet_index[wt]])
        mutant_logp = float(log_p[position, alphabet_index[mutant]])
        output.loc[index, ["mpnn_wt_logp", "mpnn_mutant_logp", "mpnn_delta_logp"]] = (
            wt_logp, mutant_logp, mutant_logp - wt_logp
        )
    return output


def run_proteinmpnn_stage(config: TargetConfig) -> dict[str, Path]:
    variants_path = config.output_dir / "variants.dms.csv"
    variants = pd.read_csv(variants_path)
    npz_path = resolve_path(config, config.require("proteinmpnn.conditional_npz"))
    alphabet = str(config.get("proteinmpnn.alphabet", "ACDEFGHIKLMNPQRSTVWYX"))
    log_p, sequence = load_conditional_log_probs(npz_path, alphabet=alphabet)
    target_sequence = str(config.require("structure.target_sequence")).upper()
    scored = score_variants(
        variants, log_p, sequence,
        target_sequence=target_sequence, alphabet=alphabet,
    )
    output = write_csv(scored, config.output_dir / "variants.mpnn.csv")
    provenance = {
        "repository_commit": config.require("proteinmpnn.repository_commit"),
        "checkpoint": config.require("proteinmpnn.checkpoint"),
        "backbone_noise": config.get("proteinmpnn.backbone_noise", 0.0),
        "seed": config.get("proteinmpnn.seed", config.get("project.random_seed", 0)),
        "command": config.require("proteinmpnn.command"),
        "alphabet": alphabet,
        "sequence": sequence,
        "log_p_shape_after_averaging": list(log_p.shape),
        "training_exposure": config.get("proteinmpnn.training_exposure", {}),
    }
    manifest = build_manifest(
        stage="proteinmpnn",
        config_path=config.path,
        inputs=[variants_path, npz_path, config.output_dir / "proteinmpnn_input.pdb"],
        parameters=provenance,
    )
    return {
        "variants": output,
        "manifest": write_json(manifest, config.output_dir / "manifest.proteinmpnn.json"),
    }
