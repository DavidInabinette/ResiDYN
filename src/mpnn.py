from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import sha256


class MPNNError(ValueError):
    pass


def _target_sequence(path: Path) -> str:
    current = None
    sequences: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            current = line[1:].split("|", 1)[0]
            sequences[current] = []
        elif current and line.strip():
            sequences[current].append(line.strip())
    if "target" not in sequences:
        raise MPNNError(f"No target sequence in {path}")
    return "".join(sequences["target"]).upper()


def load_conditional(
    path: Path, alphabet: str
) -> tuple[np.ndarray, str]:
    archive = np.load(path, allow_pickle=False)
    if "log_p" not in archive or "S" not in archive:
        raise MPNNError("ProteinMPNN NPZ requires log_p and S arrays")
    log_p = np.asarray(archive["log_p"], dtype=float)
    if log_p.ndim < 2 or log_p.shape[-1] != len(alphabet):
        raise MPNNError(
            f"log_p must end in (length, {len(alphabet)}); got {log_p.shape}"
        )
    if log_p.ndim > 2:
        log_p = log_p.mean(axis=tuple(range(log_p.ndim - 2)))
    encoded = np.asarray(archive["S"])
    if encoded.ndim > 1:
        encoded = encoded.reshape(-1, encoded.shape[-1])[0]
    encoded = encoded.astype(int).ravel()
    if len(encoded) != log_p.shape[0]:
        raise MPNNError("ProteinMPNN sequence and log-probability lengths differ")
    if np.any((encoded < 0) | (encoded >= len(alphabet))):
        raise MPNNError("ProteinMPNN sequence contains an invalid alphabet index")
    sequence = "".join(alphabet[index] for index in encoded)
    return log_p, sequence


def annotate_variants(
    variants: pd.DataFrame,
    *,
    log_p: np.ndarray,
    sequence: str,
    target_sequence: str,
    alphabet: str,
) -> pd.DataFrame:
    if sequence != target_sequence:
        mismatches = [
            index + 1
            for index, (observed, expected) in enumerate(
                zip(sequence, target_sequence)
            )
            if observed != expected
        ]
        if len(sequence) != len(target_sequence):
            raise MPNNError(
                f"ProteinMPNN length {len(sequence)} differs from target "
                f"length {len(target_sequence)}"
            )
        raise MPNNError(
            f"ProteinMPNN sequence differs from the target at {mismatches[:20]}"
        )
    lookup = {aa: index for index, aa in enumerate(alphabet)}
    output = variants.copy()
    output["mpnn_wt_logp"] = np.nan
    output["mpnn_mutant_logp"] = np.nan
    output["mpnn_delta_logp"] = np.nan
    for row_index, row in output.loc[output["eligible"].astype(bool)].iterrows():
        position = int(row["target_position"]) - 1
        wt, mutant = str(row["wt"]), str(row["mutant"])
        if wt not in lookup or mutant not in lookup:
            raise MPNNError(f"Unsupported substitution: {wt}{position + 1}{mutant}")
        if sequence[position] != wt:
            raise MPNNError(
                f"Variant WT mismatch at {position + 1}: {wt} versus "
                f"{sequence[position]}"
            )
        wt_logp = float(log_p[position, lookup[wt]])
        mutant_logp = float(log_p[position, lookup[mutant]])
        output.loc[
            row_index,
            ["mpnn_wt_logp", "mpnn_mutant_logp", "mpnn_delta_logp"],
        ] = (wt_logp, mutant_logp, mutant_logp - wt_logp)
    return output


def run_mpnn(args: object) -> dict[str, object]:
    target_dir = args.target_dir.expanduser().resolve()
    variants_path = target_dir / "tables" / "variants.csv"
    sequence_path = target_dir / "sequences.fasta"
    if not variants_path.is_file():
        raise MPNNError(
            "No variants.csv exists. Re-run prepare with a DMS input first."
        )
    for path in (args.npz, args.command_file):
        if not path.expanduser().is_file():
            raise MPNNError(f"Required file does not exist: {path}")
    target = _target_sequence(sequence_path)
    log_p, sequence = load_conditional(args.npz.expanduser(), args.alphabet)
    variants = pd.read_csv(variants_path)
    scored = annotate_variants(
        variants,
        log_p=log_p,
        sequence=sequence,
        target_sequence=target,
        alphabet=args.alphabet,
    )
    scored.to_csv(variants_path, index=False)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditional_npz": str(args.npz.resolve()),
        "conditional_npz_sha256": sha256(args.npz.resolve()),
        "repository_commit": args.commit,
        "checkpoint": args.checkpoint,
        "alphabet": args.alphabet,
        "command": args.command_file.read_text(encoding="utf-8").strip(),
        "sequence_length": len(sequence),
        "scored_eligible_variants": int(
            scored.loc[scored["eligible"].astype(bool), "mpnn_delta_logp"]
            .notna().sum()
        ),
    }
    path = target_dir / "mpnn_provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        "variants": variants_path,
        "provenance": path,
        "scored_eligible_variants": provenance["scored_eligible_variants"],
    }
