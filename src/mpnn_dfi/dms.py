from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TargetConfig, resolve_path
from .io_utils import build_manifest, write_csv, write_json


SINGLE_SUBSTITUTION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class DMSError(ValueError):
    """Raised when a DMS table cannot be mapped without ambiguity."""


def parse_single_substitution(value: object) -> tuple[str, int, str] | None:
    match = SINGLE_SUBSTITUTION.fullmatch(str(value).strip().upper())
    if not match:
        return None
    wt, position, mutant = match.groups()
    if wt == mutant:
        return None
    return wt, int(position), mutant


def ingest_dms(
    dms: pd.DataFrame,
    residues: pd.DataFrame,
    *,
    target_id: str,
    assay_id: str,
    mutation_column: str,
    score_column: str,
    reverse_direction: bool = False,
) -> pd.DataFrame:
    for required in (mutation_column, score_column):
        if required not in dms:
            raise DMSError(f"DMS table lacks required column: {required}")
    parsed = dms[mutation_column].map(parse_single_substitution)
    output = pd.DataFrame(
        {
            "target_id": target_id,
            "assay_id": assay_id,
            "source_row": np.arange(len(dms)),
            "mutation": dms[mutation_column].astype(str),
            "dms_score_raw": pd.to_numeric(dms[score_column], errors="coerce"),
            "is_single_substitution": parsed.notna(),
        }
    )
    output["wt"] = parsed.map(lambda value: value[0] if value else pd.NA)
    output["dms_position"] = parsed.map(lambda value: value[1] if value else pd.NA).astype("Int64")
    output["mutant"] = parsed.map(lambda value: value[2] if value else pd.NA)
    if reverse_direction:
        output["dms_score_raw"] = -output["dms_score_raw"]

    crosswalk_columns = [
        "structure_instance_id", "label_asym_id", "label_seq_id",
        "target_position", "dms_position", "wt", "analysis_mask",
        "mapping_identity_ok",
    ]
    crosswalk = residues[crosswalk_columns].copy()
    crosswalk = crosswalk.rename(columns={"wt": "structure_wt"})
    output = output.merge(crosswalk, on="dms_position", how="left", validate="many_to_one")
    output["wt_identity_ok"] = (
        output["wt"].astype("string")
        .eq(output["structure_wt"].astype("string"))
        .fillna(False)
    )
    analysis_mask = output["analysis_mask"].astype("boolean").fillna(False).astype(bool)
    reasons = np.select(
        [
            ~output["is_single_substitution"],
            output["dms_score_raw"].isna(),
            output["structure_instance_id"].isna(),
            ~output["wt_identity_ok"].fillna(False),
            ~analysis_mask,
        ],
        [
            "not_single_substitution",
            "missing_or_non_numeric_score",
            "unmapped_dms_position",
            "wild_type_identity_mismatch",
            "outside_analysis_mask",
        ],
        default="",
    )
    output["exclusion_reason"] = reasons
    output["eligible"] = output["exclusion_reason"].eq("")
    output["dms_score_z"] = np.nan
    eligible = output["eligible"]
    scores = output.loc[eligible, "dms_score_raw"].astype(float)
    std = float(scores.std(ddof=0))
    if len(scores) and std > 0:
        output.loc[eligible, "dms_score_z"] = (scores - float(scores.mean())) / std
    elif len(scores):
        raise DMSError("Eligible DMS scores have zero variance")
    return output


def run_dms_stage(config: TargetConfig) -> dict[str, Path]:
    dms_path = resolve_path(config, config.require("dms.path"))
    residues_path = config.output_dir / "residues.csv"
    dms = pd.read_csv(dms_path)
    residues = pd.read_csv(residues_path)
    output = ingest_dms(
        dms,
        residues,
        target_id=str(config.require("project.target_id")),
        assay_id=str(config.require("dms.assay_id")),
        mutation_column=str(config.get("dms.mutation_column", "mutant")),
        score_column=str(config.get("dms.score_column", "DMS_score")),
        reverse_direction=bool(config.get("dms.reverse_direction", False)),
    )
    output_path = write_csv(output, config.output_dir / "variants.dms.csv")
    manifest = build_manifest(
        stage="dms",
        config_path=config.path,
        inputs=[dms_path, residues_path],
        parameters={
            "mutation_column": config.get("dms.mutation_column", "mutant"),
            "score_column": config.get("dms.score_column", "DMS_score"),
            "reverse_direction": config.get("dms.reverse_direction", False),
            "n_source_rows": len(output),
            "n_eligible": int(output["eligible"].sum()),
            "exclusions": output.loc[~output["eligible"], "exclusion_reason"]
            .value_counts().to_dict(),
        },
    )
    return {
        "variants": output_path,
        "manifest": write_json(manifest, config.output_dir / "manifest.dms.json"),
    }
