from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TargetConfig, resolve_path
from .io_utils import build_manifest, write_csv, write_json


class FeatureError(ValueError):
    """Raised when feature joins violate the frozen residue key contract."""


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        examples = frame.loc[frame.duplicated(columns, keep=False), columns].head().to_dict("records")
        raise FeatureError(f"{label} contains duplicate keys {columns}: {examples}")


def assemble_tables(
    residues: pd.DataFrame,
    dfi: pd.DataFrame,
    variants: pd.DataFrame,
    consurf: pd.DataFrame,
    external_covariates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residue_key = ["structure_instance_id", "label_asym_id", "label_seq_id"]
    _assert_unique(residues, residue_key, "residue table")
    _assert_unique(dfi, residue_key, "DFI table")
    _assert_unique(consurf, ["target_position"], "ConSurf table")
    _assert_unique(external_covariates, ["target_position"], "external covariates")
    dfi_identity = residues[
        residue_key + ["target_position"]
    ].merge(
        dfi[residue_key + ["target_position"]],
        on=residue_key,
        how="inner",
        validate="one_to_one",
        suffixes=("_residue", "_dfi"),
    )
    dfi_position_mismatch = (
        dfi_identity["target_position_residue"].notna()
        & dfi_identity["target_position_dfi"].notna()
        & dfi_identity["target_position_residue"].ne(
            dfi_identity["target_position_dfi"]
        )
    )
    if dfi_position_mismatch.any():
        raise FeatureError("DFI-to-residue target-position mismatch")

    table_a = residues.merge(
        dfi.drop(columns=["target_position", "dfi_serial"], errors="ignore"),
        on=residue_key,
        how="left",
        validate="one_to_one",
    )
    table_a = table_a.merge(
        consurf,
        on="target_position",
        how="left",
        validate="one_to_one",
    )
    if "consurf_wt" in table_a:
        mismatch = (
            table_a["consurf_wt"].notna()
            & table_a["wt"].notna()
            & table_a["consurf_wt"].astype(str).str.upper().ne(
                table_a["wt"].astype(str).str.upper()
            )
        )
        if mismatch.any():
            positions = table_a.loc[mismatch, "target_position"].head(20).tolist()
            raise FeatureError(f"ConSurf WT identity mismatch at positions {positions}")
    table_a["consurf_missing"] = table_a["consurf_score"].isna().astype(int)
    if "consurf_reliable" not in table_a:
        table_a["consurf_reliable"] = False
    table_a["consurf_reliable"] = table_a["consurf_reliable"].fillna(False).astype(bool)
    table_a = table_a.merge(
        external_covariates,
        on="target_position",
        how="left",
        validate="one_to_one",
        suffixes=("", "_external"),
    )
    if "wt_external" in table_a:
        mismatch = (
            table_a["wt_external"].notna()
            & table_a["wt"].notna()
            & table_a["wt_external"].astype(str).str.upper().ne(
                table_a["wt"].astype(str).str.upper()
            )
        )
        if mismatch.any():
            positions = table_a.loc[mismatch, "target_position"].head(20).tolist()
            raise FeatureError(
                f"External-covariate WT identity mismatch at positions {positions}"
            )

    feature_columns = [
        column for column in table_a.columns
        if column not in variants.columns or column in residue_key
    ]
    table_b = variants.merge(
        table_a[residue_key + [
            column for column in feature_columns if column not in residue_key
        ]],
        on=residue_key,
        how="left",
        validate="many_to_one",
        suffixes=("", "_residue"),
    )
    table_b["exclusion_reason"] = table_b["exclusion_reason"].fillna("").astype(str)
    if "wt_residue" in table_b:
        mismatch = (
            table_b["eligible"].astype(bool)
            & table_b["wt_residue"].notna()
            & table_b["wt"].ne(table_b["wt_residue"])
        )
        if mismatch.any():
            raise FeatureError("WT identity changed during residue-to-variant join")
        table_b = table_b.drop(columns=["wt_residue"])
    table_b["residue_features_present"] = table_b["pct_dfi"].notna()
    table_b.loc[
        table_b["eligible"].astype(bool) & ~table_b["residue_features_present"],
        "exclusion_reason",
    ] = "missing_residue_features"
    table_b["eligible"] = table_b["exclusion_reason"].eq("")
    return table_a, table_b


def run_assemble_stage(config: TargetConfig) -> dict[str, Path]:
    paths = {
        "residues": config.output_dir / "residues.csv",
        "dfi": config.output_dir / "dfi.csv",
        "variants": config.output_dir / "variants.mpnn.csv",
        "consurf": config.output_dir / "consurf.csv",
        "covariates": resolve_path(config, config.require("features.residue_covariates")),
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    table_a, table_b = assemble_tables(
        frames["residues"], frames["dfi"], frames["variants"],
        frames["consurf"], frames["covariates"],
    )
    outputs = {
        "table_a": write_csv(table_a, config.output_dir / "table_a_residues.csv"),
        "table_b": write_csv(table_b, config.output_dir / "table_b_variants.csv"),
    }
    manifest = build_manifest(
        stage="assemble",
        config_path=config.path,
        inputs=list(paths.values()),
        parameters={
            "n_residue_rows": len(table_a),
            "n_variant_rows": len(table_b),
            "n_eligible_variants": int(table_b["eligible"].sum()),
            "analysis_mask_unchanged": int(table_a["analysis_mask"].sum())
            == int(frames["residues"]["analysis_mask"].sum()),
        },
    )
    outputs["manifest"] = write_json(manifest, config.output_dir / "manifest.assemble.json")
    return outputs
