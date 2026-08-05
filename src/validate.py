from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ValidationError(ValueError):
    pass


RECOMMENDED_NUMERIC = [
    "mpnn_delta_logp",
    "sasa",
    "contact_number",
    "consurf_score",
    "coordinate_quality",
    "gap_adjacent",
    "ligand_distance",
]
RECOMMENDED_CATEGORICAL = ["secondary_structure", "wt", "mutant"]


def _pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    transform = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("features", transform), ("ridge", Ridge())])


def _fold_count(groups: int, requested: int, minimum: int) -> int:
    folds = min(requested, groups // minimum)
    if folds < 2:
        raise ValidationError(
            f"Need at least {2 * minimum} eligible residues; found {groups}"
        )
    return folds


def _assign_folds(groups: pd.Series, folds: int, seed: int) -> dict[object, int]:
    counts = groups.value_counts()
    items = list(counts.items())
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    sizes = np.zeros(folds, dtype=int)
    assignment = {}
    for group, count in items:
        candidates = np.flatnonzero(sizes == sizes.min())
        fold = int(rng.choice(candidates))
        assignment[group] = fold
        sizes[fold] += int(count)
    return assignment


def grouped_predictions(
    data: pd.DataFrame,
    *,
    numeric_m0: list[str],
    categorical: list[str],
    outer_folds: int,
    inner_folds: int,
    minimum_residues_per_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.loc[data["eligible"].astype(bool)].copy().reset_index(drop=True)
    required = {
        "target_position", "dms_score_z", "pct_dfi",
        *numeric_m0, *categorical,
    }
    missing = required - set(frame)
    if missing:
        raise ValidationError(f"Missing validation columns: {sorted(missing)}")
    if frame["dms_score_z"].isna().any():
        raise ValidationError("Eligible DMS outcomes contain missing standardized scores")
    n_outer = _fold_count(
        frame["target_position"].nunique(),
        outer_folds,
        minimum_residues_per_fold,
    )
    assignment = _assign_folds(
        frame["target_position"], n_outer, seed
    )
    frame["outer_fold"] = frame["target_position"].map(assignment).astype(int)
    models = {
        "m0": numeric_m0,
        "m1": numeric_m0 + ["pct_dfi"],
    }
    tuning = []
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    for name, numeric in models.items():
        frame[f"prediction_{name}"] = np.nan
        columns = numeric + categorical
        for fold in range(n_outer):
            test = frame["outer_fold"].eq(fold)
            train = ~test
            train_groups = frame.loc[train, "target_position"]
            if set(train_groups) & set(frame.loc[test, "target_position"]):
                raise AssertionError("Residue leakage across outer folds")
            n_inner = min(inner_folds, train_groups.nunique())
            if n_inner < 2:
                raise ValidationError("Too few residues for inner grouped validation")
            search = GridSearchCV(
                _pipeline(numeric, categorical),
                {"ridge__alpha": alphas},
                scoring="neg_mean_squared_error",
                cv=GroupKFold(n_splits=n_inner),
                n_jobs=1,
            )
            search.fit(
                frame.loc[train, columns],
                frame.loc[train, "dms_score_z"],
                groups=train_groups,
            )
            frame.loc[test, f"prediction_{name}"] = search.predict(
                frame.loc[test, columns]
            )
            tuning.append(
                {
                    "model": name.upper(),
                    "outer_fold": fold,
                    "alpha": float(search.best_params_["ridge__alpha"]),
                    "inner_score_neg_mse": float(search.best_score_),
                    "train_residues": int(train_groups.nunique()),
                    "test_residues": int(
                        frame.loc[test, "target_position"].nunique()
                    ),
                }
            )
    return frame, pd.DataFrame(tuning)


def _bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    groups = {
        key: group.index.to_numpy()
        for key, group in frame.groupby("target_position", sort=False)
    }
    keys = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        sample = frame.loc[indices]
        y = sample["dms_score_z"]
        rho0 = float(spearmanr(sample["prediction_m0"], y).statistic)
        rho1 = float(spearmanr(sample["prediction_m1"], y).statistic)
        rows.append(
            {
                "replicate": replicate + 1,
                "rho_m0": rho0,
                "rho_m1": rho1,
                "delta_rho": rho1 - rho0,
            }
        )
    return pd.DataFrame(rows)


def run_validation(args: object) -> dict[str, object]:
    target_dir = args.target_dir.expanduser().resolve()
    variants_path = target_dir / "tables" / "variants.csv"
    truth_path = target_dir / "tables" / "source_of_truth.csv"
    if not variants_path.is_file() or not truth_path.is_file():
        raise ValidationError("The target requires variants.csv and source_of_truth.csv")
    variants, truth = pd.read_csv(variants_path), pd.read_csv(truth_path)
    residue_features = truth.drop_duplicates("target_position")
    existing_residue_columns = [
        column for column in residue_features.columns
        if column not in variants.columns or column == "target_position"
    ]
    merged = variants.merge(
        residue_features[existing_residue_columns],
        on="target_position",
        how="left",
        validate="many_to_one",
    )
    numeric = [column for column in RECOMMENDED_NUMERIC if column in merged]
    categorical = [column for column in RECOMMENDED_CATEGORICAL if column in merged]
    if "mpnn_delta_logp" not in numeric:
        raise ValidationError("Run the ProteinMPNN import before validation")
    oof, tuning = grouped_predictions(
        merged,
        numeric_m0=numeric,
        categorical=categorical,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        minimum_residues_per_fold=args.minimum_residues_per_fold,
        seed=args.seed,
    )
    bootstrap = _bootstrap(oof, args.bootstrap, args.seed + 1)
    y = oof["dms_score_z"]
    metrics = {}
    for name in ("m0", "m1"):
        metrics[name.upper()] = {
            "spearman": float(
                spearmanr(oof[f"prediction_{name}"], y).statistic
            ),
            "mae": float(mean_absolute_error(y, oof[f"prediction_{name}"])),
        }
    delta = metrics["M1"]["spearman"] - metrics["M0"]["spearman"]
    valid = bootstrap["delta_rho"].dropna()
    summary = {
        "features": {
            "numeric_m0": numeric,
            "categorical_m0": categorical,
            "dynamics_m1": "pct_dfi",
            "recommended_but_missing": [
                column
                for column in RECOMMENDED_NUMERIC + RECOMMENDED_CATEGORICAL
                if column not in merged
            ],
        },
        "metrics": metrics,
        "delta_spearman_m1_minus_m0": delta,
        "bootstrap": {
            "replicates": len(bootstrap),
            "valid_replicates": len(valid),
            "delta_rho_ci95": [
                float(valid.quantile(0.025)),
                float(valid.quantile(0.975)),
            ],
            "fraction_delta_positive": float((valid > 0).mean()),
        },
    }
    tables = target_dir / "tables"
    oof.to_csv(tables / "validation_oof.csv", index=False)
    tuning.to_csv(tables / "validation_tuning.csv", index=False)
    bootstrap.to_csv(tables / "validation_bootstrap.csv", index=False)
    summary_path = target_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "summary": summary_path,
        "oof_predictions": tables / "validation_oof.csv",
        "delta_spearman": delta,
    }

