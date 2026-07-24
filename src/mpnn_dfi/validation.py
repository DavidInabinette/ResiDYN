from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

from .config import TargetConfig
from .io_utils import build_manifest, write_csv, write_json


class ValidationError(ValueError):
    """Raised when grouped validation cannot be performed safely."""


@dataclass(frozen=True)
class ModelSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


def choose_fold_count(
    n_groups: int,
    requested: int,
    minimum_groups_per_fold: int,
) -> int:
    possible = n_groups // minimum_groups_per_fold
    folds = min(requested, possible)
    if folds < 2:
        raise ValidationError(
            f"Need at least {2 * minimum_groups_per_fold} mapped residues; got {n_groups}"
        )
    return folds


def assign_balanced_group_folds(
    groups: Iterable[object],
    *,
    n_folds: int,
    seed: int,
) -> dict[object, int]:
    counts = pd.Series(list(groups)).value_counts()
    rng = np.random.default_rng(seed)
    items = list(counts.items())
    rng.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    fold_sizes = np.zeros(n_folds, dtype=int)
    assignment: dict[object, int] = {}
    for group, count in items:
        candidates = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(candidates))
        assignment[group] = fold
        fold_sizes[fold] += int(count)
    return assignment


def _pipeline(spec: ModelSpec) -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, list(spec.numeric)),
            ("categorical", categorical, list(spec.categorical)),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge())])


def _validate_columns(frame: pd.DataFrame, specs: dict[str, ModelSpec]) -> None:
    required = {"dms_score_z", "target_position"}
    for spec in specs.values():
        required.update(spec.numeric)
        required.update(spec.categorical)
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"Variant table lacks model columns: {sorted(missing)}")


def nested_grouped_predictions(
    frame: pd.DataFrame,
    *,
    specs: dict[str, ModelSpec],
    alphas: Iterable[float],
    outer_folds: int,
    inner_folds: int,
    minimum_residues_per_fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.loc[frame["eligible"].astype(bool)].copy().reset_index(drop=True)
    if data["dms_score_z"].isna().any():
        raise ValidationError("Eligible variants contain missing standardized outcomes")
    _validate_columns(data, specs)
    groups = data["target_position"]
    n_groups = groups.nunique()
    n_outer = choose_fold_count(n_groups, outer_folds, minimum_residues_per_fold)
    assignment = assign_balanced_group_folds(groups, n_folds=n_outer, seed=seed)
    data["outer_fold"] = groups.map(assignment).astype(int)
    tuning_rows: list[dict[str, object]] = []

    for model_name, spec in specs.items():
        prediction_column = f"prediction_{model_name.lower()}"
        data[prediction_column] = np.nan
        for fold in range(n_outer):
            test = data["outer_fold"].eq(fold)
            train = ~test
            train_groups = data.loc[train, "target_position"]
            test_groups = set(data.loc[test, "target_position"])
            if test_groups & set(train_groups):
                raise AssertionError("Residue leakage across outer folds")
            n_inner = min(inner_folds, train_groups.nunique())
            if n_inner < 2:
                raise ValidationError("Too few training residues for inner grouped CV")
            search = GridSearchCV(
                _pipeline(spec),
                param_grid={"ridge__alpha": [float(value) for value in alphas]},
                scoring="neg_mean_squared_error",
                cv=GroupKFold(n_splits=n_inner),
                refit=True,
                n_jobs=1,
            )
            feature_columns = list(spec.numeric + spec.categorical)
            search.fit(
                data.loc[train, feature_columns],
                data.loc[train, "dms_score_z"],
                groups=train_groups,
            )
            data.loc[test, prediction_column] = search.predict(
                data.loc[test, feature_columns]
            )
            tuning_rows.append(
                {
                    "model": model_name,
                    "outer_fold": fold,
                    "alpha": float(search.best_params_["ridge__alpha"]),
                    "inner_score_neg_mse": float(search.best_score_),
                    "n_train_rows": int(train.sum()),
                    "n_test_rows": int(test.sum()),
                    "n_train_residues": int(train_groups.nunique()),
                    "n_test_residues": int(len(test_groups)),
                }
            )
        if data[prediction_column].isna().any():
            raise AssertionError(f"Missing OOF predictions for {model_name}")
    return data, pd.DataFrame(tuning_rows)


def _metric_rows(oof: pd.DataFrame, model_names: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outcome = oof["dms_score_z"].to_numpy(float)
    for name in model_names:
        prediction = oof[f"prediction_{name.lower()}"].to_numpy(float)
        rows.append(
            {
                "model": name,
                "spearman": float(spearmanr(prediction, outcome).statistic),
                "mae": float(mean_absolute_error(outcome, prediction)),
                "n_variants": len(oof),
                "n_residues": int(oof["target_position"].nunique()),
            }
        )
    return rows


def paired_residue_bootstrap(
    oof: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    groups = {
        key: group.index.to_numpy()
        for key, group in oof.groupby("target_position", sort=False)
    }
    keys = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for replicate in range(replicates):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[key] for key in sampled])
        sample = oof.loc[indices]
        y = sample["dms_score_z"]
        rho_m0 = float(spearmanr(sample["prediction_m0"], y).statistic)
        rho_m1 = float(spearmanr(sample["prediction_m1"], y).statistic)
        rows.append(
            {
                "replicate": replicate + 1,
                "rho_m0": rho_m0,
                "rho_m1": rho_m1,
                "delta_rho": rho_m1 - rho_m0,
            }
        )
    return pd.DataFrame(rows)


def summarize_validation(
    oof: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, object]:
    metrics = _metric_rows(oof, ["M0", "M1"])
    by_name = {row["model"]: row for row in metrics}
    delta = by_name["M1"]["spearman"] - by_name["M0"]["spearman"]
    valid_bootstrap = bootstrap["delta_rho"].dropna()
    return {
        "models": metrics,
        "delta_spearman_m1_minus_m0": delta,
        "bootstrap": {
            "replicates": len(bootstrap),
            "valid_replicates": len(valid_bootstrap),
            "delta_rho_median": float(valid_bootstrap.median()),
            "delta_rho_ci95": [
                float(valid_bootstrap.quantile(0.025)),
                float(valid_bootstrap.quantile(0.975)),
            ],
            "fraction_delta_positive": float((valid_bootstrap > 0).mean()),
        },
    }


def run_validation_stage(config: TargetConfig) -> dict[str, Path]:
    table_path = config.output_dir / "table_b_variants.csv"
    table = pd.read_csv(table_path)
    numeric = tuple(config.require("validation.numeric_baseline"))
    categorical = tuple(config.require("validation.categorical_baseline"))
    dynamics = str(config.get("validation.dynamics_feature", "pct_dfi"))
    specs = {
        "M0": ModelSpec(numeric=numeric, categorical=categorical),
        "M1": ModelSpec(numeric=numeric + (dynamics,), categorical=categorical),
    }
    seed = int(config.get("project.random_seed", 0))
    oof, tuning = nested_grouped_predictions(
        table,
        specs=specs,
        alphas=config.get("validation.alphas", [0.01, 0.1, 1.0, 10.0, 100.0]),
        outer_folds=int(config.get("validation.outer_folds", 5)),
        inner_folds=int(config.get("validation.inner_folds", 4)),
        minimum_residues_per_fold=int(
            config.get("validation.minimum_residues_per_fold", 5)
        ),
        seed=seed,
    )
    bootstrap = paired_residue_bootstrap(
        oof,
        replicates=int(config.get("validation.bootstrap_replicates", 1000)),
        seed=seed + 1,
    )
    summary = summarize_validation(oof, bootstrap)
    outputs = {
        "oof": write_csv(oof, config.output_dir / "validation_oof.csv"),
        "tuning": write_csv(tuning, config.output_dir / "validation_tuning.csv"),
        "bootstrap": write_csv(bootstrap, config.output_dir / "validation_bootstrap.csv"),
        "summary": write_json(summary, config.output_dir / "validation_summary.json"),
    }
    manifest = build_manifest(
        stage="validation",
        config_path=config.path,
        inputs=[table_path],
        parameters={
            "models": {
                name: {"numeric": spec.numeric, "categorical": spec.categorical}
                for name, spec in specs.items()
            },
            "alphas": config.get("validation.alphas"),
            "outer_folds_requested": config.get("validation.outer_folds", 5),
            "inner_folds_requested": config.get("validation.inner_folds", 4),
            "minimum_residues_per_fold": config.get(
                "validation.minimum_residues_per_fold", 5
            ),
            "bootstrap_replicates": config.get("validation.bootstrap_replicates", 1000),
            "seed": seed,
        },
    )
    outputs["manifest"] = write_json(manifest, config.output_dir / "manifest.validation.json")
    return outputs

