from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .config import TargetConfig
from .io_utils import build_manifest, write_csv, write_json


class DFIError(ValueError):
    """Raised when the ENM/DFI calculation fails a structural invariant."""


@dataclass(frozen=True)
class SpectralQC:
    singular_values_ascending: np.ndarray
    inversion_threshold: float
    zero_mode_count: int
    expected_zero_modes: int
    pseudoinverse_symmetry_error: float
    negative_mode_count: int

    @property
    def passed(self) -> bool:
        return (
            self.zero_mode_count == self.expected_zero_modes
            and self.negative_mode_count == 0
            and self.pseudoinverse_symmetry_error < 1e-8
        )


def build_hessian(
    coordinates: np.ndarray,
    *,
    gamma: float = 100.0,
    spring_law: str = "d6",
    cutoff_angstrom: float | None = None,
) -> np.ndarray:
    """Build a fully connected anisotropic-network Hessian.

    `d6` reproduces the supplied implementation's effective spring weight:
    `gamma**3 / d**6`, where its variable `r` was squared distance.
    """

    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise DFIError("coordinates must have shape (n_residues, 3)")
    if len(xyz) < 3:
        raise DFIError("At least three residues are required")
    if not np.isfinite(xyz).all():
        raise DFIError("coordinates contain non-finite values")
    if spring_law not in {"d6", "uniform"}:
        raise DFIError(f"Unsupported spring law: {spring_law}")
    hessian = np.zeros((3 * len(xyz), 3 * len(xyz)), dtype=float)
    for i in range(len(xyz) - 1):
        for j in range(i + 1, len(xyz)):
            delta = xyz[i] - xyz[j]
            squared_distance = float(delta @ delta)
            if squared_distance <= 0:
                raise DFIError(f"Duplicate C-alpha coordinates at indices {i} and {j}")
            distance = np.sqrt(squared_distance)
            if cutoff_angstrom is not None and distance > cutoff_angstrom:
                continue
            spring = gamma ** 3 / squared_distance ** 3 if spring_law == "d6" else gamma
            block = spring * np.outer(delta, delta) / squared_distance
            i_slice = slice(3 * i, 3 * i + 3)
            j_slice = slice(3 * j, 3 * j + 3)
            hessian[i_slice, i_slice] += block
            hessian[j_slice, j_slice] += block
            hessian[i_slice, j_slice] -= block
            hessian[j_slice, i_slice] -= block
    return hessian


def pseudoinverse_hessian(
    hessian: np.ndarray,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
    expected_zero_modes: int = 6,
) -> tuple[np.ndarray, SpectralQC]:
    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise DFIError("Hessian must be square")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * scale)
    positive = eigenvalues > threshold
    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[positive] = 1.0 / eigenvalues[positive]
    covariance = (eigenvectors * inverse_values) @ eigenvectors.T
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    qc = SpectralQC(
        singular_values_ascending=np.sort(np.abs(eigenvalues)),
        inversion_threshold=threshold,
        zero_mode_count=int((np.abs(eigenvalues) <= threshold).sum()),
        expected_zero_modes=expected_zero_modes,
        pseudoinverse_symmetry_error=symmetry_error,
        negative_mode_count=int((eigenvalues < -threshold).sum()),
    )
    return covariance, qc


def legacy_directions() -> np.ndarray:
    directions = np.array(
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        dtype=float,
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def fibonacci_directions(n_directions: int) -> np.ndarray:
    if n_directions < 6:
        raise DFIError("At least six directions are required")
    indices = np.arange(n_directions, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / n_directions
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))


def perturbation_matrix(
    covariance: np.ndarray,
    directions: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if covariance.shape[0] % 3:
        raise DFIError("Covariance dimension must be divisible by three")
    n_residues = covariance.shape[0] // 3
    if covariance.shape != (3 * n_residues, 3 * n_residues):
        raise DFIError("Covariance is not a square 3N x 3N matrix")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise DFIError("directions must have shape (n_directions, 3)")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0):
        raise DFIError("Perturbation directions may not be zero")
    directions = directions / norms[:, None]
    response = np.zeros((n_residues, n_residues), dtype=float)
    for perturbed in range(n_residues):
        force = np.zeros((3 * n_residues, len(directions)), dtype=float)
        force[3 * perturbed:3 * perturbed + 3, :] = directions.T
        displacement = covariance @ force
        magnitudes = np.linalg.norm(
            displacement.reshape(n_residues, 3, len(directions)), axis=1
        )
        response[:, perturbed] = magnitudes.mean(axis=1)
    if normalize:
        total = float(response.sum())
        if total <= 0:
            raise DFIError("Perturbation response has non-positive total")
        response /= total
    return response


def percentile_rank(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return rankdata(array, method="max") / len(array)


def calculate_dfi(response: np.ndarray) -> pd.DataFrame:
    raw = np.asarray(response, dtype=float).sum(axis=1)
    mean = float(raw.mean())
    std = float(raw.std(ddof=0))
    return pd.DataFrame(
        {
            "dfi": raw,
            "relative_dfi": raw / mean if mean else np.nan,
            "pct_dfi": percentile_rank(raw),
            "z_dfi": (raw - mean) / std if std else 0.0,
        }
    )


def functional_dci(response: np.ndarray, functional_indices: Iterable[int]) -> pd.DataFrame:
    """Calculate functional-set-to-responding-residue DCI.

    Rows of `response` are responding residues and columns are perturbed
    residues. `functional_indices` are zero-based perturbation columns.
    """

    matrix = np.asarray(response, dtype=float)
    indices = np.array(sorted(set(int(value) for value in functional_indices)), dtype=int)
    if len(indices) == 0:
        raise DFIError("At least one functional residue is required for DCI")
    if indices.min() < 0 or indices.max() >= matrix.shape[1]:
        raise DFIError("Functional DCI index is outside the perturbation matrix")
    numerator = matrix[:, indices].mean(axis=1)
    denominator = matrix.mean(axis=1)
    dci = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    return pd.DataFrame({"functional_dci": dci, "pct_functional_dci": percentile_rank(dci)})


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _quartiles(values: np.ndarray) -> np.ndarray:
    order = rankdata(values, method="average") / len(values)
    return np.minimum((order * 4).astype(int), 3)


def rotation_qc(
    coordinates: np.ndarray,
    reference_pct_dfi: np.ndarray,
    *,
    trials: int,
    seed: int,
    directions: np.ndarray,
    hessian_kwargs: dict[str, object],
    inverse_kwargs: dict[str, object],
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    trials_out: list[dict[str, float]] = []
    reference_quartiles = _quartiles(reference_pct_dfi)
    for trial in range(trials):
        rotation = random_rotation(rng)
        rotated = np.asarray(coordinates) @ rotation.T
        hessian = build_hessian(rotated, **hessian_kwargs)
        covariance, _ = pseudoinverse_hessian(hessian, **inverse_kwargs)
        response = perturbation_matrix(covariance, directions)
        pct = calculate_dfi(response)["pct_dfi"].to_numpy(float)
        rho = float(spearmanr(reference_pct_dfi, pct).statistic)
        change = float(np.mean(reference_quartiles != _quartiles(pct)))
        trials_out.append(
            {"trial": trial + 1, "spearman": rho, "quartile_change_fraction": change}
        )
    return {
        "trials": trials_out,
        "minimum_spearman": min(item["spearman"] for item in trials_out) if trials_out else 1.0,
        "maximum_quartile_change_fraction": (
            max(item["quartile_change_fraction"] for item in trials_out)
            if trials_out else 0.0
        ),
    }


def run_dfi_stage(config: TargetConfig) -> dict[str, Path]:
    residues_path = config.output_dir / "residues.csv"
    residues = pd.read_csv(residues_path)
    context = residues.loc[residues["dfi_context_mask"].astype(bool)].copy()
    context = context.sort_values("dfi_serial")
    coordinates = context[["ca_x", "ca_y", "ca_z"]].to_numpy(float)

    cutoff = config.get("dfi.cutoff_angstrom")
    hessian_kwargs = {
        "gamma": float(config.get("dfi.gamma", 100.0)),
        "spring_law": str(config.get("dfi.spring_law", "d6")),
        "cutoff_angstrom": None if cutoff is None else float(cutoff),
    }
    inverse_kwargs = {
        "absolute_tolerance": float(config.get("dfi.absolute_tolerance", 1e-10)),
        "relative_tolerance": float(config.get("dfi.relative_tolerance", 1e-12)),
        "expected_zero_modes": 6,
    }
    mode = str(config.get("dfi.direction_mode", "fibonacci"))
    directions = (
        legacy_directions()
        if mode == "legacy7"
        else fibonacci_directions(int(config.get("dfi.n_directions", 256)))
    )
    hessian = build_hessian(coordinates, **hessian_kwargs)
    covariance, spectral = pseudoinverse_hessian(hessian, **inverse_kwargs)
    response = perturbation_matrix(covariance, directions)
    dfi = calculate_dfi(response)
    for column in dfi.columns:
        context[column] = dfi[column].to_numpy()

    functional_positions = {
        int(value) for value in config.get("dfi.functional_target_positions", [])
    }
    if functional_positions:
        functional_indices = [
            index
            for index, position in enumerate(context["target_position"])
            if pd.notna(position) and int(position) in functional_positions
        ]
        if len(functional_indices) != len(functional_positions):
            found = {
                int(context.iloc[index]["target_position"]) for index in functional_indices
            }
            raise DFIError(
                f"Functional positions missing from DFI context: "
                f"{sorted(functional_positions - found)}"
            )
        dci = functional_dci(response, functional_indices)
        for column in dci.columns:
            context[column] = dci[column].to_numpy()

    rotation = rotation_qc(
        coordinates,
        context["pct_dfi"].to_numpy(float),
        trials=int(config.get("dfi.rotation_trials", 8)),
        seed=int(config.get("project.random_seed", 0)),
        directions=directions,
        hessian_kwargs=hessian_kwargs,
        inverse_kwargs=inverse_kwargs,
    )
    min_rho = float(config.get("dfi.rotation_min_spearman", 0.995))
    max_change = float(config.get("dfi.rotation_max_quartile_change", 0.05))
    qc = {
        "spectral": {
            "passed": spectral.passed,
            "zero_mode_count": spectral.zero_mode_count,
            "expected_zero_modes": spectral.expected_zero_modes,
            "inversion_threshold": spectral.inversion_threshold,
            "pseudoinverse_symmetry_error": spectral.pseudoinverse_symmetry_error,
            "negative_mode_count": spectral.negative_mode_count,
            "smallest_singular_values": spectral.singular_values_ascending[:10],
            "seventh_smallest": (
                spectral.singular_values_ascending[6]
                if len(spectral.singular_values_ascending) > 6 else None
            ),
        },
        "rotation": rotation,
        "rotation_thresholds": {
            "minimum_spearman": min_rho,
            "maximum_quartile_change_fraction": max_change,
        },
        "rotation_passed": (
            rotation["minimum_spearman"] >= min_rho
            and rotation["maximum_quartile_change_fraction"] <= max_change
        ),
    }
    output_dir = config.output_dir
    outputs = {
        "dfi": write_csv(
            context[[
                "structure_instance_id", "label_asym_id", "label_seq_id",
                "target_position", "dfi_serial", "dfi", "relative_dfi",
                "pct_dfi", "z_dfi",
            ] + (["functional_dci", "pct_functional_dci"] if functional_positions else [])],
            output_dir / "dfi.csv",
        ),
        "response": output_dir / "perturbation_matrix.npy",
        "qc": write_json(qc, output_dir / "dfi_qc.json"),
    }
    np.save(outputs["response"], response)
    manifest = build_manifest(
        stage="dfi",
        config_path=config.path,
        inputs=[residues_path],
        parameters={
            **hessian_kwargs,
            **inverse_kwargs,
            "direction_mode": mode,
            "n_directions": len(directions),
            "functional_target_positions": sorted(functional_positions),
        },
        warnings=[] if spectral.passed and qc["rotation_passed"] else [
            "DFI quality control did not pass all production thresholds."
        ],
    )
    outputs["manifest"] = write_json(manifest, output_dir / "manifest.dfi.json")
    if bool(config.get("dfi.require_qc_pass", True)) and not (
        spectral.passed and qc["rotation_passed"]
    ):
        raise DFIError(
            "DFI production QC failed. Inspect dfi_qc.json; adjust a frozen "
            "method setting only with scientific justification, or set "
            "dfi.require_qc_pass=false for a labeled plumbing run."
        )
    return outputs
