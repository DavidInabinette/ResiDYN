from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


class DCIError(ValueError):
    pass


@dataclass(frozen=True)
class SpectralQC:
    zero_modes: int
    negative_modes: int
    threshold: float
    symmetry_error: float
    smallest_modes: np.ndarray
    seventh_mode: float | None

    @property
    def passed(self) -> bool:
        return (
            self.zero_modes == 6
            and self.negative_modes == 0
            and self.symmetry_error < 1e-8
        )


def hessian(
    coordinates: np.ndarray,
    *,
    gamma: float = 100.0,
    spring_law: str = "d6",
    cutoff: float | None = None,
) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
        raise DCIError("C-alpha coordinates must have shape (N, 3), with N >= 3")
    if not np.isfinite(xyz).all():
        raise DCIError("C-alpha coordinates contain non-finite values")
    if spring_law not in {"d6", "uniform"}:
        raise DCIError(f"Unknown spring law: {spring_law}")
    matrix = np.zeros((3 * len(xyz), 3 * len(xyz)))
    for i in range(len(xyz) - 1):
        for j in range(i + 1, len(xyz)):
            delta = xyz[i] - xyz[j]
            squared_distance = float(delta @ delta)
            if squared_distance <= 0:
                raise DCIError(f"Duplicate C-alpha coordinates at {i} and {j}")
            distance = np.sqrt(squared_distance)
            if cutoff is not None and distance > cutoff:
                continue
            spring = gamma**3 / squared_distance**3 if spring_law == "d6" else gamma
            block = spring * np.outer(delta, delta) / squared_distance
            ii, jj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            matrix[ii, ii] += block
            matrix[jj, jj] += block
            matrix[ii, jj] -= block
            matrix[jj, ii] -= block
    return matrix


def invert_hessian(
    matrix: np.ndarray,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-12,
) -> tuple[np.ndarray, SpectralQC]:
    matrix = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    threshold = max(
        absolute_tolerance,
        relative_tolerance * max(float(np.max(np.abs(values))), 1.0),
    )
    inverse = np.zeros_like(values)
    inverse[values > threshold] = 1.0 / values[values > threshold]
    covariance = (vectors * inverse) @ vectors.T
    modes = np.sort(np.abs(values))
    qc = SpectralQC(
        zero_modes=int((np.abs(values) <= threshold).sum()),
        negative_modes=int((values < -threshold).sum()),
        threshold=float(threshold),
        symmetry_error=float(np.max(np.abs(covariance - covariance.T))),
        smallest_modes=modes[:10],
        seventh_mode=float(modes[6]) if len(modes) > 6 else None,
    )
    return covariance, qc


def fibonacci_directions(count: int = 256) -> np.ndarray:
    if count < 6:
        raise DCIError("At least six perturbation directions are required")
    index = np.arange(count, dtype=float) + 0.5
    z = 1 - 2 * index / count
    radius = np.sqrt(np.maximum(0, 1 - z * z))
    angle = np.pi * (3 - np.sqrt(5)) * index
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


def legacy_directions() -> np.ndarray:
    directions = np.array(
        [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        dtype=float,
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def perturbation_matrix(
    covariance: np.ndarray,
    directions: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise DCIError("Covariance must be square")
    if covariance.shape[0] % 3:
        raise DCIError("Covariance dimension must be divisible by three")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise DCIError("Directions must have shape (K, 3)")
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    count = covariance.shape[0] // 3
    response = np.zeros((count, count))
    for perturbed in range(count):
        forces = np.zeros((3 * count, len(directions)))
        forces[3 * perturbed:3 * perturbed + 3] = directions.T
        displacement = covariance @ forces
        magnitudes = np.linalg.norm(
            displacement.reshape(count, 3, len(directions)), axis=1
        )
        response[:, perturbed] = magnitudes.mean(axis=1)
    if normalize:
        total = float(response.sum())
        if total <= 0:
            raise DCIError("Perturbation response has non-positive total")
        response /= total
    return response


def percentile(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return rankdata(array, method="max") / len(array)


def dfi_table(response: np.ndarray) -> pd.DataFrame:
    values = np.asarray(response).sum(axis=1)
    mean, std = float(values.mean()), float(values.std(ddof=0))
    return pd.DataFrame(
        {
            "dfi": values,
            "relative_dfi": values / mean if mean else np.nan,
            "pct_dfi": percentile(values),
            "z_dfi": (values - mean) / std if std else 0.0,
        }
    )


def functional_dci(
    response: np.ndarray, functional_indices: Iterable[int]
) -> pd.DataFrame:
    """Functional set to responding-residue DCI.

    Response rows are responding residues and columns are perturbed residues.
    """

    indices = np.array(sorted(set(int(i) for i in functional_indices)), dtype=int)
    if len(indices) == 0:
        raise DCIError("Functional DCI requires at least one residue")
    matrix = np.asarray(response, dtype=float)
    if indices.min() < 0 or indices.max() >= matrix.shape[1]:
        raise DCIError("Functional residue index is outside the response matrix")
    numerator = matrix[:, indices].mean(axis=1)
    denominator = matrix.mean(axis=1)
    dci = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator != 0,
    )
    return pd.DataFrame({"functional_dci": dci, "pct_functional_dci": percentile(dci)})


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _quartiles(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average") / len(values)
    return np.minimum((ranks * 4).astype(int), 3)


def calculate(
    coordinates: np.ndarray,
    *,
    direction_count: int = 256,
    rotation_trials: int = 8,
    seed: int = 20260724,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    directions = fibonacci_directions(direction_count)
    covariance, spectral = invert_hessian(hessian(coordinates))
    response = perturbation_matrix(covariance, directions)
    table = dfi_table(response)
    reference = table["pct_dfi"].to_numpy(float)
    reference_quartiles = _quartiles(reference)
    rng = np.random.default_rng(seed)
    rotations = []
    for trial in range(rotation_trials):
        rotated = np.asarray(coordinates) @ _rotation(rng).T
        rotated_covariance, _ = invert_hessian(hessian(rotated))
        rotated_dfi = dfi_table(
            perturbation_matrix(rotated_covariance, directions)
        )["pct_dfi"].to_numpy(float)
        rotations.append(
            {
                "trial": trial + 1,
                "spearman": float(spearmanr(reference, rotated_dfi).statistic),
                "quartile_change_fraction": float(
                    np.mean(reference_quartiles != _quartiles(rotated_dfi))
                ),
            }
        )
    minimum_rho = min((item["spearman"] for item in rotations), default=1.0)
    maximum_change = max(
        (item["quartile_change_fraction"] for item in rotations), default=0.0
    )
    qc = {
        "method": {
            "spring_law": "gamma^3/d^6",
            "gamma": 100.0,
            "cutoff": None,
            "direction_mode": "fibonacci",
            "direction_count": direction_count,
        },
        "spectral": {
            "passed": spectral.passed,
            "zero_mode_count": spectral.zero_modes,
            "negative_mode_count": spectral.negative_modes,
            "inversion_threshold": spectral.threshold,
            "pseudoinverse_symmetry_error": spectral.symmetry_error,
            "smallest_modes": spectral.smallest_modes,
            "seventh_mode": spectral.seventh_mode,
        },
        "rotation": {
            "trials": rotations,
            "minimum_spearman": minimum_rho,
            "maximum_quartile_change_fraction": maximum_change,
            "passed": minimum_rho >= 0.995 and maximum_change <= 0.05,
        },
    }
    qc["passed"] = bool(qc["spectral"]["passed"] and qc["rotation"]["passed"])
    return table, response, qc

