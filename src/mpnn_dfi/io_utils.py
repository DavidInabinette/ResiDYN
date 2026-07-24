from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def write_json(value: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return output


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def build_manifest(
    *,
    stage: str,
    config_path: Path,
    inputs: Iterable[Path],
    parameters: dict[str, Any],
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    source_files: dict[str, Any] = {}
    for path in inputs:
        resolved = path.resolve()
        source_files[str(resolved)] = {
            "exists": resolved.exists(),
            "sha256": sha256_file(resolved) if resolved.is_file() else None,
        }
    return {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "inputs": source_files,
        "parameters": parameters,
        "warnings": list(warnings),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

