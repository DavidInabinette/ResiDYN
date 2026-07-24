from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import TargetConfig, resolve_path
from .io_utils import build_manifest, write_csv, write_json


class ConservationError(ValueError):
    """Raised when conservation scores cannot be interpreted."""


def read_consurf_grades(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "%", "POS")):
            continue
        fields = re.split(r"\s+", stripped)
        if len(fields) < 4 or not fields[0].isdigit():
            continue
        try:
            score = float(fields[2])
            grade_text = fields[3].rstrip("*")
            grade = int(grade_text)
        except ValueError:
            continue
        rows.append(
            {
                "target_position": int(fields[0]),
                "consurf_wt": fields[1].upper(),
                "consurf_score": score,
                "consurf_grade": grade,
                "consurf_reliable": not fields[3].endswith("*"),
            }
        )
    if not rows:
        raise ConservationError(f"No ConSurf grade rows recognized in {path}")
    frame = pd.DataFrame(rows)
    if frame["target_position"].duplicated().any():
        raise ConservationError("ConSurf file contains duplicate positions")
    return frame


def read_conservation(path: str | Path, fmt: str) -> pd.DataFrame:
    if fmt.lower() == "grades":
        return read_consurf_grades(path)
    if fmt.lower() == "csv":
        frame = pd.read_csv(path)
        required = {"target_position", "consurf_score"}
        missing = required - set(frame.columns)
        if missing:
            raise ConservationError(f"Conservation CSV lacks: {sorted(missing)}")
        if "consurf_reliable" not in frame:
            frame["consurf_reliable"] = True
        return frame
    raise ConservationError(f"Unsupported conservation format: {fmt}")


def run_consurf_stage(config: TargetConfig) -> dict[str, Path]:
    path = resolve_path(config, config.require("consurf.path"))
    frame = read_conservation(path, str(config.get("consurf.format", "grades")))
    output = write_csv(frame, config.output_dir / "consurf.csv")
    manifest = build_manifest(
        stage="consurf",
        config_path=config.path,
        inputs=[path],
        parameters={
            "format": config.get("consurf.format", "grades"),
            "n_positions": len(frame),
            "n_unreliable": int((~frame["consurf_reliable"].astype(bool)).sum()),
        },
    )
    return {
        "consurf": output,
        "manifest": write_json(manifest, config.output_dir / "manifest.consurf.json"),
    }

