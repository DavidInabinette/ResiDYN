from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from .config import TargetConfig, load_config
from .conservation import run_consurf_stage
from .dfi import run_dfi_stage
from .dms import run_dms_stage
from .features import run_assemble_stage
from .proteinmpnn import run_proteinmpnn_stage
from .structure import run_structure_stage
from .validation import run_validation_stage


Stage = Callable[[TargetConfig], dict[str, Path]]
STAGES: dict[str, Stage] = {
    "structure": run_structure_stage,
    "dfi": run_dfi_stage,
    "dms": run_dms_stage,
    "mpnn": run_proteinmpnn_stage,
    "consurf": run_consurf_stage,
    "assemble": run_assemble_stage,
    "validate": run_validation_stage,
}


def _run(stage: str, config: TargetConfig) -> dict[str, str]:
    outputs = STAGES[stage](config)
    return {name: str(path) for name, path in outputs.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mpnn-dfi",
        description="ProteinMPNN–DFI Phase 1 MVP+ pipeline",
    )
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stage == "all":
        result = {stage: _run(stage, config) for stage in STAGES}
    else:
        result = {args.stage: _run(args.stage, config)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

