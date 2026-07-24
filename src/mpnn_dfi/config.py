from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a target configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class TargetConfig:
    path: Path
    data: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    @property
    def output_dir(self) -> Path:
        return resolve_path(self, self.require("project.output_dir"))

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.data
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None or value == "":
            raise ConfigError(f"Missing required configuration value: {dotted}")
        return value


def load_config(path: str | Path) -> TargetConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration must be a YAML mapping: {config_path}")
    cfg = TargetConfig(config_path, data)
    for required in (
        "project.target_id",
        "project.structure_instance_id",
        "project.output_dir",
        "structure.path",
        "structure.label_asym_id",
        "structure.target_sequence",
    ):
        cfg.require(required)
    return cfg


def resolve_path(config: TargetConfig, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config.base_dir / path).resolve()
    return path

