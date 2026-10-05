"""Pipeline configuration: YAML defaults plus ``--set dotted.key=value`` overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PIPELINE = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"


class ConfigError(ValueError):
    pass


class Config(dict):
    """Nested dict with dotted access: ``cfg.get_path("hollow.wall")``."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(f"unknown config section {dotted!r}")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigError(f"unknown config key {dotted!r}")
        node[parts[-1]] = value


def _merge(base: dict, extra: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if k not in out:
            raise ConfigError(f"unknown config key {where + k!r}")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v, where + k + ".")
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Defaults from config/pipeline.yaml, then an optional user YAML (only keys that exist in the
    defaults are allowed, which catches typos), then ``key=value`` overrides (values parsed as YAML)."""
    cfg = yaml.safe_load(DEFAULT_PIPELINE.read_text())
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    cfg = Config(cfg)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ConfigError(f"override {item!r} must look like key=value")
        cfg.set_path(key.strip(), yaml.safe_load(raw))
    return cfg
