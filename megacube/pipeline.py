"""End-to-end pipeline used by ``megacube build``: input -> coverage -> geometry -> outputs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .config import Config
from .coords import to_model_frame
from .geometry import Geometry, build_geometry
from .mapping import Mapping, coverage
from .model import Build
from .sources.base import open_source

log = logging.getLogger(__name__)


def load_build(spec: str) -> Build:
    """Any source spec (synthetic, megacube JSON in either frame, later .sav) -> model frame."""
    return to_model_frame(open_source(spec).read())


def to_trimesh(solid) -> trimesh.Trimesh:
    mesh = solid.to_mesh64()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, np.int64), process=False)


def write_stl(solid, path: Path) -> None:
    to_trimesh(solid).export(path)


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run(spec: str, out_dir: str | Path, mapping: Mapping, cfg: Config) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build = load_build(spec)
    cov = coverage(build, mapping)
    (out / "coverage.txt").write_text(cov.format_text() + "\n")
    geo: Geometry = build_geometry(build, mapping, cfg)

    files = {}
    for name, solid in geo.groups().items():
        path = out / f"assembled_{name}.stl"
        write_stl(solid, path)
        files[name] = path.name
    if not geo.placeholders.is_empty():
        write_stl(geo.placeholders, out / "debug_placeholders.stl")
        files["debug_placeholders"] = "debug_placeholders.stl"

    report = {"input": spec, "source": build.source, "coverage": cov.to_dict(), "geometry": geo.report, "files": files}
    (out / "report.json").write_text(json.dumps(_jsonable(report), indent=1))
    return report
