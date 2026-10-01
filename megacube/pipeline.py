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
from .printability import check_part, support_report
from .sources.base import open_source
from .split import Part, split_geometry

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


def check_parts(parts: list[Part], geo: Geometry, cfg: Config) -> dict[str, Any]:
    """Phase 4 checks for every part, in its print orientation."""
    ck = cfg["checks"]
    vol = np.asarray(cfg.get_path("scale.build_volume"), float)
    centre = np.array(geo.filled.bounding_box()).reshape(2, 3).mean(0)
    out, findings = {}, []
    for p in parts:
        bb = p.bbox()
        size = bb[3:] - bb[:3]
        fits = bool(np.all(np.sort(size[:2]) <= np.sort(vol[:2]) + 1e-9) and size[2] <= vol[2] + 1e-9)
        f = check_part(p.name, p.solids, p.to_assembled, centre, float(ck["min_wall"]), float(ck["min_gap"]),
                       int(ck["samples"]))
        sup, f_sup = support_report(p.name, p.solids, float(ck["overhang_deg"]), p.to_assembled, centre,
                                    reach=float(ck["overhang_reach"]))
        findings += f + f_sup
        out[p.name] = {
            "size_mm": size.round(2).tolist(), "fits_build_volume": fits,
            "volumes_mm3": {g: round(s.volume(), 1) for g, s in p.solids.items()},
            "supports": sup,
            "thin_walls": sum(1 for x in f if x.kind == "thin_wall"),
            "narrow_gaps": sum(1 for x in f if x.kind == "narrow_gap"),
            "internal_supports": len(f_sup),
        }
    return {"parts": out, "findings": [x.to_dict() for x in findings],
            "limits": {"min_wall_mm": ck["min_wall"], "min_gap_mm": ck["min_gap"], "overhang_deg": ck["overhang_deg"]}}


def format_checks(checks: dict[str, Any]) -> str:
    lim = checks["limits"]
    lines = [f"Printability (walls < {lim['min_wall_mm']} mm, gaps < {lim['min_gap_mm']} mm, "
             f"overhangs > {lim['overhang_deg']} deg)"]
    for name, p in checks["parts"].items():
        s = p["supports"]
        lines.append(f"  {name:<7} {p['size_mm']} mm  fits build volume: {p['fits_build_volume']}  "
                     f"bed contact {s['bed_contact_mm2']} mm2")
        lines.append(f"          overhangs: short ledges {s['short_ledges_mm2']} mm2, support from plate "
                     f"{s['support_from_plate_mm2']} mm2, INTERNAL (on model) {s['support_on_model_mm2']} mm2; "
                     f"thin walls: {p['thin_walls']}, narrow gaps: {p['narrow_gaps']}")
    if checks["findings"]:
        lines.append("")
        lines.append("  Findings (location = centre of the flagged area, assembled-model mm; part = print coords):")
        for f in checks["findings"]:
            what = {"thin_wall": f"thin wall {f['value_mm']} mm", "narrow_gap": f"narrow gap {f['value_mm']} mm",
                    "support_on_model": f"needs support on the model ({f['value_mm']} mm2)"}[f["kind"]]
            lines.append(f"  - {f['part']}/{f['group']}: {what} on the {f['face']} at {f['location_assembled_mm']} "
                         f"(part {f['location_mm']}, {f['samples']} samples)")
    else:
        lines.append("  No features below the printable limits.")
    return "\n".join(lines)


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
        files[f"assembled_{name}"] = path.name
    if not geo.placeholders.is_empty():
        write_stl(geo.placeholders, out / "debug_placeholders.stl")
        files["debug_placeholders"] = "debug_placeholders.stl"

    parts, split_rep = split_geometry(geo, cfg)
    part_dir = out / "parts"
    part_dir.mkdir(exist_ok=True)
    for p in parts:
        for g, solid in p.solids.items():
            write_stl(solid, part_dir / f"{p.name}_{g}.stl")
            files[f"{p.name}_{g}"] = f"parts/{p.name}_{g}.stl"
    checks = check_parts(parts, geo, cfg)
    (out / "printability.txt").write_text(format_checks(checks) + "\n")

    report = {"input": spec, "source": build.source, "coverage": cov.to_dict(), "geometry": geo.report,
              "split": split_rep, "checks": checks, "files": files}
    (out / "report.json").write_text(json.dumps(_jsonable(report), indent=1))
    return report
