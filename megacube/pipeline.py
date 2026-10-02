"""End-to-end pipeline used by ``megacube build``: input -> coverage -> geometry -> outputs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .coords import to_model_frame
from .geometry import Geometry, build_geometry
from .mapping import Mapping, coverage
from .model import Build
from .printability import check_part, support_report
from .sources.base import open_source
from .export import write_3mf, write_stl
from .split import Part, split_geometry
from .validate import format_reports, validate_file

log = logging.getLogger(__name__)


def load_build(spec: str) -> Build:
    """Any source spec (synthetic, megacube JSON in either frame, later .sav) -> model frame."""
    return to_model_frame(open_source(spec).read())


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


def export_all(out: Path, geo: Geometry, parts: list[Part]) -> dict[str, str]:
    """Assembled STLs, per-part STLs (shared origin per part), per-part and combined 3MF."""
    files = {}
    (out / "assembled").mkdir(exist_ok=True)
    for g, solid in geo.groups().items():
        files[f"assembled/{g}"] = f"assembled/{g}.stl"
        write_stl(solid, out / files[f"assembled/{g}"])
    if not geo.placeholders.is_empty():
        files["assembled/debug_placeholders"] = "assembled/debug_placeholders.stl"
        write_stl(geo.placeholders, out / files["assembled/debug_placeholders"])
    (out / "parts").mkdir(exist_ok=True)
    for p in parts:
        for g, solid in p.solids.items():
            files[f"parts/{p.name}_{g}"] = f"parts/{p.name}_{g}.stl"
            write_stl(solid, out / files[f"parts/{p.name}_{g}"])
        files[f"parts/{p.name}.3mf"] = f"parts/{p.name}.3mf"
        write_3mf([(p.name, p.solids)], out / files[f"parts/{p.name}.3mf"])
    files["parts/all_parts.3mf"] = "parts/all_parts.3mf"
    write_3mf([(p.name, p.solids) for p in parts], out / "parts/all_parts.3mf")
    return files


def summary_markdown(report: dict[str, Any]) -> str:
    g, cov, sp, ck = report["geometry"], report["coverage"], report["split"], report["checks"]
    val = report.get("validation", [])
    lines = [f"# megacube build: {report['input']}", ""]
    src = report.get("source", {})
    if src.get("kind") == "synthetic":
        lines += ["> Synthetic test build, not your cube.", ""]
    lines += ["## Mapping",
              f"- {cov['total_objects']} objects; mapped by a rule: {100 * cov['mapped_fraction']:.1f}%, "
              f"printed: {100 * cov['printed_fraction']:.1f}%",
              f"- unmapped classes: {cov['unmapped_classes'] or 'none'} (placeholders in "
              "`assembled/debug_placeholders.stl`, `preview/debug_unmapped.png`)" if cov["unmapped_classes"] else
              "- unmapped classes: none",
              f"- rules in use that aren't fully verified: {', '.join(cov['not_fully_verified_rules_in_use']) or 'none'}",
              "", "## Scale and size",
              f"- scale factor **{g['scale']:.6f}** (1 game metre = {g['scale'] * 1000:.3f} mm); "
              f"build is {g['model_extent_m']} m in game, printed {g['printed_extent_mm']} mm",
              f"- shell {report['config']['hollow']['wall']} mm, roof: {g.get('roof')}, windows: "
              f"{g.get('windows', {}).get('mode', 'none')} ({g.get('windows', {}).get('made', 0)} made, "
              f"{g.get('windows', {}).get('skipped', 0)} skipped), attach: {g['attach']['mode']} "
              f"({g['attach']['floating']} floating)", "",
              f"## Parts ({sp['mode']})"]
    for name, p in ck["parts"].items():
        s = p["supports"]
        lines.append(f"- **{name}** {p['size_mm']} mm, fits: {p['fits_build_volume']}, bed contact {s['bed_contact_mm2']} mm2, "
                     f"internal supports {s['support_on_model_mm2']} mm2, plate supports {s['support_from_plate_mm2']} mm2, "
                     f"thin walls {p['thin_walls']}, narrow gaps {p['narrow_gaps']}")
    if "pins" in sp:
        lines.append(f"- pins: {sp['pins']}")
    lines += ["", "## Validation",
              f"- {sum(1 for r in val if r['ok'])}/{len(val)} STL files valid (watertight, consistent winding, "
              "outward normals, no degenerate triangles, no self-intersections)"]
    lines += [f"  - FAIL {r['file']}" for r in val if not r["ok"]]
    warnings = g["warnings"] + sp["warnings"]
    if warnings:
        lines += ["", "## Warnings"] + [f"- {w}" for w in warnings]
    if report.get("previews"):
        lines += ["", "## Previews"] + [f"![{k}]({v})" for k, v in report["previews"].items()]
    lines += ["", "Details: `report.json`, `coverage.txt`, `printability.txt`, `validation.txt`."]
    return "\n".join(lines) + "\n"


def run(spec: str, out_dir: str | Path, mapping: Mapping, cfg: Config, previews: bool = True,
        validate: bool = True) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build = load_build(spec)
    cov = coverage(build, mapping)
    (out / "coverage.txt").write_text(cov.format_text() + "\n")
    geo: Geometry = build_geometry(build, mapping, cfg)
    parts, split_rep = split_geometry(geo, cfg)
    checks = check_parts(parts, geo, cfg)
    (out / "printability.txt").write_text(format_checks(checks) + "\n")
    files = export_all(out, geo, parts)

    report = {"input": spec, "source": build.source, "config": dict(cfg), "coverage": cov.to_dict(),
              "geometry": geo.report, "split": split_rep, "checks": checks, "files": files}
    if validate:
        reports = [validate_file(out / f) for f in files.values() if f.endswith(".stl")]
        for r in reports:
            r.file = str(Path(r.file).relative_to(out))
        (out / "validation.txt").write_text(format_reports(reports) + "\n")
        report["validation"] = [r.to_dict() for r in reports]
    if previews:
        from .preview import write_previews

        report["previews"] = write_previews(out / "preview", geo, parts, size=int(cfg.get_path("preview.size", 700)))
    (out / "report.json").write_text(json.dumps(_jsonable(report), indent=1))
    (out / "summary.md").write_text(summary_markdown(report))
    return report
