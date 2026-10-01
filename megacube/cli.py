"""Command-line entry point: ``megacube <command> ...``."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_sniff(args: argparse.Namespace) -> int:
    from .sources.sniff import sniff

    print(json.dumps(sniff(args.file, max_depth=args.depth), indent=2))
    return 0


def _bbox(text: str | None):
    if not text:
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 6:
        raise SystemExit("--bbox expects xmin,ymin,zmin,xmax,ymax,zmax")
    return vals


def cmd_synth(args: argparse.Namespace) -> int:
    from .sources.synthetic import SyntheticCubeSource

    build = SyntheticCubeSource(n=args.n, foundation=args.foundation, features=not args.no_features).read()
    build.save(args.output)
    print(f"wrote {args.output}: {len(build.objects)} objects (game frame)")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    """Phase 1 shape: any source -> selection -> model frame -> intermediate JSON."""
    from .coords import to_model_frame
    from .sources.base import open_source, select_objects

    build = open_source(args.input).read()
    if args.bbox:
        build = select_objects(build, bbox=_bbox(args.bbox))
    out = to_model_frame(build)
    out.save(args.output)
    print(f"wrote {args.output}: {len(out.objects)} objects, frame=model (mm, right-handed, Z-up)")
    for cls, n in out.class_counts().items():
        print(f"  {n:6d}  {cls}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    from .mapping import coverage, load_mapping
    from .sources.base import open_source

    build = open_source(args.input).read()
    report = coverage(build, load_mapping(args.classes))
    print(report.format_text())
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report.to_dict(), fh, indent=1)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    import logging

    from .config import load_config
    from .mapping import load_mapping
    from .pipeline import run

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s")
    cfg = load_config(args.config, args.set)
    report = run(args.input, args.out, load_mapping(args.classes), cfg)
    geo = report["geometry"]
    cov = report["coverage"]
    print(f"coverage: {100 * cov['mapped_fraction']:.1f}% mapped, unmapped: {cov['unmapped_classes'] or 'none'}")
    print(f"scale {geo['scale']:.6f} (1 game metre = {geo['scale'] * 1000:.3f} mm); printed size {geo['printed_extent_mm']} mm")
    for w in geo["warnings"] + report["split"]["warnings"]:
        print(f"WARNING: {w}")
    from .pipeline import format_checks

    print(format_checks(report["checks"]))
    print(f"outputs in {args.out}: {len(report['files'])} STL files, report.json, coverage.txt, printability.txt")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="megacube", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sniff", help="identify the encoding layers of an unknown build file (e.g. a megaprint)")
    s.add_argument("file")
    s.add_argument("--depth", type=int, default=4, help="maximum number of decoding layers to peel")
    s.set_defaults(func=cmd_sniff)

    s = sub.add_parser("synth", help="write the synthetic test cube (game frame) as megacube JSON")
    s.add_argument("-o", "--output", default="synthetic_raw.json")
    s.add_argument("--n", type=int, default=6, help="foundations per cube edge")
    s.add_argument("--foundation", default="Build_Foundation_8x1_01_C")
    s.add_argument("--no-features", action="store_true", help="bare cube without belts/signs")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("normalize", help="read a build (synthetic, .json, .sav) and write the intermediate JSON")
    s.add_argument("input", help="'synthetic', 'synthetic:n=4', a megacube JSON, or a .sav (Phase 1)")
    s.add_argument("-o", "--output", default="build.json")
    s.add_argument("--bbox", help="keep objects with origin inside xmin,ymin,zmin,xmax,ymax,zmax (game cm)")
    s.set_defaults(func=cmd_normalize)

    s = sub.add_parser("coverage", help="report how the build's classes map to primitives (Phase 2)")
    s.add_argument("input", help="'synthetic' or a megacube JSON (either frame)")
    s.add_argument("--classes", help="class mapping YAML (default: config/classes.yaml)")
    s.add_argument("--json", help="also write the report as JSON")
    s.set_defaults(func=cmd_coverage)

    s = sub.add_parser("build", help="generate printable geometry (Phases 3-5)")
    s.add_argument("input", help="'synthetic', a megacube JSON (either frame), later a .sav")
    s.add_argument("-o", "--out", default="out", help="output directory")
    s.add_argument("--config", help="pipeline YAML overriding config/pipeline.yaml")
    s.add_argument("--classes", help="class mapping YAML (default: config/classes.yaml)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a pipeline setting, e.g. --set hollow.wall=2.4 (repeatable)")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_build)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
