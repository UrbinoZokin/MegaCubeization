"""Command-line entry point: ``megacube <command> ...``."""
from __future__ import annotations

import argparse
import json
import sys


def cmd_sniff(args: argparse.Namespace) -> int:
    from .sources.sniff import sniff

    print(json.dumps(sniff(args.file, max_depth=args.depth), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="megacube", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sniff", help="identify the encoding layers of an unknown build file (e.g. a megaprint)")
    s.add_argument("file")
    s.add_argument("--depth", type=int, default=4, help="maximum number of decoding layers to peel")
    s.set_defaults(func=cmd_sniff)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
