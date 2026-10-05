"""SCIM megaprint (``.cbp``) input. **Format unknown**, deliberately not implemented.

No public spec or parser exists (docs/phase0_research.md section 3.2). When the file arrives,
run ``megacube sniff megaprint.cbp`` and decide from what it actually contains.
"""
from __future__ import annotations

from pathlib import Path

from .sniff import sniff


class MegaprintSource:
    kind = "megaprint"

    def __init__(self, path: str | Path, **options):
        self.path = Path(path)
        self.options = options

    def read(self):
        stages = [s["layer"] for s in sniff(self.path)["stages"]]
        raise NotImplementedError(
            f"Megaprint parsing isn't implemented: the format is undocumented. Detected encoding layers: "
            f"{' -> '.join(stages)}. Run 'megacube sniff {self.path}' and share the output.")
