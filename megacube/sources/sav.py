"""``.sav`` input. Phase 1, **not implemented yet** (waits for the real file and your go-ahead).

Planned implementation (see docs/phase0_research.md):

1. ``tools/sav_extract.js`` (Node) loads the save with ``@etothepii/satisfactory-file-parser`` and
   writes a compact ``megacube-build`` JSON in the game frame. It reads actors *and* the
   ``FGLightweightBuildableSubsystem`` instances, belt ``mSplineData``, lift ``mTopTransform``,
   customization data and the swatch colour table.
2. This class runs that script via ``subprocess`` and returns ``Build.load(output)``.
3. ``select_objects(bbox=...)`` cuts the cube out of the world.
"""
from __future__ import annotations

from pathlib import Path


class SavSource:
    kind = "sav"

    def __init__(self, path: str | Path, **options):
        self.path = Path(path)
        self.options = options

    def read(self):
        raise NotImplementedError(
            "Reading .sav files is Phase 1, which waits for the real save and your go-ahead. "
            "Until then use 'synthetic' or a megacube-build JSON.")
