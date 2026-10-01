"""The parser interface: every input format becomes a game-frame :class:`~megacube.model.Build`.

To add the real input in Phase 1, implement a ``BuildSource`` whose ``read()`` returns a
``Build(frame="unreal")`` filled with :class:`~megacube.model.BuildObject` records. The rest of
the pipeline (normalisation, mapping, geometry) doesn't change.

Contract for ``read()``:

* one ``BuildObject`` per placed object, lightweight buildables included (one per instance);
* ``class_name`` is the short class (``Build_Foundation_8x4_01_C``); ``type_path`` is the full path;
* transforms, spline points and lift tops exactly as stored (cm, left-handed, local where the
  save is local). **No unit or handedness conversion in the source**;
* anything the source can't interpret is kept (``props``) or reported, never silently dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ..model import Build


class BuildSource(Protocol):
    kind: str

    def read(self) -> Build:
        """Return the build in the game frame (``frame == "unreal"``)."""
        ...


class JsonBuildSource:
    """A ``megacube-build`` JSON file in either frame (e.g. written by the Phase 1 extractor)."""

    kind = "json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> Build:
        build = Build.load(self.path)
        build.source.setdefault("file", str(self.path))
        return build


def open_source(spec: str, **options) -> BuildSource:
    """Pick a source from a path or spec string.

    ``synthetic`` / ``synthetic:n=4`` -> synthetic cube; ``*.json`` -> megacube JSON;
    ``*.sav`` -> save (Phase 1); ``*.cbp`` -> megaprint (pending format identification).
    """
    if spec == "synthetic" or spec.startswith("synthetic:"):
        from .synthetic import SyntheticCubeSource

        params = dict(kv.split("=", 1) for kv in spec.partition(":")[2].split(",") if kv)
        return SyntheticCubeSource(**{**{k: _num(v) for k, v in params.items()}, **options})
    suffix = Path(spec).suffix.lower()
    if suffix == ".json":
        return JsonBuildSource(spec)
    if suffix == ".sav":
        from .sav import SavSource

        return SavSource(spec, **options)
    if suffix in (".cbp", ".megaprint"):
        from .megaprint import MegaprintSource

        return MegaprintSource(spec, **options)
    raise ValueError(f"don't know how to read {spec!r} (expected synthetic, .json, .sav or .cbp)")


def _num(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def select_objects(build: Build, bbox=None, ids=None, classes=None) -> Build:
    """Keep objects whose origin lies in ``bbox`` = (xmin, ymin, zmin, xmax, ymax, zmax), in the
    build's own frame and units, and/or whose id/class is listed. Used to cut the cube out of a
    whole-world save."""
    keep = []
    lo = hi = None
    if bbox is not None:
        b = np.asarray(bbox, float).reshape(2, 3)
        lo, hi = b.min(0), b.max(0)
    ids = set(ids) if ids else None
    classes = set(classes) if classes else None
    for o in build.objects:
        if lo is not None:
            p = np.asarray(o.transform.translation)
            if np.any(p < lo) or np.any(p > hi):
                continue
        if ids is not None and o.id not in ids:
            continue
        if classes is not None and o.class_name not in classes:
            continue
        keep.append(o)
    source = dict(build.source)
    source["selection"] = {"bbox": None if bbox is None else list(map(float, bbox)),
                           "ids": None if ids is None else len(ids),
                           "classes": None if classes is None else sorted(classes),
                           "kept": len(keep), "of": len(build.objects)}
    return Build(keep, frame=build.frame, source=source, extras=build.extras)
