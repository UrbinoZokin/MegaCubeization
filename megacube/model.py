"""Data model shared by every input source and the geometry pipeline.

The same schema describes a build in two frames:

* ``frame == "unreal"``: exactly as the game stores it: centimetres, left-handed, Z-up.
  This is the **parser interface**: any source (synthetic generator, the future ``.sav``
  extractor, a megaprint reader) produces a ``Build`` in this frame.
* ``frame == "model"``: millimetres, right-handed, Z-up, for slicers. Produced only by
  :func:`megacube.coords.to_model_frame`. The intermediate JSON (Phase 1 output) uses this frame.

Spline points and lift top transforms are stored **local to the object's transform**, as in the
save, so no precision is lost. Use :meth:`BuildObject.matrix` to place them in the world.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (x, y, z, w)

FRAMES = {
    "unreal": {"units": "cm", "handedness": "left", "up": "+Z"},
    "model": {"units": "mm", "handedness": "right", "up": "+Z"},
}


def quat_to_matrix(q) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix acting on column vectors (v' = R v).

    This is the convention Satisfactory saves use (verified on public saves, see
    docs/phase0_research.md section 5).
    """
    x, y, z, w = (float(c) for c in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero-length quaternion")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def matrix_to_quat(m) -> Quat:
    """3x3 rotation matrix -> quaternion (x, y, z, w) with w >= 0 (inverse of quat_to_matrix)."""
    m = np.asarray(m, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(c) for c in q)


@dataclass
class Transform:
    rotation: Quat = (0.0, 0.0, 0.0, 1.0)
    translation: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    def matrix(self) -> np.ndarray:
        """4x4 affine matrix: world = R @ (scale * local) + t."""
        m = np.eye(4)
        m[:3, :3] = quat_to_matrix(self.rotation) * np.asarray(self.scale, float)[None, :]
        m[:3, 3] = self.translation
        return m


@dataclass
class SplinePoint:
    location: Vec3
    arrive_tangent: Vec3
    leave_tangent: Vec3


@dataclass
class BuildObject:
    id: str
    class_name: str
    transform: Transform
    storage: str = "actor"  # actor | lightweight | synthetic | megaprint
    type_path: str | None = None
    swatch: str | None = None  # e.g. "SwatchDesc_Slot16_C"
    paint: dict[str, Any] | None = None  # pattern / material / finish / explicit colours, as found
    spline: list[SplinePoint] | None = None  # belts: local-space spline (Unreal cubic Hermite)
    lift_top: Transform | None = None  # lifts: top transform relative to the object
    props: dict[str, Any] = field(default_factory=dict)  # anything else worth keeping (sign text, ...)

    def matrix(self) -> np.ndarray:
        return self.transform.matrix()


@dataclass
class Build:
    objects: list[BuildObject]
    frame: str = "unreal"
    source: dict[str, Any] = field(default_factory=dict)  # provenance: kind, file, selection, notes
    extras: dict[str, Any] = field(default_factory=dict)  # e.g. swatch colour table

    def __post_init__(self):
        if self.frame not in FRAMES:
            raise ValueError(f"unknown frame {self.frame!r}; expected one of {sorted(FRAMES)}")

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.objects:
            counts[o.class_name] = counts.get(o.class_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    # ---------------------------------------------------------------- JSON I/O
    def to_dict(self) -> dict[str, Any]:
        def clean(d):
            return {k: v for k, v in d.items() if v not in (None, {}, [])}

        return {
            "format": "megacube-build",
            "version": 1,
            "frame": self.frame,
            **FRAMES[self.frame],
            "source": self.source,
            "extras": self.extras,
            "class_counts": self.class_counts(),
            "objects": [clean(asdict(o)) for o in self.objects],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Build":
        if d.get("format") != "megacube-build":
            raise ValueError("not a megacube build file (missing format: megacube-build)")
        objs = []
        for o in d["objects"]:
            o = dict(o)
            o["transform"] = _transform(o["transform"])
            if o.get("spline"):
                o["spline"] = [SplinePoint(tuple(p["location"]), tuple(p["arrive_tangent"]), tuple(p["leave_tangent"]))
                               for p in o["spline"]]
            if o.get("lift_top"):
                o["lift_top"] = _transform(o["lift_top"])
            objs.append(BuildObject(**o))
        return cls(objects=objs, frame=d["frame"], source=d.get("source", {}), extras=d.get("extras", {}))

    @classmethod
    def load(cls, path: str | Path) -> "Build":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _transform(d: dict[str, Any]) -> Transform:
    return Transform(tuple(d["rotation"]), tuple(d["translation"]), tuple(d.get("scale", (1.0, 1.0, 1.0))))
