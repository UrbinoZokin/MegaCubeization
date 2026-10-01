"""Synthetic test build: a hollow cube of foundations with belts, a lift and signs on its faces.

This stands in for the real file until it arrives. It isn't a guess at your cube's contents.
It's built to exercise the pipeline:

* six faces of ``n x n`` foundations, side faces rotated to stand vertically (as a SCIM edit would);
* belts on the top face resting on conveyor poles, 1 m above the surface (the typical pole height
  measured in public saves): a straight one, and an L-shaped one with a 90 degree Hermite curve;
* a belt rotated onto the +X face (local up = +X), and a conveyor lift running up the -X face;
* signs on the +Y and -Y faces; the -Y "marker" sign sits near the +X edge, lower half, so it is
  asymmetric and its position checks the coordinate conversion;
* a wall and a ramp standing on the top face (pivot conventions), conveyor poles (excluded by rule),
  a power pole and a modded class (unmapped: must be reported and placeholdered, not dropped).

All coordinates are in the game frame (cm, left-handed, Z-up), like a real parse.
"""
from __future__ import annotations

import math

import numpy as np

from ..model import Build, BuildObject, SplinePoint, Transform, matrix_to_quat

FOUNDATION_HEIGHTS = {"Build_Foundation_8x1_01_C": 100.0, "Build_Foundation_8x2_01_C": 200.0,
                      "Build_Foundation_8x4_01_C": 400.0}


def frame(z_axis, x_hint) -> np.ndarray:
    """Rotation matrix whose columns are local X, Y, Z (proper rotation, det = +1)."""
    z = np.asarray(z_axis, float)
    z /= np.linalg.norm(z)
    x = np.asarray(x_hint, float)
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _tf(rot: np.ndarray, pos) -> Transform:
    return Transform(matrix_to_quat(rot), tuple(float(c) for c in pos))


def hermite_polyline(points, arcs=None):
    """World-space spline points with tangents for a path of straight runs and circular arcs.

    ``points``: corner-free path vertices. ``arcs``: {segment index: True} marks segments that
    are 90 degree arcs; their tangent length is 4*tan(theta/4)*r, the best Hermite fit to a circle.
    Points between two straight runs get Catmull-Rom tangents, so the curve stays C1 like a game spline.
    """
    pts = [np.asarray(p, float) for p in points]
    arcs = arcs or {}
    seg_tangents = []
    for i in range(len(pts) - 1):
        chord = pts[i + 1] - pts[i]
        if i in arcs:
            r = np.linalg.norm(chord) / math.sqrt(2)  # 90 degree arc: chord = r*sqrt(2)
            k = 4 * math.tan(math.pi / 8) * r
            d0 = arcs[i]["start_dir"] / np.linalg.norm(arcs[i]["start_dir"])
            d1 = arcs[i]["end_dir"] / np.linalg.norm(arcs[i]["end_dir"])
            seg_tangents.append((d0 * k, d1 * k))
        else:
            seg_tangents.append((chord, chord))
    out = []
    for i, p in enumerate(pts):
        arrive = seg_tangents[i - 1][1] if i > 0 else seg_tangents[0][0]
        leave = seg_tangents[i][0] if i < len(seg_tangents) else seg_tangents[-1][1]
        if 0 < i < len(pts) - 1 and (i - 1) not in arcs and i not in arcs:
            arrive = leave = (pts[i + 1] - pts[i - 1]) / 2  # Catmull-Rom: C1-continuous like game splines
        out.append((p, arrive, leave))
    return out


class SyntheticCubeSource:
    kind = "synthetic"

    def __init__(self, n: int = 6, foundation: str = "Build_Foundation_8x1_01_C", features: bool = True):
        if foundation not in FOUNDATION_HEIGHTS:
            raise ValueError(f"foundation must be one of {sorted(FOUNDATION_HEIGHTS)}")
        self.n, self.foundation, self.features = int(n), foundation, bool(features)

    def read(self) -> Build:
        n, h = self.n, FOUNDATION_HEIGHTS[self.foundation]
        half = n * 800.0 / 2  # outer half-size of the cube
        objs: list[BuildObject] = []

        def add(cls, tf, **kw):
            objs.append(BuildObject(id=f"synthetic:{len(objs)}:{cls}", class_name=cls, transform=tf,
                                    storage="synthetic", **kw))

        # --- six faces of foundations; each face spans the full cube width (they overlap at edges)
        cells = (np.arange(n) - (n - 1) / 2) * 800.0
        for normal, x_hint in [((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (1, 0, 0)), ((1, 0, 0), (0, 0, -1)),
                               ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (1, 0, 0)), ((0, -1, 0), (1, 0, 0))]:
            rot = frame(normal, x_hint)
            centre = np.asarray(normal, float) * (half - h / 2)
            for u in cells:
                for v in cells:
                    add(self.foundation, _tf(rot, centre + rot[:, 0] * u + rot[:, 1] * v),
                        swatch="SwatchDesc_Slot16_C")
        if not self.features:
            return self._build(objs, half)

        top = half  # top surface height
        identity = np.eye(3)

        def add_belt(cls, path, up, arcs=None):
            pts = hermite_polyline(path, arcs)
            first_dir = path[1] - np.asarray(path[0], float)
            rot = frame(up, first_dir)
            origin = np.asarray(path[0], float)
            loc = lambda v: tuple(float(c) for c in rot.T @ v)  # noqa: E731 (world -> belt-local)
            spline = [SplinePoint(loc(p - origin), loc(a), loc(lv)) for p, a, lv in pts]
            add(cls, _tf(rot, origin), spline=spline, swatch="SwatchDesc_Slot0_C")

        # Top face, on poles 1 m above the surface.
        z = top + 100.0
        add_belt("Build_ConveyorBeltMk5_C", [np.array((-2000.0, -1200.0, z)), np.array((2000.0, -1200.0, z))], (0, 0, 1))
        for x in (-2000.0, 2000.0):
            add("Build_ConveyorPole_C", _tf(identity, (x, -1200.0, top)))
        add_belt("Build_ConveyorBeltMk3_C",
                 [np.array((-2000.0, 400.0, z)), np.array((0.0, 400.0, z)), np.array((800.0, 1200.0, z)),
                  np.array((800.0, 2000.0, z))], (0, 0, 1),
                 arcs={1: {"start_dir": np.array((1.0, 0, 0)), "end_dir": np.array((0, 1.0, 0))}})
        for p in ((-2000.0, 400.0), (800.0, 2000.0)):
            add("Build_ConveyorPole_C", _tf(identity, (*p, top)))

        # +X face: belt lying on the face (SCIM-style rotation, local up = +X), 20 cm off the surface.
        xs = half + 20.0
        add_belt("Build_ConveyorBeltMk2_C",
                 [np.array((xs, -1600.0, -1600.0)), np.array((xs, 0.0, 0.0)), np.array((xs, 1600.0, 800.0))], (1, 0, 0))

        # -X face: conveyor lift running up the face (its centre 150 cm in front of the surface).
        add("Build_ConveyorLiftMk5_C", _tf(identity, (-half - 150.0, 1200.0, -1600.0)),
            lift_top=Transform(translation=(0.0, 0.0, 2400.0)))

        # +Y face: two signs facing +Y (identity rotation: front normal = local +Y), 10 cm off the surface.
        add("Build_StandaloneWidgetSign_Huge_C", _tf(identity, (-800.0, half + 10.0, 0.0)),
            props={"text": ["synthetic"]})
        add("Build_StandaloneWidgetSign_Small_C", _tf(identity, (1600.0, half + 10.0, 1200.0)))
        # -Y face: orientation marker near the +X edge, lower half. Facing -Y = yaw 180 degrees.
        add("Build_StandaloneWidgetSign_SmallVeryWide_C", _tf(frame((0, 0, 1), (-1, 0, 0)), (1600.0, -half - 10.0, -1200.0)),
            props={"role": "orientation-marker"})

        # Top face decorations: a wall on a foundation edge line, a ramp on a cell (high end at -X).
        add("Build_Wall_8x4_01_C", _tf(identity, (1600.0, -2000.0, top)))
        add("Build_Ramp_8x2_01_C", _tf(identity, (-1200.0, -2000.0, top + 100.0)))

        # Unmapped on purpose: must be counted, warned about and placeholdered, never dropped.
        add("Build_PowerPoleMk1_C", _tf(identity, (2000.0, 2000.0, top)))
        add("Build_Wall_10_C", _tf(identity, (-2000.0, 2000.0, top)), type_path="/MoreDecorations/Wall/Build_Wall_10.Build_Wall_10_C")
        return self._build(objs, half)

    def _build(self, objs, half) -> Build:
        return Build(objs, frame="unreal", source={
            "kind": "synthetic", "note": "synthetic test cube, not the user's build",
            "params": {"n": self.n, "foundation": self.foundation, "features": self.features,
                       "outer_size_cm": 2 * half}})
