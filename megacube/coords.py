"""Unreal (cm, left-handed, Z-up) -> model (mm, right-handed, Z-up) conversion.

Unreal's frame is X forward, Y right, Z up, which is left-handed. Mirroring the Y axis
(``S = diag(1, -1, 1)``) gives a right-handed frame. Mirroring the *axis* (not the geometry)
means the build is not mirrored: something to the right of an observer facing +X in the game
is still to their right in the model (at -Y, since +Y is "left" in a right-handed Z-up frame).

For a point ``p = R v + t`` in game space, the model point is
``S p * 10 = (S R S)(S v * 10) + S t * 10``, so:

* positions and local offsets: ``(x, y, z) -> (10x, -10y, 10z)``
* rotations: ``R' = S R S``, i.e. quaternion ``(x, y, z, w) -> (-x, y, -z, w)``
* scale factors are unchanged (``S`` is diagonal, so it commutes with them)

Primitive definitions in the class mapping are written in *game-local* coordinates and
converted with :func:`local_to_model`. Asymmetric shapes (corner ramps, signs) therefore come out
the right way round.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .model import Build, BuildObject, SplinePoint, Transform

CM_TO_MM = 10.0
MIRROR = np.diag([1.0, -1.0, 1.0])


def vec_to_model(v) -> tuple[float, float, float]:
    x, y, z = (float(c) for c in v)
    return (x * CM_TO_MM, -y * CM_TO_MM, z * CM_TO_MM)


def local_to_model(v) -> np.ndarray:
    """Game-local cm -> model-local mm for arrays of points/offsets/tangents (..., 3)."""
    return np.asarray(v, float) * np.array([CM_TO_MM, -CM_TO_MM, CM_TO_MM])


def direction_to_model(v) -> np.ndarray:
    """Unit directions: mirror Y, no unit scaling."""
    return np.asarray(v, float) * np.array([1.0, -1.0, 1.0])


def quat_to_model(q) -> tuple[float, float, float, float]:
    x, y, z, w = (float(c) for c in q)
    return (-x, y, -z, w)


def transform_to_model(t: Transform) -> Transform:
    return Transform(quat_to_model(t.rotation), vec_to_model(t.translation), tuple(float(s) for s in t.scale))


def object_to_model(o: BuildObject) -> BuildObject:
    spline = None
    if o.spline:
        spline = [SplinePoint(vec_to_model(p.location), vec_to_model(p.arrive_tangent), vec_to_model(p.leave_tangent))
                  for p in o.spline]
    return replace(o, transform=transform_to_model(o.transform), spline=spline,
                   lift_top=transform_to_model(o.lift_top) if o.lift_top else None)


def to_model_frame(build: Build) -> Build:
    """Convert a game-frame build to the slicer frame (idempotent for builds already in it)."""
    if build.frame == "model":
        return build
    source = dict(build.source)
    source["conversion"] = "unreal(cm, LH, Z-up) -> model(mm, RH, Z-up): mirror Y, x10"
    return Build([object_to_model(o) for o in build.objects], frame="model", source=source, extras=build.extras)
