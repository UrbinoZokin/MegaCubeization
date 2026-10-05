"""Unreal spline evaluation and belt sweeping.

Satisfactory belts store a ``USplineComponent``-style spline: each point has a location plus
arrive/leave tangents. Segment ``k`` is the cubic Hermite curve between point ``k`` (leave
tangent) and point ``k+1`` (arrive tangent), the same basis as ``FMath::CubicInterp``.
"""
from __future__ import annotations

import math

import numpy as np
from manifold3d import Manifold, Mesh64


def hermite(p0, t0, p1, t1, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)[:, None]
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * t0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * t1


def hermite_derivative(p0, t0, p1, t1, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)[:, None]
    s2 = s * s
    return (6 * s2 - 6 * s) * p0 + (3 * s2 - 4 * s + 1) * t0 + (-6 * s2 + 6 * s) * p1 + (3 * s2 - 2 * s) * t1


def sample_spline(locations, arrive, leave, max_segment: float, max_angle_deg: float):
    """Adaptive samples along the whole spline: positions (N,3) and unit tangents (N,3).

    Each segment gets enough samples that consecutive samples are at most ``max_segment`` apart
    and the tangent turns by at most ``max_angle_deg`` between them.
    """
    loc, arr, lea = (np.asarray(a, float) for a in (locations, arrive, leave))
    if len(loc) < 2:
        raise ValueError("a spline needs at least two points")
    pos, tan = [], []
    probe = np.linspace(0.0, 1.0, 33)
    for k in range(len(loc) - 1):
        p0, t0, p1, t1 = loc[k], lea[k], loc[k + 1], arr[k + 1]
        pts = hermite(p0, t0, p1, t1, probe)
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        d = hermite_derivative(p0, t0, p1, t1, probe)
        d_unit = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        turn = float(np.degrees(np.arccos(np.clip((d_unit[:-1] * d_unit[1:]).sum(1), -1, 1)).sum()))
        n = max(1, math.ceil(length / max(max_segment, 1e-9)), math.ceil(turn / max(max_angle_deg, 1e-9)))
        s = np.linspace(0.0, 1.0, n + 1)
        if k < len(loc) - 2:
            s = s[:-1]  # the next segment starts with this point
        pos.append(hermite(p0, t0, p1, t1, s))
        tan.append(hermite_derivative(p0, t0, p1, t1, s))
    P, T = np.vstack(pos), np.vstack(tan)
    norms = np.linalg.norm(T, axis=1)
    for i in np.where(norms < 1e-9)[0]:  # degenerate tangent: fall back to the chord direction
        j0, j1 = max(i - 1, 0), min(i + 1, len(P) - 1)
        T[i] = P[j1] - P[j0]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]  # drop duplicate samples
    return P[keep], T[keep]


def profile_frames(T: np.ndarray, up) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample (U, S): U = ``up`` made perpendicular to the tangent, S = T x U.

    Where the belt runs parallel to ``up`` (a vertical drop), the previous U is carried over.
    """
    up = np.asarray(up, float)
    U = np.empty_like(T)
    prev = None
    for i, t in enumerate(T):
        u = up - (up @ t) * t
        n = np.linalg.norm(u)
        if n < 1e-6:
            if prev is None:  # starts vertical: pick any perpendicular
                u = np.cross(t, [1.0, 0.0, 0.0])
                if np.linalg.norm(u) < 1e-6:
                    u = np.cross(t, [0.0, 1.0, 0.0])
            else:
                u = prev - (prev @ t) * t
            n = np.linalg.norm(u)
        U[i] = prev = u / n
    S = np.cross(T, U)
    return U, S


def rect_rings(P, U, S, half_width: float, u_top, u_bottom) -> np.ndarray:
    """Profile rectangles (N, 4, 3): corners at +/- half_width along S, u_top/u_bottom along U."""
    u_top = np.broadcast_to(np.asarray(u_top, float), (len(P),))
    u_bottom = np.broadcast_to(np.asarray(u_bottom, float), (len(P),))
    corners = []
    for side, which in ((-1, "b"), (1, "b"), (1, "t"), (-1, "t")):
        u = u_bottom if which == "b" else u_top
        corners.append(P + S * (side * half_width) + U * u[:, None])
    return np.stack(corners, axis=1)


def ring_pieces(rings: np.ndarray, overlap: float = 0.0) -> list[np.ndarray]:
    """Convex pieces between consecutive rings (8 points each). Their union is the swept solid.

    Unioning convex hulls instead of building one tube mesh keeps tight bends valid: folds on the
    inside of a curve just overlap, and the union removes them. ``overlap`` stretches each piece
    a little past its rings so neighbours overlap instead of merely touching (more robust booleans).
    """
    out = []
    for i in range(len(rings) - 1):
        a, b = rings[i], rings[i + 1]
        if overlap > 0:
            step = b.mean(0) - a.mean(0)
            n = np.linalg.norm(step)
            if n > 1e-12:
                step = step / n * overlap
                a, b = a - step, b + step
        out.append(np.vstack([a, b]))
    return out


def tube_folds(rings: np.ndarray) -> bool:
    """True if some profile corner moves backwards between consecutive rings, i.e. the swept tube
    would fold through itself (bend tighter than the profile is wide)."""
    centres = rings.mean(axis=1)
    axis = np.diff(centres, axis=0)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    corner_steps = np.diff(rings, axis=0)  # (N-1, 4, 3)
    return bool(((corner_steps * axis[:, None, :]).sum(-1) <= 0).any())


def tube_mesh(rings: np.ndarray) -> Manifold:
    """One closed mesh through all rings (N, 4, 3): 8 triangles per segment plus two caps.

    Far fewer triangles than unioning segment hulls, and none of the near-coplanar overlaps that
    leave degenerate slivers. The caller checks ``tube_folds`` first."""
    n = len(rings)
    verts = rings.reshape(-1, 3)
    idx = np.arange(n * 4).reshape(n, 4)
    tris = []
    for i in range(n - 1):
        for k in range(4):
            a, b = idx[i, k], idx[i, (k + 1) % 4]
            c, d = idx[i + 1, (k + 1) % 4], idx[i + 1, k]
            tris += [(a, b, c), (a, c, d)]
    tris += [(idx[0, 0], idx[0, 2], idx[0, 1]), (idx[0, 0], idx[0, 3], idx[0, 2])]
    tris += [(idx[-1, 0], idx[-1, 1], idx[-1, 2]), (idx[-1, 0], idx[-1, 2], idx[-1, 3])]
    tris = np.array(tris, dtype=np.uint32)
    # orientation depends on the ring winding; fix it with the signed volume
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    if np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() < 0:
        tris = tris[:, [0, 2, 1]]
    return Manifold(Mesh64(np.ascontiguousarray(verts, dtype=np.float64), np.ascontiguousarray(tris)))
