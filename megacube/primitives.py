"""Procedural primitives: boxes, convex hulls and extruded profiles for the body, plus light
elements (belt sweeps, sign panels, lifts) that can be attached to the surface and get windows.

No game meshes or assets are used, only rule parameters (config/classes.yaml) and the
transform/spline data from the build file.

Rule parameters are in game-local cm; ``local_to_model`` turns them into model-local mm
(mirroring Y, so asymmetric shapes keep their handedness, see megacube/coords.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from manifold3d import CrossSection, FillRule, Manifold, OpType

from .coords import CM_TO_MM, local_to_model
from .mapping import Rule
from .model import BuildObject
from .sweep import profile_frames, rect_rings, ring_pieces, sample_spline, tube_folds, tube_mesh


# ------------------------------------------------------------------------- manifold helpers
def transform(solid: Manifold, matrix4: np.ndarray) -> Manifold:
    return solid.transform(np.asarray(matrix4, float)[:3, :4])


def hull(points: np.ndarray) -> Manifold | None:
    """Convex hull, or None for degenerate (flat/empty) point sets."""
    pts = np.asarray(points, float)
    if len(pts) < 4:
        return None
    solid = Manifold.hull_points(pts)
    if solid.is_empty() or solid.volume() <= 1e-9 * max(np.ptp(pts, axis=0).max(), 1e-9) ** 3:
        return None
    return solid


def union(solids) -> Manifold:
    solids = [s for s in solids if s is not None and not s.is_empty()]
    if not solids:
        return Manifold()
    return Manifold.batch_boolean(solids, OpType.Add)


def box(size, center=(0.0, 0.0, 0.0)) -> Manifold:
    return Manifold.cube(tuple(float(v) for v in size), True).translate(tuple(float(c) for c in center))


# ------------------------------------------------------------------------- body primitives
def body_primitive(rule: Rule, weld: float = 0.0) -> Manifold:
    """The rule's shape in model-local mm, positioned by its offset. ``weld`` grows each side."""
    size = np.asarray(rule.params["size"], float) * CM_TO_MM
    grown = size + 2.0 * weld
    centre = local_to_model(rule.offset())
    if rule.primitive == "box":
        return box(grown, centre)
    if rule.primitive == "hull":
        pts = local_to_model(np.asarray(rule.params["points"], float) * (grown / CM_TO_MM)) + centre
        solid = hull(pts)
        if solid is None:
            raise ValueError(f"rule {rule.name}: point set is degenerate")
        return solid
    if rule.primitive == "extrude":
        prof = np.asarray(rule.params["profile"], float) * grown[[0, 2]]
        section = CrossSection([prof], FillRule.NonZero)
        # extrude along +Z, then rotate +90 deg about X: (u, v, w) -> (u, -w, v); profile lands in X/Z
        solid = Manifold.extrude(section, float(grown[1])).rotate((90.0, 0.0, 0.0))
        return solid.translate((float(centre[0]), float(centre[1] + grown[1] / 2), float(centre[2])))
    raise ValueError(f"rule {rule.name}: {rule.primitive} is not a body primitive")


def placeholder(size_cm) -> Manifold:
    return box(np.asarray(size_cm, float) * CM_TO_MM)


# ------------------------------------------------------------------------- light elements
def window_half(half: float, inset: float, min_width: float) -> float:
    """Half-width of a window behind an element of half-width ``half``: ``inset`` smaller per side,
    but never narrower than ``min_width`` (a narrower light pipe would be a thin wall itself), as
    long as the element still overlaps it by a fifth of the inset on each side. Returns 0 when
    no printable window fits."""
    want = half - inset
    if 2 * want >= min_width:
        return want
    if 0 < min_width / 2 <= half - 0.2 * inset:
        return min_width / 2
    return 0.0


def swept_solids(rings: np.ndarray, overlap: float) -> list[Manifold]:
    """A swept profile as one clean tube mesh, or as overlapping convex pieces if it would fold."""
    if len(rings) >= 2 and not tube_folds(rings):
        solid = tube_mesh(rings)
        if not solid.is_empty() and solid.volume() > 0:
            return [solid]
    return [h for h in (hull(p) for p in ring_pieces(rings, overlap)) if h is not None]


@dataclass
class LightElement:
    obj_id: str
    class_name: str
    rule: str
    kind: str
    notes: list[str] = field(default_factory=list)
    attach_gap: float | None = None  # true-scale mm, measured before attaching
    window_depth: float | None = None

    def solids(self) -> list[Manifold]:
        raise NotImplementedError

    def points(self) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SweepElement(LightElement):
    """A belt: rectangular profile swept along a sampled spline (world coordinates)."""
    P: np.ndarray = None  # (N,3) spline samples
    U: np.ndarray = None  # (N,3) profile up (outward, away from the surface it rests on)
    S: np.ndarray = None  # (N,3) profile side
    half_width: float = 0.0
    u_top: float = 0.0
    u_bottom: np.ndarray = None  # (N,) per sample, relative to P along U

    def rings(self) -> np.ndarray:
        return rect_rings(self.P, self.U, self.S, self.half_width, self.u_top, self.u_bottom)

    def points(self) -> np.ndarray:
        return self.rings().reshape(-1, 3)

    def solids(self) -> list[Manifold]:
        return swept_solids(self.rings(), overlap=1e-3 * self.half_width)

    def back_samples(self):
        """Ray origins on the underside (3 across the width per sample) and inward directions."""
        offs = np.array([-0.8, 0.0, 0.8]) * self.half_width
        o = self.P[:, None, :] + self.S[:, None, :] * offs[None, :, None] + (self.U * self.u_bottom[:, None])[:, None, :]
        d = np.repeat(-self.U[:, None, :], 3, axis=1)
        return o.reshape(-1, 3), d.reshape(-1, 3), np.repeat(np.arange(len(self.P)), 3)

    def apply_attach(self, sample_gap: np.ndarray, mode: str, embed: float) -> None:
        """``sample_gap``: per spline sample, distance from the underside to the body (inf = none)."""
        finite = np.isfinite(sample_gap)
        if not finite.any() or mode == "none":
            return
        shift = 0.0
        if mode == "snap":
            shift = float(sample_gap[finite].min())
            self.P = self.P - self.U * shift
        residual = np.where(finite, sample_gap - shift, 0.0)
        self.u_bottom = self.u_bottom - (residual + embed)  # reach the surface everywhere, then sink in

    def window_solids(self, depth: np.ndarray, inset: float, overlap: float, min_width: float = 0.0) -> list[Manifold]:
        """A slot behind the belt, ``inset`` narrower per side, swept like the belt itself.

        The depth follows the cavity sample by sample (a single tube mesh, so varying depth is
        safe). Runs of samples where the cavity is out of reach are left without a slot."""
        hw = window_half(self.half_width, inset, min_width)
        ok = np.isfinite(depth)
        if hw <= 0 or not ok.any():
            return []
        top = self.u_bottom + overlap
        bottom = self.u_bottom - np.where(ok, depth, 0.0) - overlap
        rings = rect_rings(self.P, self.U, self.S, hw, top, bottom)
        out, start = [], None
        for i in range(len(ok) + 1):  # one tube per contiguous run of reachable samples
            if i < len(ok) and ok[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start >= 2:
                    out += swept_solids(rings[start:i], overlap=1e-3 * self.half_width)
                start = None
        return out


@dataclass
class BoxElement(LightElement):
    """A sign panel or a lift: an oriented box (world coordinates).

    ``axes`` columns are unit vectors; ``half`` the half-extents along them. ``inward`` is
    (axis index, sign): the direction from the element toward the body it's mounted on.
    """
    centre: np.ndarray = None
    axes: np.ndarray = None
    half: np.ndarray = None
    inward: tuple[int, int] | None = None

    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        return self.centre + (signs * self.half) @ self.axes.T

    def points(self) -> np.ndarray:
        return self.corners()

    def solids(self) -> list[Manifold]:
        return [hull(self.corners())]

    def inward_dir(self) -> np.ndarray:
        k, sgn = self.inward
        return self.axes[:, k] * sgn

    def back_samples(self, grid: int = 3):
        k, _ = self.inward
        a, b = [i for i in range(3) if i != k]
        d = self.inward_dir()
        uu, vv = np.meshgrid(np.linspace(-0.8, 0.8, grid), np.linspace(-0.8, 0.8, grid))
        o = (self.centre + d * self.half[k] + np.outer(uu.ravel() * self.half[a], self.axes[:, a])
             + np.outer(vv.ravel() * self.half[b], self.axes[:, b]))
        return o, np.repeat(d[None, :], len(o), axis=0), np.zeros(len(o), int)

    def apply_attach(self, sample_gap: np.ndarray, mode: str, embed: float) -> None:
        g = sample_gap[np.isfinite(sample_gap)]
        if not len(g) or mode == "none":
            return
        k, _ = self.inward
        d = self.inward_dir()
        if mode == "snap":  # move until the closest point touches, then sink in by `embed`
            self.centre = self.centre + d * (float(g.min()) + embed)
            residual = float(g.max() - g.min())  # e.g. a panel over a step: grow into the rest
            if residual > 0:
                self.half[k] += residual / 2
                self.centre = self.centre + d * residual / 2
        else:  # extend: grow the back face down to the surface (+ embed)
            grow = float(g.max()) + embed
            self.half[k] += grow / 2
            self.centre = self.centre + d * grow / 2

    def window_solids(self, depth: np.ndarray, inset: float, overlap: float, min_width: float = 0.0) -> list[Manifold]:
        finite = depth[np.isfinite(depth)]
        if not len(finite):
            return []
        dmax = float(finite.max())
        k, _ = self.inward
        a, b = [i for i in range(3) if i != k]
        ha, hb = window_half(self.half[a], inset, min_width), window_half(self.half[b], inset, min_width)
        if ha <= 0 or hb <= 0:
            return []
        d = self.inward_dir()
        back = self.centre + d * (self.half[k] - overlap)
        pts = []
        for sa in (-ha, ha):
            for sb in (-hb, hb):
                p = back + self.axes[:, a] * sa + self.axes[:, b] * sb
                pts += [p, p + d * (dmax + 2 * overlap)]
        return [hull(np.array(pts))]


def _world_axes(m4: np.ndarray, local_axes: np.ndarray):
    """Unit world axes and the scale along each, for model-local axis vectors (columns)."""
    w = m4[:3, :3] @ local_axes
    lengths = np.linalg.norm(w, axis=0)
    return w / lengths, lengths


def make_light_element(obj: BuildObject, rule: Rule, m4: np.ndarray, *, min_width: float,
                       min_thickness: float, max_segment: float, max_angle_deg: float) -> LightElement:
    """Build a light element in world (recentred, true-scale mm) coordinates.

    ``min_width``/``min_thickness`` are model-mm floors: printed minimums divided by the scale.
    """
    p = rule.params
    common = dict(obj_id=obj.id, class_name=obj.class_name, rule=rule.name)
    if rule.primitive == "belt":
        if not obj.spline or len(obj.spline) < 2:
            raise ValueError(f"{obj.id}: belt without spline data")
        loc = np.array([s.location for s in obj.spline])
        arr = np.array([s.arrive_tangent for s in obj.spline])
        lea = np.array([s.leave_tangent for s in obj.spline])
        P, T = sample_spline(loc, arr, lea, max_segment, max_angle_deg)
        U, S = profile_frames(T, (0.0, 0.0, 1.0))  # belts rest on their local XY plane
        R = m4[:3, :3]
        Pw = P @ R.T + m4[:3, 3]
        Uw = U @ R.T
        Uw /= np.linalg.norm(Uw, axis=1, keepdims=True)
        Sw = S @ R.T
        Sw /= np.linalg.norm(Sw, axis=1, keepdims=True)
        width = float(p["width"]) * CM_TO_MM
        thick = float(p["thickness"]) * CM_TO_MM
        el = SweepElement(**common, kind="belt", P=Pw, U=Uw, S=Sw,
                          half_width=max(width, min_width) / 2, u_top=float(p.get("top_offset", 0.0)) * CM_TO_MM,
                          u_bottom=np.full(len(Pw), float(p.get("top_offset", 0.0)) * CM_TO_MM - max(thick, min_thickness)))
        if width < min_width:
            el.notes.append(f"width exaggerated {width:.0f} -> {min_width:.0f} model mm")
        if thick < min_thickness:
            el.notes.append(f"thickness exaggerated {thick:.0f} -> {min_thickness:.0f} model mm")
        return el

    if rule.primitive == "panel":
        w, h = (float(v) * CM_TO_MM for v in p["size"])
        t = float(p["thickness"]) * CM_TO_MM
        # model-local: width along X, thickness along Y, height along Z (front = game +Y = model -Y)
        axes, scale = _world_axes(m4, np.eye(3))
        half = np.array([max(w, min_width) / 2, max(t, min_thickness) / 2, max(h, min_width) / 2]) * scale
        centre = (m4 @ np.r_[local_to_model(rule.offset()), 1.0])[:3]
        el = BoxElement(**common, kind="panel", centre=centre, axes=axes, half=half, inward=(1, +1))
        if t < min_thickness:
            el.notes.append(f"thickness exaggerated {t:.0f} -> {min_thickness:.0f} model mm")
        return el

    if rule.primitive == "lift":
        fx, fy = (float(v) * CM_TO_MM for v in p["footprint"])
        height = float(obj.lift_top.translation[2]) if obj.lift_top else 0.0
        axes, scale = _world_axes(m4, np.eye(3))
        half = np.array([max(fx, min_width) / 2, max(fy, min_width) / 2, max(abs(height), min_width) / 2]) * scale
        centre = (m4 @ np.r_[local_to_model(rule.offset()) + np.array([0.0, 0.0, height / 2]), 1.0])[:3]
        el = BoxElement(**common, kind="lift", centre=centre, axes=axes, half=half, inward=None)
        if not obj.lift_top:
            el.notes.append("no mTopTransform: lift height unknown, drawn as a minimal block")
        return el
    raise ValueError(f"rule {rule.name}: {rule.primitive} is not a light primitive")
