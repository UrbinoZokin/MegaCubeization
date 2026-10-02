"""Snap-fit clips and sliding keys that lock the six panels together (``split.panel_joint: snap``).

Every connector is drawn in a local frame and then placed on a joint:

* ``a``: insertion direction, from the panel that carries the connector into the one that receives
  it; ``a = 0`` is the joint plane;
* ``u``: thickness direction of the carrying panel, pointing outward; ``u = 0`` is its inner face;
* ``v``: along the edge (``v = u x a``, so the frame is right-handed).

**Clip** (on the top and bottom edges of the four side panels). A flat, split arrowhead: two prongs
side by side with a slot between them, each with a barb on its outer edge. The panel prints inner
face down, so the clip lies on the bed (``u`` from 0 to ``thickness``) and the prongs bend sideways,
parallel to the layers (the strong direction). The prongs start inside the panel, in a relief
pocket open towards the cavity, so they're long enough to bend without breaking. Their free length
comes from the allowed strain: for a cantilever, strain = 1.5 * width * deflection / length^2.

**Socket** (in the top/bottom panel's inner face). A narrow neck the prongs pass through,
then a wider chamber. The barbs spring out in the chamber and their square catch faces hook behind
the neck's lips. Once engaged, every face of the clip keeps ``clearance`` to the socket, so the
assembled model has no overlaps.

**Key** (on the vertical edges of the left and right panels). A short tongue that slides down a
groove in the front/back panel's inner face while the panel is pressed down. Keys keep the vertical
seams flush; they don't lock anything (the clips do).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection, Manifold

from .primitives import box, hull

ROOT_OVERLAP = 0.5  # prong/tongue roots reach this far into solid panel material, so they fuse
PLA_MODULUS_MPA = 3500.0  # for the rough insertion-force estimate only
FRICTION = 0.3


def _box(lo, hi) -> Manifold:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box(hi - lo, (lo + hi) / 2)


@dataclass(frozen=True)
class Frame:
    """A joint site: local (a, v, u) coordinates -> assembled printed mm."""

    origin: np.ndarray
    a: np.ndarray
    u: np.ndarray

    @property
    def v(self) -> np.ndarray:
        return np.cross(self.u, self.a)

    def matrix(self) -> np.ndarray:
        return np.column_stack([self.a, self.v, self.u, self.origin]).astype(float)

    def place(self, solid: Manifold) -> Manifold:
        return solid.transform(self.matrix())

    def point(self, a: float, v: float, u: float) -> np.ndarray:
        return self.origin + self.a * a + self.v * v + self.u * u


@dataclass(frozen=True)
class SnapSpec:
    thickness: float  # clip thickness (u), printed flat on the bed
    prong: float  # width of each prong (v): the bending direction
    split: float  # slot between the prongs; they bend into it
    barb: float  # how far each barb sticks out past its prong
    neck: float  # length of the socket neck = thickness of the lip the barb hooks behind
    lead_in: float  # ramp from the barb down to the tip
    max_strain: float  # sets the prong length (PLA ~2 %, PETG ~3 %)
    relief: float  # slots around and above the prongs inside their panel
    clearance: float  # every clip face to the socket, once engaged
    skin: float  # material left behind a socket or above a relief pocket
    land: float = 0.4  # flat top of the barb: one 0.4 mm nozzle line
    tip_in: float = 0.3  # tip chamfer (narrower than the prong: finds the neck by itself)
    tip_gap: float = 0.25  # tip to the bottom of the socket

    @classmethod
    def from_config(cls, cfg) -> "SnapSpec":
        s = cfg.get_path("split.snap")
        return cls(thickness=float(s["thickness"]), prong=float(s["prong"]), split=float(s["split"]),
                   barb=float(s["barb"]), neck=float(s["neck"]), lead_in=float(s["lead_in"]),
                   max_strain=float(s["max_strain"]), relief=float(s["relief"]),
                   clearance=float(cfg.get_path("split.clearance")), skin=float(cfg.get_path("split.min_skin")))

    # ------------------------------------------------------------------ derived dimensions
    @property
    def width(self) -> float:
        return 2 * self.prong + self.split

    @property
    def deflection(self) -> float:
        """How far each prong bends inward while its barb passes the neck."""
        return self.barb - self.clearance

    @property
    def catch_at(self) -> float:
        """Position of the catch face, one clearance past the lip."""
        return self.neck + self.clearance

    @property
    def out_length(self) -> float:
        """How far the clip sticks out of its panel's edge."""
        return self.catch_at + self.land + self.lead_in

    @property
    def lever(self) -> float:
        """Prong length from its root to the barb, from the strain limit."""
        return math.sqrt(1.5 * self.prong * max(self.deflection, 1e-9) / self.max_strain)

    @property
    def in_length(self) -> float:
        """How deep the prong roots sit inside the panel (the relief pocket's length)."""
        return max(self.lever - self.catch_at - self.land / 2, 0.0)

    @property
    def depth(self) -> float:
        """Socket depth."""
        return self.out_length + self.tip_gap

    @property
    def strain(self) -> float:
        lever = self.in_length + self.catch_at + self.land / 2
        return 1.5 * self.prong * self.deflection / lever ** 2

    def required_wall(self) -> float:
        """Thinnest panel wall that fits a socket (depth + skin) and a relief pocket (clip + slit + skin)."""
        return max(self.depth + self.skin, self.thickness + self.relief + self.skin)

    def insertion_force_n(self) -> float:
        """Rough push force per clip (two prongs), PLA, with friction on the lead-in ramp."""
        inertia = self.thickness * self.prong ** 3 / 12
        lever = self.in_length + self.catch_at + self.land / 2
        lateral = 3 * PLA_MODULUS_MPA * inertia * self.deflection / lever ** 3
        tan_a = (self.barb + self.tip_in) / self.lead_in
        return 2 * lateral * (FRICTION + tan_a) / max(1 - FRICTION * tan_a, 0.1)

    def problems(self, min_gap: float) -> list[str]:
        out = []
        if self.deflection < 0.2:
            out.append(f"barb {self.barb} mm minus clearance {self.clearance} mm leaves only "
                       f"{self.deflection:.2f} mm of catch: the clips won't hold")
        if self.split < 2 * self.deflection + 0.2:
            out.append(f"slot between the prongs ({self.split} mm) is too narrow for both to bend "
                       f"{self.deflection:.2f} mm inward")
        if self.relief < min_gap:
            out.append(f"relief slots ({self.relief} mm) are narrower than checks.min_gap ({min_gap} mm): "
                       "the prongs may print fused to the panel")
        if self.prong < 0.8 or self.thickness < 0.8:
            out.append("prongs thinner than 0.8 mm (two perimeters of a 0.4 mm nozzle)")
        return out

    def summary(self) -> dict:
        return {"thickness_mm": self.thickness, "width_mm": round(self.width, 3), "barb_mm": self.barb,
                "catch_mm": round(self.deflection, 3), "sticks_out_mm": round(self.out_length, 3),
                "prong_root_depth_mm": round(self.in_length, 3), "socket_depth_mm": round(self.depth, 3),
                "bending_strain": round(self.strain, 4), "required_wall_mm": round(self.required_wall(), 3),
                "insertion_force_n_est": round(self.insertion_force_n(), 1)}

    # ------------------------------------------------------------------ solids, local frame
    def clip(self) -> Manifold:
        """The two prongs (local frame), from their roots inside the panel to the tip."""
        root = -self.in_length - ROOT_OVERLAP
        hs, hw = self.split / 2, self.width / 2
        tip = self.out_length
        prong = [(root, hs), (tip, hs), (tip, hw - self.tip_in), (self.catch_at + self.land, hw + self.barb),
                 (self.catch_at, hw + self.barb), (self.catch_at, hw), (root, hw)]
        mirror = [(a, -v) for a, v in reversed(prong)]
        return CrossSection([prong, mirror]).extrude(self.thickness)

    def pocket(self) -> Manifold:
        """Relief cut from the clip's own panel: free space around and above the prongs, open
        towards the cavity and through the panel's edge."""
        hw = self.width / 2 + self.relief
        return _box((-self.in_length, -hw, -1.0), (1.0, hw, self.thickness + self.relief))

    def socket(self) -> Manifold:
        """Cut from the receiving panel: neck, then the wider chamber the barbs catch in."""
        c, hw = self.clearance, self.width / 2
        u0, u1 = -c, self.thickness + c
        neck = _box((-0.5, -hw - c, u0), (self.neck + 0.01, hw + c, u1))
        chamber = _box((self.neck, -hw - self.barb - c, u0), (self.depth, hw + self.barb + c, u1))
        return neck + chamber

    def keepout_carrier(self, wall: float, margin: float = 0.8) -> Manifold:
        """Region of the clip's panel that must be plain opaque material (no light pipe, no hole)."""
        hw = self.width / 2 + self.relief + margin
        return _box((-self.in_length - 1.0, -hw, 0.01), (-0.01, hw, wall - 0.01))

    def keepout_receiver(self, margin: float = 0.8) -> Manifold:
        """Region of the receiving panel that must be plain opaque material."""
        c, hw = self.clearance, self.width / 2 + self.barb + self.clearance + margin
        return _box((0.01, -hw, -c - margin), (self.depth + self.skin, hw, self.thickness + c + margin))

    def barb_zone(self) -> Manifold:
        """Box around the barbs and tip, where the clip is deliberately thinner than two lines."""
        hw = self.width / 2 + self.barb + 0.2
        return _box((self.catch_at - 0.2, -hw, -0.2), (self.out_length + 0.2, hw, self.thickness + 0.2))

    def half_extent(self, margin: float = 0.8) -> float:
        """Half the footprint along the edge, keep-out included."""
        return self.width / 2 + max(self.relief, self.barb + self.clearance) + margin


@dataclass(frozen=True)
class KeySpec:
    width: float  # tongue thickness (u), printed flat
    depth: float  # how far it reaches into the neighbouring panel (a)
    length: float  # along the seam (v)
    clearance: float
    skin: float
    chamfer: float = 0.5

    @classmethod
    def from_config(cls, cfg) -> "KeySpec":
        k = cfg.get_path("split.keys")
        return cls(width=float(k["width"]), depth=float(k["depth"]), length=float(k["length"]),
                   clearance=float(cfg.get_path("split.clearance")), skin=float(cfg.get_path("split.min_skin")))

    def required_wall(self) -> float:
        return max(self.depth + self.clearance + self.skin, self.width + self.skin)

    def tongue(self) -> Manifold:
        """Tongue with chamfered ends (so it finds the groove), flat on the bed side (u = 0)."""
        hl, ch = self.length / 2, self.chamfer
        pts = []
        for v in (-hl + ch, hl - ch):
            pts += [(a, v, u) for a in (-ROOT_OVERLAP, self.depth) for u in (0.0, self.width)]
        for v in (-hl, hl):
            pts += [(a, v, u) for a in (-ROOT_OVERLAP, self.depth - ch) for u in (0.0, self.width - ch / 2)]
        return hull(np.array(pts))

    def groove(self, v0: float, v1: float) -> Manifold:
        """Cut from the receiving panel between v0 and v1 (open at the end the tongue enters)."""
        c = self.clearance
        return _box((-0.5, v0, -c), (self.depth + c, v1, self.width + c))

    def keepout_carrier(self, wall: float) -> Manifold:
        hl = self.length / 2 + 0.5
        return _box((-1.0, -hl, 0.01), (-0.01, hl, wall - 0.01))

    def keepout_receiver(self, v0: float, v1: float, margin: float = 0.5) -> Manifold:
        c = self.clearance
        return _box((0.01, v0 - margin, -c - margin), (self.depth + c + self.skin, v1 + margin, self.width + c + margin))


def required_wall(cfg, mode: str | None = None) -> float | None:
    """Wall the configured split needs for its connectors, or None if it has no requirement."""
    mode = mode or cfg.get_path("split.mode")
    if mode != "panels" or cfg.get_path("split.panel_joint", "snap") != "snap":
        return None
    return max(SnapSpec.from_config(cfg).required_wall(), KeySpec.from_config(cfg).required_wall())
