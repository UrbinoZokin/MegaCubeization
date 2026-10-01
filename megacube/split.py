"""Phase 4: split the scaled model into printable parts, add alignment pins/sockets, orient flat.

Modes (``split.mode``):

* ``one``: a single part, printed bottom-down. Pair it with ``hollow.roof: pyramid`` (the
  default ``roof: auto`` does) so the closed cavity needs no internal supports.
* ``box_lid``: cut at the cavity ceiling. The box prints upright (open top). The lid is a flat
  plate printed inner face down. Square posts in the box corners carry round pins, and the lid gets
  matching sockets with clearance.
* ``panels``: six panels with butt joints. Top and bottom are full size, front/back sit between
  them, left/right between front/back. Each panel prints inner face down. Pins stick out of the
  joint edges horizontally, so they have a diamond cross-section (self-supporting 45 degree
  flanks); sockets go into the neighbouring panel's inner face.

All solids of one part share one transform, so their STLs line up in the slicer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from manifold3d import Manifold

from .config import Config
from .geometry import Geometry, drop_slivers
from .primitives import box, hull, union

log = logging.getLogger(__name__)
# Cut planes lie exactly on the cavity faces. manifold3d resolves the coplanar cuts cleanly;
# nudging them into the cavity instead left 1e-4 mm steps that broke the thin-wall check.
EPS = 0.0
BIG = 1e4


@dataclass
class Part:
    name: str
    solids: dict[str, Manifold]  # print orientation, printed mm
    to_print: np.ndarray  # 4x4: assembled -> print coordinates
    notes: list[str] = field(default_factory=list)
    pins: list[tuple[Manifold, str]] = field(default_factory=list)  # (pin, receiving part), assembled coords

    @property
    def to_assembled(self) -> np.ndarray:
        return np.linalg.inv(self.to_print)

    def bbox(self) -> np.ndarray:
        boxes = np.array([s.bounding_box() for s in self.solids.values() if not s.is_empty()])
        return np.r_[boxes[:, :3].min(0), boxes[:, 3:].max(0)]


def _region(lo, hi) -> Manifold:
    lo = np.where(np.isinf(lo), -BIG, lo)
    hi = np.where(np.isinf(hi), BIG, hi)
    return box(hi - lo, (lo + hi) / 2)


def _apply(m4: np.ndarray, solid: Manifold) -> Manifold:
    return solid.transform(m4[:3, :4])


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4)
    if axis == "x":
        m[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def _settle(m4: np.ndarray, solids: dict[str, Manifold]) -> np.ndarray:
    """Append a translation so the part sits on z = 0, centred on the XY origin."""
    boxes = np.array([_apply(m4, s).bounding_box() for s in solids.values() if not s.is_empty()])
    lo, hi = boxes[:, :3].min(0), boxes[:, 3:].max(0)
    t = np.eye(4)
    t[:3, 3] = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
    return t @ m4


def round_pin(base, length: float, diameter: float, chamfer: float = 0.3) -> Manifold:
    r = diameter / 2
    body = Manifold.cylinder(max(length - chamfer, 1e-3), r, r, 48)
    tip = Manifold.cylinder(chamfer, r, max(r - chamfer, r * 0.4), 48).translate((0, 0, max(length - chamfer, 1e-3)))
    return (body + tip).translate((float(base[0]), float(base[1]), float(base[2])))


def diamond_pin(base, axis, u, v, diag: float, length: float, taper: float = 0.85) -> Manifold:
    """Pin with a square cross-section rotated 45 degrees: diagonals along ``u`` (vertical when
    printed) and ``v``, so its flanks are 45 degree slopes that print without support."""
    base, axis, u, v = (np.asarray(x, float) for x in (base, axis, u, v))
    pts = []
    for scale, offset in ((1.0, -0.05), (taper, length)):
        c = base + axis * offset
        h = diag / 2 * scale
        pts += [c + u * h, c - u * h, c + v * h, c - v * h]
    return hull(np.array(pts))


@dataclass
class Joint:
    pin_part: str
    socket_part: str
    base: np.ndarray  # on the joint plane, at the INNER face of the pin panel's wall, mid-edge
    axis: np.ndarray  # pin direction, into the socket panel
    u: np.ndarray  # pin-panel thickness direction, pointing outward (inner face -> outer face)
    v: np.ndarray  # along the edge
    edge_len: float


def _core_and_inner(geo: Geometry, cfg: Config):
    """Outer core box (cube faces) and inner box (cavity faces), printed mm."""
    core = geo.core_box if geo.core_box is not None else np.array(geo.filled.bounding_box()).reshape(2, 3)
    if geo.cavity is not None and not geo.cavity.is_empty():
        inner = np.array(geo.cavity.bounding_box()).reshape(2, 3)
    else:
        w = float(cfg.get_path("hollow.wall"))
        inner = core + np.array([[w, w, w], [-w, -w, -w]])
    return core, inner


def split_geometry(geo: Geometry, cfg: Config) -> tuple[list[Part], dict]:
    mode = cfg.get_path("split.mode")
    sc = cfg["split"]
    groups = geo.groups()
    rep: dict = {"mode": mode, "warnings": []}

    def warn(msg):
        log.warning(msg)
        rep["warnings"].append(msg)

    if mode == "one":
        if geo.report.get("roof") == "flat" and geo.cavity is not None:
            warn("one-piece print with a flat cavity ceiling needs internal supports; use hollow.roof=pyramid")
        m4 = _settle(np.eye(4), groups)
        return [Part("cube", {g: _apply(m4, s) for g, s in groups.items()}, m4)], rep

    core, inner = _core_and_inner(geo, cfg)
    wall = float(np.min(np.r_[inner[0] - core[0], core[1] - inner[1]]))
    clearance, skin = float(sc["clearance"]), float(sc["min_skin"])
    max_len = wall - skin - 0.3  # socket = pin + 0.3 mm must leave `skin` before the outer surface
    pin_len = min(float(sc["pin_length"]), max_len)
    if pin_len < 0.8:
        warn(f"walls too thin for alignment pins (wall {wall:.2f} mm): pins skipped")
        pin_len = 0.0
    elif pin_len < float(sc["pin_length"]):
        rep["pin_length_clamped_mm"] = round(pin_len, 3)
    light_guard = groups["light"]

    if mode == "box_lid":
        return _box_lid(geo, groups, core, inner, sc, pin_len, clearance, light_guard, rep, warn)
    if mode == "panels":
        return _panels(groups, core, inner, wall, sc, pin_len, clearance, light_guard, rep, warn)
    raise ValueError(f"unknown split.mode {mode!r} (one | box_lid | panels)")


def _box_lid(geo, groups, core, inner, sc, pin_len, clearance, light_guard, rep, warn):
    z_cut = inner[1, 2] - EPS
    below = _region(np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, z_cut]))
    above = _region(np.array([-np.inf, -np.inf, z_cut]), np.array([np.inf, np.inf, np.inf]))
    box_solids = {g: s ^ below for g, s in groups.items()}
    lid_solids = {g: s ^ above for g, s in groups.items()}

    post = float(sc["corner_post"])
    pin_d = float(sc["pin_size"])
    posts, pins, sockets = [], [], []
    for sx in (0, 1):
        for sy in (0, 1):
            x0 = inner[0, 0] - 0.01 if sx == 0 else inner[1, 0] - post
            y0 = inner[0, 1] - 0.01 if sy == 0 else inner[1, 1] - post
            lo = np.array([x0, y0, inner[0, 2] - 0.01])
            hi = np.array([x0 + post + 0.01, y0 + post + 0.01, z_cut])
            posts.append(_region(lo, hi))
            c = np.array([x0 + post / 2 + (0.005 if sx == 0 else 0), y0 + post / 2 + (0.005 if sy == 0 else 0), z_cut])
            if pin_len > 0:
                pins.append(round_pin(c - np.array([0, 0, 0.05]), pin_len + 0.05, pin_d))
                sockets.append(Manifold.cylinder(pin_len + 0.3 + 0.1, pin_d / 2 + clearance, pin_d / 2 + clearance, 48)
                               .translate((float(c[0]), float(c[1]), float(z_cut - 0.1))))
    posts_u = union(posts) - light_guard
    if posts_u.min_gap(light_guard, 0.05) < 0.05:
        warn("a corner post touches a light element: check that no light pipe is covered")
    host = "dark" if "dark" in box_solids else "body"  # inner structure in the dark filament if there is one
    box_solids[host] = box_solids[host] + posts_u + union(pins)
    sock = union(sockets)
    for g in lid_solids:
        lid_solids[g] = lid_solids[g] - sock
    if pin_len > 0 and (sock ^ light_guard).volume() > 1e-6:
        warn("a lid socket cuts into a light element")
    rep["pins"] = {"count": len(pins), "diameter_mm": pin_d, "length_mm": round(pin_len, 3),
                   "socket_clearance_mm": clearance, "corner_post_mm": post}

    parts = []
    for name, solids, m4 in (("box", box_solids, np.eye(4)), ("lid", lid_solids, np.eye(4))):
        solids = {g: drop_slivers(s, 1e-3)[0] for g, s in solids.items()}
        solids = {g: s for g, s in solids.items() if s is not None and not s.is_empty()}
        m4 = _settle(m4, solids)
        parts.append(Part(name, {g: _apply(m4, s) for g, s in solids.items()}, m4,
                          pins=[(pin, "lid") for pin in pins] if name == "box" else []))
    return parts, rep


PANEL_INWARD = {"top": (0, 0, -1), "bottom": (0, 0, 1), "front": (0, 1, 0), "back": (0, -1, 0),
                "left": (1, 0, 0), "right": (-1, 0, 0)}
PANEL_ROT = {"top": np.eye(4), "bottom": _rot("x", 180), "front": _rot("x", -90), "back": _rot("x", 90),
             "left": _rot("y", 90), "right": _rot("y", -90)}


def _panel_regions(inner):
    (x0, y0, z0), (x1, y1, z1) = inner
    inf = np.inf
    x0e, x1e, y0e, y1e, z0e, z1e = x0 + EPS, x1 - EPS, y0 + EPS, y1 - EPS, z0 + EPS, z1 - EPS
    return {
        "top": ((-inf, -inf, z1e), (inf, inf, inf)),
        "bottom": ((-inf, -inf, -inf), (inf, inf, z0e)),
        "front": ((-inf, -inf, z0e), (inf, y0e, z1e)),
        "back": ((-inf, y1e, z0e), (inf, inf, z1e)),
        "left": ((-inf, y0e, z0e), (x0e, y1e, z1e)),
        "right": ((x1e, y0e, z0e), (inf, y1e, z1e)),
    }


def _panel_joints(core, inner) -> list[Joint]:
    (X0, Y0, Z0), (X1, Y1, Z1) = core
    (x0, y0, z0), (x1, y1, z1) = inner
    ex, ey, ez = np.eye(3)
    joints = []
    # front/back: top and bottom edges, full width
    for name, y_in, out in (("front", y0, -ey), ("back", y1, ey)):
        for other, z, ax in (("top", z1 - EPS, ez), ("bottom", z0 + EPS, -ez)):
            joints.append(Joint(name, other, np.array([(X0 + X1) / 2, y_in, z]), ax, out, ex, X1 - X0))
    # left/right: top and bottom edges (between front and back) and the two vertical edges
    for name, x_in, out in (("left", x0, -ex), ("right", x1, ex)):
        for other, z, ax in (("top", z1 - EPS, ez), ("bottom", z0 + EPS, -ez)):
            joints.append(Joint(name, other, np.array([x_in, (y0 + y1) / 2, z]), ax, out, ey, y1 - y0))
        for other, y, ax in (("front", y0 + EPS, -ey), ("back", y1 - EPS, ey)):
            joints.append(Joint(name, other, np.array([x_in, y, (z0 + z1) / 2]), ax, out, ez, z1 - z0))
    return joints


def _panels(groups, core, inner, wall, sc, pin_len, clearance, light_guard, rep, warn):  # noqa: C901
    regions = {k: _region(np.array(lo, float), np.array(hi, float)) for k, (lo, hi) in _panel_regions(inner).items()}
    solids = {k: {g: s ^ r for g, s in groups.items()} for k, r in regions.items()}

    # The pin sits against the cavity side of the wall; the socket in the neighbouring panel then
    # only needs `min_skin` towards the outside (the inner side is solid panel), so:
    #   pin diagonal + sqrt(2) * clearance + skin <= wall
    skin = float(sc["min_skin"])
    diag = min(float(sc["pin_size"]), wall - skin - math.sqrt(2) * clearance)
    n_pins = int(sc["pins_per_edge"])
    pins: dict[str, list] = {k: [] for k in regions}
    sockets: dict[str, list] = {k: [] for k in regions}
    unplaced = 0
    if pin_len > 0 and diag >= 0.8 and n_pins > 0:
        sock_diag = diag + 2 * math.sqrt(2) * clearance
        for j in _panel_joints(core, inner):
            want = [(i + 0.5) / n_pins - 0.5 for i in range(n_pins)]  # e.g. -0.25, +0.25 of the edge
            for f in want:
                placed = False
                for shift in (0.0, 0.08, -0.08, 0.16, -0.16):  # slide along the edge to dodge light
                    base = j.base + j.v * (f + shift) * j.edge_len * 0.9 + j.u * (diag / 2)
                    sock = diamond_pin(base, j.axis, j.u, j.v, sock_diag, pin_len + 0.3, taper=1.0)
                    if (sock ^ light_guard).volume() > 1e-6:
                        continue
                    pins[j.pin_part].append((diamond_pin(base, j.axis, j.u, j.v, diag, pin_len), j.socket_part))
                    sockets[j.socket_part].append(sock)
                    placed = True
                    break
                unplaced += not placed
    else:
        warn(f"pins skipped (wall {wall:.2f} mm too thin for a {sc['pin_size']} mm diamond pin)")
    if unplaced:
        warn(f"{unplaced} pin(s) could not be placed without cutting into light elements")
    rep["pins"] = {"count": sum(len(v) for v in pins.values()), "diamond_diagonal_mm": round(diag, 3),
                   "length_mm": round(pin_len, 3), "socket_clearance_mm": clearance}

    parts = []
    for name in ("top", "bottom", "front", "back", "left", "right"):
        sol = dict(solids[name])
        if pins[name]:
            sol["body"] = sol["body"] + union([pin for pin, _ in pins[name]])
        if sockets[name]:
            sk = union(sockets[name])
            sol = {g: s - sk for g, s in sol.items()}
        sol = {g: drop_slivers(s, 1e-3)[0] for g, s in sol.items()}
        sol = {g: s for g, s in sol.items() if s is not None and not s.is_empty()}
        if not sol:
            continue
        m4 = _settle(PANEL_ROT[name], sol)
        parts.append(Part(name, {g: _apply(m4, s) for g, s in sol.items()}, m4, pins=pins[name]))
    return parts, rep
