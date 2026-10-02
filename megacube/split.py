"""Phase 4: split the scaled model into printable parts, add connectors, orient flat.

Modes (``split.mode``):

* ``panels`` (default): six panels, each printed inner face down. Top and bottom are full size,
  front/back sit between them, left/right between front/back. ``split.panel_joint`` picks the
  connectors:

  - ``snap`` (default): the four side panels carry snap clips on their top and bottom edges, which
    lock into sockets in the top and bottom panels. Sliding keys on the left/right panels' vertical
    edges run in grooves in the front/back panels and keep the vertical seams flush (geometry in
    megacube/snap.py). Every clip engages with the same downward push, so the assembly is: bottom
    panel inner face up; press front and back down onto it; slide left and right down between them;
    press the top on. Clips and keys avoid light pipes and holes by sliding along their edge.
  - ``pins``: diamond-section alignment pins on every joint (self-supporting 45 degree flanks) and
    sockets in the neighbouring panel. They only align: glue the panels.
* ``box_lid``: cut at the cavity ceiling. The box prints upright (open top). The lid is a flat
  plate printed inner face down. Square posts in the box corners carry round pins, and the lid gets
  matching sockets with clearance.
* ``one``: a single part, printed bottom-down. Pair it with ``hollow.roof: pyramid`` (the
  default ``roof: auto`` does) so the closed cavity needs no internal supports.

All solids of one part share one transform, so their STLs line up in the slicer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from manifold3d import Manifold

from .config import Config
from .geometry import Geometry, clean, drop_slivers, fix_pinches, separate_touching
from .primitives import box, hull, union
from .snap import Frame, KeySpec, SnapSpec

log = logging.getLogger(__name__)
# Cut planes sit 1 micron inside the cavity air, never exactly on a face: cuts that coincide with
# the cavity ceiling can leave zero-thickness flaps (overlapping up/down faces) after the boolean.
# The lid/panels then have a 1 micron recess over the cavity, far below one layer.
EPS = 1e-3
BIG = 1e4
FREE_TOL = 1e-3  # mm3 of non-opaque material a connector's keep-out region may contain (booleans' dust)


@dataclass
class Part:
    name: str
    solids: dict[str, Manifold]  # print orientation, printed mm
    to_print: np.ndarray  # 4x4: assembled -> print coordinates
    notes: list[str] = field(default_factory=list)
    # connectors that reach into another part: (solid in assembled coords, receiving part)
    connectors: list[tuple[Manifold, str]] = field(default_factory=list)
    role: str = "part"  # part | coupon (a small test piece, not part of the model)
    # designed small features, (lo, hi, label) in assembled coords: the checks mark findings there as expected
    zones: list[tuple[np.ndarray, np.ndarray, str]] = field(default_factory=list)

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


def _tidy(solids: dict[str, Manifold]) -> dict[str, Manifold]:
    out = {g: clean(fix_pinches(separate_touching(drop_slivers(s, 1e-3)[0])[0])[0]) for g, s in solids.items()}
    return {g: s for g, s in out.items() if s is not None and not s.is_empty()}


def _make_part(name, solids, rot, connectors=(), role="part", notes=(), zones=()) -> Part | None:
    solids = _tidy(solids)
    if not solids:
        return None
    m4 = _settle(rot, solids)
    return Part(name, {g: _apply(m4, s) for g, s in solids.items()}, m4, list(notes), list(connectors), role, list(zones))


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
    """Outer core box (cube faces), inner box (cavity faces) and the thinnest wall, printed mm."""
    core = geo.core_box if geo.core_box is not None else np.array(geo.filled.bounding_box()).reshape(2, 3)
    if geo.cavity is not None and not geo.cavity.is_empty():
        inner = np.array(geo.cavity.bounding_box()).reshape(2, 3)
    else:
        w = float(cfg.get_path("hollow.wall"))
        inner = core + np.array([[w, w, w], [-w, -w, -w]])
    wall = geo.wall if geo.wall is not None else float(np.min(np.r_[inner[0] - core[0], core[1] - inner[1]]))
    return core, inner, float(wall)


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

    core, inner, wall = _core_and_inner(geo, cfg)
    if mode == "panels" and sc.get("panel_joint", "snap") == "snap":
        rep["panel_joint"] = "snap"
        return _snap_panels(groups, inner, wall, cfg, rep, warn)
    if mode == "panels" and sc.get("panel_joint") != "pins":
        raise ValueError(f"unknown split.panel_joint {sc.get('panel_joint')!r} (snap | pins)")

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
        rep["panel_joint"] = "pins"
        return _pin_panels(groups, core, inner, wall, sc, pin_len, clearance, light_guard, rep, warn)
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
    parts = [_make_part("box", box_solids, np.eye(4), [(pin, "lid") for pin in pins]),
             _make_part("lid", lid_solids, np.eye(4))]
    return [p for p in parts if p is not None], rep


PANEL_INWARD = {"top": (0, 0, -1), "bottom": (0, 0, 1), "front": (0, 1, 0), "back": (0, -1, 0),
                "left": (1, 0, 0), "right": (-1, 0, 0)}
PANEL_ROT = {"top": np.eye(4), "bottom": _rot("x", 180), "front": _rot("x", -90), "back": _rot("x", 90),
             "left": _rot("y", 90), "right": _rot("y", -90)}
PANEL_ORDER = ("top", "bottom", "front", "back", "left", "right")


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


def _split_panels(groups, inner):
    regions = {k: _region(np.array(lo, float), np.array(hi, float)) for k, (lo, hi) in _panel_regions(inner).items()}
    return {k: {g: s ^ r for g, s in groups.items()} for k, r in regions.items()}


# ------------------------------------------------------------------------------- snap panels
SIDES = {  # side panel: (axis of its inner face, cavity side (0 = min, 1 = max), outward direction)
    "front": (1, 0, (0.0, -1.0, 0.0)),
    "back": (1, 1, (0.0, 1.0, 0.0)),
    "left": (0, 0, (-1.0, 0.0, 0.0)),
    "right": (0, 1, (1.0, 0.0, 0.0)),
}
ASSEMBLY = [
    "Lay the bottom panel on the table, inner face up.",
    "Press the front and back panels down onto it until all their bottom clips click.",
    "Slide the left and right panels down between them: their keys run in the grooves on the front/back "
    "panels' inner faces, and their bottom clips click into the bottom panel.",
    "Put the LED in through the bottom opening if you haven't yet (it stays reachable through it).",
    "Press the top panel down evenly until all its clips click. The square catches don't release: "
    "test-fit the snap coupon first.",
]


def _segment_candidates(lo: float, hi: float, step: float):
    """Positions in [lo, hi], nearest to its middle first."""
    if hi < lo:
        return []
    mid = (lo + hi) / 2
    n = int((hi - lo) / 2 / step)
    out = [mid]
    for i in range(1, n + 1):
        out += [mid + i * step, mid - i * step]
    return out


def _snap_panels(groups, inner, wall, cfg, rep, warn):  # noqa: C901
    sc = cfg["split"]
    spec, keys = SnapSpec.from_config(cfg), KeySpec.from_config(cfg)
    for msg in spec.problems(float(cfg.get_path("checks.min_gap"))):
        warn(f"snap clips: {msg}")
    solids = _split_panels(groups, inner)
    opaque = union([s for g, s in groups.items() if g != "light"])
    host = "dark" if "dark" in groups else "body"  # connectors in one filament: the one at the inner face

    def free(*regions) -> bool:
        return (union(regions) - opaque).volume() < FREE_TOL

    add = {k: [] for k in PANEL_ORDER}  # clips and tongues, assembled coords
    cut = {k: [] for k in PANEL_ORDER}  # relief pockets, sockets, grooves
    connectors: dict[str, list] = {k: [] for k in PANEL_ORDER}
    zones: dict[str, list] = {k: [] for k in PANEL_ORDER}
    barb = spec.barb_zone()
    sites: list[dict] = []
    unplaced, shifted, skipped_seams = [], 0, []

    # --- clips: side panels' top and bottom edges -> sockets in the top/bottom panel
    need = max(spec.required_wall(), keys.required_wall())
    n_clips = int(sc["snap"]["per_edge"])
    if wall < need - 1e-6:
        warn(f"walls ({wall:.2f} mm) are thinner than the snap clips need ({need:.2f} mm): clips and keys "
             "skipped. hollow.wall is raised automatically for split.panel_joint=snap; did hollow.enabled=false?")
        n_clips = 0
    hx = spec.half_extent()
    corner = hx + keys.width + 2 * keys.clearance + 2.0  # clear of the panel corners and the key grooves
    for name, (k, side, out) in SIDES.items():
        e = 1 - k  # the horizontal axis along the edge
        span_lo, span_hi = inner[0, e], inner[1, e]
        for other, a_dir, plane in (("top", 1.0, inner[1, 2] - EPS), ("bottom", -1.0, inner[0, 2] + EPS)):
            for i in range(n_clips):
                seg_lo = span_lo + (span_hi - span_lo) * i / n_clips
                seg_hi = span_lo + (span_hi - span_lo) * (i + 1) / n_clips
                cands = _segment_candidates(max(seg_lo + hx, span_lo + corner), min(seg_hi - hx, span_hi - corner), hx)
                placed = None
                for j, pos in enumerate(cands):
                    origin = np.zeros(3)
                    origin[k], origin[e], origin[2] = inner[side, k], pos, plane
                    f = Frame(origin, np.array([0.0, 0.0, a_dir]), np.array(out))
                    if free(f.place(spec.keepout_carrier(wall)), f.place(spec.keepout_receiver())):
                        placed, shifted = f, shifted + (j > 0)
                        break
                if placed is None:
                    unplaced.append(f"{name}/{other} #{i + 1}")
                    continue
                clip = placed.place(spec.clip())
                add[name].append(clip)
                cut[name].append(placed.place(spec.pocket()))
                cut[other].append(placed.place(spec.socket()))
                connectors[name].append((clip, other))
                zb = np.array(placed.place(barb).bounding_box()).reshape(2, 3)
                zones[name].append((zb[0], zb[1], "snap clip barb (0.4 mm land: one nozzle line, by design)"))
                sites.append({"kind": "clip", "panel": name, "into": other, "frame": placed,
                              "at_mm": placed.origin.round(2).tolist()})
    if unplaced:
        warn(f"{len(unplaced)} clip(s) not placed: every position along their edge would cut a light pipe "
             f"or hole ({', '.join(unplaced)})")

    # --- keys: left/right panels' vertical edges slide down grooves in the front/back panels
    n_keys = int(sc["keys"]["per_edge"]) if n_clips else 0
    z_lo, z_hi = inner[0, 2], inner[1, 2]
    height = z_hi - z_lo
    for name in ("left", "right"):
        k, side, out = SIDES[name]
        u = np.array(out)
        for other, a_vec, plane in (("front", (0.0, -1.0, 0.0), inner[0, 1] + EPS),
                                    ("back", (0.0, 1.0, 0.0), inner[1, 1] - EPS)):
            if n_keys <= 0:
                break
            a = np.array(a_vec)
            seam = Frame(np.array([inner[side, 0], plane, 0.0]), a, u)
            sgn = float(seam.v[2])  # local v = sgn * z
            done = False
            for squeeze in (1.0, 0.7, 0.4):  # move the keys up (shorter groove) if the groove would hit light
                zs = [z_hi - squeeze * (1 - (i + 0.5) / n_keys) * height for i in range(n_keys)]
                bottom = min(zs) - keys.length / 2 - keys.clearance
                v0, v1 = sorted((sgn * bottom, sgn * (z_hi + 1.0)))
                frames = [Frame(np.array([inner[side, 0], plane, z]), a, u) for z in zs]
                if not free(seam.place(keys.keepout_receiver(v0, v1)), *[f.place(keys.keepout_carrier(wall)) for f in frames]):
                    continue
                cut[other].append(seam.place(keys.groove(v0, v1)))
                for f in frames:
                    tongue = f.place(keys.tongue())
                    add[name].append(tongue)
                    connectors[name].append((tongue, other))
                    sites.append({"kind": "key", "panel": name, "into": other, "frame": f,
                                  "at_mm": f.origin.round(2).tolist()})
                done = True
                break
            if not done:
                skipped_seams.append(f"{name}/{other}")
    if skipped_seams:
        warn(f"no keys on {len(skipped_seams)} vertical seam(s) ({', '.join(skipped_seams)}): their grooves would "
             "cut a light pipe or hole; those seams are held by the clips only")

    # --- build the parts
    assembled = {}
    for name in PANEL_ORDER:
        sol = dict(solids[name])
        if cut[name]:
            cu = union(cut[name])
            sol = {g: s - cu for g, s in sol.items()}
        if add[name]:
            extra = union(add[name])
            sol = {g: (s + extra if g == host else s - extra) for g, s in sol.items()}
            if host not in sol:
                sol[host] = extra
        assembled[name] = sol
    parts = [p for p in (_make_part(n, assembled[n], PANEL_ROT[n], connectors[n], zones=zones[n]) for n in PANEL_ORDER) if p]

    clips = [s for s in sites if s["kind"] == "clip"]
    coupon_site = None
    if clips:
        site = next((s for s in clips if s["panel"] == "front" and s["into"] == "top"), clips[0])
        parts += _coupon(assembled, site, spec, zones[site["panel"]])
        f = site["frame"]
        coupon_site = {"panel": site["panel"], "into": site["into"], "origin": f.origin.tolist(), "a": f.a.tolist(),
                       "u": f.u.tolist(), "clip_thickness_mm": spec.thickness}

    rep["snap"] = {
        "clips": len(clips), "keys": sum(1 for s in sites if s["kind"] == "key"),
        "clips_moved_to_dodge_light": shifted, "clips_not_placed": unplaced, "seams_without_keys": skipped_seams,
        "material": host, "clip": spec.summary(),
        "key": {"width_mm": keys.width, "depth_mm": keys.depth, "length_mm": keys.length,
                "required_wall_mm": round(keys.required_wall(), 3)},
        "clearance_mm": spec.clearance, "wall_mm": round(wall, 3),
        "sites": [{k: v for k, v in s.items() if k != "frame"} for s in sites],
        "coupon_site": coupon_site,
        "assembly": ASSEMBLY,
    }
    return parts, rep


def _coupon(assembled: dict, site: dict, spec: SnapSpec, zones=()) -> list[Part]:
    """A small test pair cut from the real panels around one clip: print it first to check the
    fit (it uses exactly the clip, pocket and socket of the full panels)."""
    f: Frame = site["frame"]
    carrier, receiver = site["panel"], site["into"]
    clip_box = f.place(box((spec.in_length + 7.0 + spec.out_length, 20.0, 60.0),
                           ((spec.out_length + 1.0 - spec.in_length - 6.0) / 2, 0.0, 28.0)))
    sock_box = f.place(box((61.0, 20.0, 60.0), (29.5, 0.0, 22.0)))
    clip_solids = {g: s ^ clip_box for g, s in assembled[carrier].items()}
    sock_solids = {g: s ^ sock_box for g, s in assembled[receiver].items()}
    clip = f.place(spec.clip())
    note = f"cut from the {carrier}/{receiver} joint at {site['at_mm']} (assembled mm)"
    out = [_make_part("coupon_clip", clip_solids, PANEL_ROT[carrier], [(clip, "coupon_socket")], "coupon", [note], zones),
           _make_part("coupon_socket", sock_solids, PANEL_ROT[receiver], [], "coupon", [note])]
    return [p for p in out if p is not None]


# ------------------------------------------------------------------------------- pin panels
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


def _pin_panels(groups, core, inner, wall, sc, pin_len, clearance, light_guard, rep, warn):  # noqa: C901
    solids = _split_panels(groups, inner)

    # The pin sits towards the cavity side of the wall, with its socket kept inside the wall's
    # footprint (so the socket never crosses the panel's inner edge) and `min_skin` to the outside:
    #   margin + socket diagonal + skin <= wall, socket diagonal = pin diagonal + 2*sqrt(2)*clearance
    skin = float(sc["min_skin"])
    margin = 0.05
    diag = min(float(sc["pin_size"]), wall - skin - margin - 2 * math.sqrt(2) * clearance)
    n_pins = int(sc["pins_per_edge"])
    pins: dict[str, list] = {k: [] for k in solids}
    sockets: dict[str, list] = {k: [] for k in solids}
    unplaced = 0
    if pin_len > 0 and diag >= 0.8 and n_pins > 0:
        sock_diag = diag + 2 * math.sqrt(2) * clearance
        for j in _panel_joints(core, inner):
            want = [(i + 0.5) / n_pins - 0.5 for i in range(n_pins)]  # e.g. -0.25, +0.25 of the edge
            for f in want:
                placed = False
                for shift in (0.0, 0.08, -0.08, 0.16, -0.16):  # slide along the edge to dodge light
                    base = j.base + j.v * (f + shift) * j.edge_len * 0.9 + j.u * (margin + sock_diag / 2)
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
    for name in PANEL_ORDER:
        sol = dict(solids[name])
        if pins[name]:
            sol["body"] = sol["body"] + union([pin for pin, _ in pins[name]])
        if sockets[name]:
            sk = union(sockets[name])
            sol = {g: s - sk for g, s in sol.items()}
        part = _make_part(name, sol, PANEL_ROT[name], pins[name])
        if part is not None:
            parts.append(part)
    return parts, rep
