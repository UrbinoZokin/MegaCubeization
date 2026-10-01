"""Phase 3: geometry generation.

Everything is built at true scale in model millimetres (1 game cm = 10 mm), recentred on the
build, and scaled to printed millimetres at the end. Printed-size parameters (minimum belt
width, wall thickness, ...) are converted with the scale factor, which is computed first from the
build's bounding box (see ``fit_scale``).

Steps:
 1. instantiate primitives per mapping rule: body (opaque), light (belts/signs/lifts),
    debug placeholders (unmapped classes);
 2. union the body (manifold3d batch boolean), fill enclosed voids;
 3. light elements: exaggerate to printable minimums, attach to the surface (raycasts);
 4. hollow: cavity = erosion of the filled body by the wall thickness (optionally with a
    self-supporting pyramid roof for one-piece prints), optional dark inner layer;
 5. windows behind every light element (holes, or light pipes filled with translucent material);
 6. inlay booleans so body, dark and light never overlap; LED/cable opening in the bottom face;
 7. scale to printed mm, bottom at z = 0.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from manifold3d import Manifold, OpType

from .config import Config
from .mapping import Mapping
from .model import Build, quat_to_matrix
from .primitives import (BoxElement, LightElement, body_primitive, box, hull, make_light_element,
                         placeholder, transform, union)

log = logging.getLogger(__name__)


@dataclass
class Geometry:
    scale: float  # printed mm per true-scale model mm
    body: Manifold
    light: Manifold
    dark: Manifold | None
    placeholders: Manifold  # unmapped classes, for the debug output only
    filled: Manifold  # outer envelope of the body (no cavity), printed mm
    cavity: Manifold | None  # printed mm
    origin_model_mm: np.ndarray  # model-frame point that maps to the printed origin (before z lift)
    alignment: np.ndarray  # 3x3 rotation applied before scaling
    lift_z: float  # printed mm added so the bottom sits at z = 0
    report: dict[str, Any] = field(default_factory=dict)
    core_box: np.ndarray | None = None  # (2,3) printed mm: the cube's own outer faces (no protrusions)

    def groups(self) -> dict[str, Manifold]:
        out = {"body": self.body, "light": self.light}
        if self.dark is not None and not self.dark.is_empty():
            out["dark"] = self.dark
        return out


# ------------------------------------------------------------------------------- helpers
def erode(solid: Manifold, t: float) -> Manifold:
    """Erosion by an axis-aligned cube of half-size ``t``: intersection of the 26 translates.

    Exact for convex solids and for faces aligned with the axes, a close approximation otherwise.
    Thin parts (< 2t) vanish, which is what a constant-thickness shell needs."""
    if t <= 0:
        return solid
    copies = [solid.translate(tuple(float(c) * t for c in v)) for v in itertools.product((-1, 0, 1), repeat=3) if any(v)]
    return Manifold.batch_boolean([solid] + copies, OpType.Intersect)


def dilate(solid: Manifold, t: float) -> Manifold:
    """Dilation by an axis-aligned cube of half-size ``t``.

    Convex solids (e.g. the cavity of a cube-like build): exact Minkowski sum, cheap for convex
    inputs. Otherwise: union of the 26 translates (the dual of ``erode``), close for shapes larger
    than ``t`` but it can leave a few degenerate triangles on coplanar faces."""
    if t <= 0:
        return solid
    if solid.num_tri() <= 5000:  # Minkowski cost grows with face count; only try it on small meshes
        hull_volume = Manifold.hull_points(np.asarray(solid.to_mesh64().vert_properties)[:, :3]).volume()
        if hull_volume <= solid.volume() * (1 + 1e-9):
            return solid.minkowski_sum(Manifold.cube((2 * t, 2 * t, 2 * t), True))
    copies = [solid.translate(tuple(float(c) * t for c in v)) for v in itertools.product((-1, 0, 1), repeat=3) if any(v)]
    return Manifold.batch_boolean([solid] + copies, OpType.Add)


def drop_slivers(solid: Manifold | None, min_volume: float) -> tuple[Manifold | None, int]:
    """Remove disconnected zero-volume shells that booleans can leave behind on coplanar faces."""
    if solid is None or solid.is_empty():
        return solid, 0
    parts = solid.decompose()
    keep = [p for p in parts if abs(p.volume()) >= min_volume]
    if len(keep) == len(parts):
        return solid, 0
    return (Manifold.compose(keep) if keep else Manifold()), len(parts) - len(keep)


def fill_voids(solid: Manifold) -> Manifold:
    """Drop enclosed cavities: keep only the positive-volume shells (decompose gives voids as negative)."""
    parts = solid.decompose()
    return union([p for p in parts if p.volume() > 0])


def dominant_box(solid: Manifold) -> np.ndarray:
    """(2,3) box from the six outer planes carrying the most outward-facing area, i.e. the faces
    of a cube-like build, ignoring smaller things standing on them."""
    mesh = solid.to_mesh64()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    area = np.linalg.norm(n, axis=1) / 2
    n = n / np.maximum(2 * area[:, None], 1e-30)
    centre = v[f].mean(axis=1)
    lo, hi = np.array(solid.bounding_box()).reshape(2, 3)
    box_ = np.vstack([lo, hi])
    res = 1e-6 * float(np.max(hi - lo))
    for k in range(3):
        for side, sgn in ((0, -1.0), (1, 1.0)):
            sel = n[:, k] * sgn > 0.999
            if not sel.any():
                continue
            coords = centre[sel, k]
            weight: dict[float, float] = {}
            exact: dict[float, float] = {}
            for c, a in zip(coords, area[sel]):
                key = float(np.round(c / res))
                weight[key] = weight.get(key, 0.0) + a
                exact.setdefault(key, float(c))
            box_[side, k] = exact[max(weight, key=weight.get)]
    return box_


def pyramid_roof(cavity: Manifold) -> Manifold:
    """Convex region whose top is a 45 degree hip roof over the cavity's bounding box, so a
    one-piece print needs no support inside the closed cube."""
    x0, y0, z0, x1, y1, z1 = cavity.bounding_box()
    wx, wy = x1 - x0, y1 - y0
    eave = z1 - min(wx, wy) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pts = [(x, y, z0 - 1.0) for x in (x0 - 1, x1 + 1) for y in (y0 - 1, y1 + 1)]
    pts += [(x, y, eave) for x in (x0, x1) for y in (y0, y1)]
    if wx >= wy:
        pts += [(x0 + wy / 2, cy, z1), (x1 - wy / 2, cy, z1)]
    else:
        pts += [(cx, y0 + wx / 2, z1), (cx, y1 - wx / 2, z1)]
    return hull(np.array(pts))


def cast_rays(solid: Manifold, origins: np.ndarray, dirs: np.ndarray, max_dist: float) -> np.ndarray:
    """Distance along each ray to where it enters ``solid`` (0 if it starts inside, inf if no hit)."""
    out = np.full(len(origins), np.inf)
    if solid is None or solid.is_empty():
        return out
    for i, (o, d) in enumerate(zip(origins, dirs)):
        hits = solid.ray_cast(tuple(map(float, o)), tuple(map(float, o + d * max_dist)))
        if hits:
            h = hits[0]
            out[i] = 0.0 if np.dot(h.normal, d) > 0 else h.distance * max_dist
    return out


def dominant_alignment(mats: list[np.ndarray], tol_deg: float = 2.0) -> np.ndarray:
    """Rotation A that maps the most common building axes onto X/Y/Z (cube faces axis-aligned).

    Each object contributes its 3 local axes (and negatives). The most popular direction becomes
    one axis, the most popular perpendicular one the second. The assignment to world X/Y/Z is
    chosen to rotate as little as possible."""
    if not mats:
        return np.eye(3)
    vecs = np.concatenate([np.hstack([m, -m]).T for m in mats])
    cos_tol = np.cos(np.radians(tol_deg))
    sims = vecs @ vecs.T
    votes = (sims > cos_tol).sum(1)
    a = vecs[int(np.argmax(votes))]
    perp = np.abs(vecs @ a) < np.sin(np.radians(tol_deg))
    if not perp.any():
        return np.eye(3)
    votes_b = np.where(perp, (sims > cos_tol).sum(1), -1)
    b = vecs[int(np.argmax(votes_b))]
    b = b - a * (a @ b)
    b /= np.linalg.norm(b)
    basis = np.column_stack([a, b, np.cross(a, b)])
    best, best_angle = np.eye(3), np.inf
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            B = basis[:, perm] * np.array(signs)
            if np.linalg.det(B) < 0:
                continue
            A = B.T  # maps basis vectors onto world axes
            angle = np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1)))
            if angle < best_angle:
                best, best_angle = A, angle
    return best


_CUBE_ROTATIONS = None


def cube_rotations() -> np.ndarray:
    """The 24 proper rotations that map the coordinate axes onto themselves."""
    global _CUBE_ROTATIONS
    if _CUBE_ROTATIONS is None:
        mats = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                m = np.zeros((3, 3))
                for row, (col, sg) in enumerate(zip(perm, signs)):
                    m[row, col] = sg
                if np.linalg.det(m) > 0:
                    mats.append(m)
        _CUBE_ROTATIONS = np.array(mats)
    return _CUBE_ROTATIONS


def snap_matrix(m4: np.ndarray, ref: np.ndarray, pos_grid: float, rot_tol_deg: float) -> tuple[np.ndarray, float]:
    """Round a world matrix's translation to ``pos_grid`` (relative to ``ref``) and its rotation to
    the nearest 90-degree orientation if within ``rot_tol_deg``. Returns (matrix, position change)."""
    out = m4.copy()
    if rot_tol_deg > 0:
        lin = m4[:3, :3]
        scales = np.linalg.norm(lin, axis=0)
        R = lin / scales
        cand = cube_rotations()
        cos = (np.einsum("kij,ij->k", cand, R) - 1) / 2  # cos of the angle between R and each candidate
        k = int(np.argmax(cos))
        if np.degrees(np.arccos(np.clip(cos[k], -1, 1))) <= rot_tol_deg:
            out[:3, :3] = cand[k] * scales
    moved = 0.0
    if pos_grid > 0:
        t = ref + np.round((m4[:3, 3] - ref) / pos_grid) * pos_grid
        moved = float(np.linalg.norm(t - m4[:3, 3]))
        out[:3, 3] = t
    return out, moved


def fit_scale(extent_model: np.ndarray, max_dimension: float) -> float:
    return float(max_dimension / max(float(np.max(extent_model)), 1e-9))


def _bbox(points: np.ndarray) -> np.ndarray:
    return np.vstack([points.min(0), points.max(0)])


# ------------------------------------------------------------------------------- main
def build_geometry(build: Build, mapping: Mapping, cfg: Config, *, split_mode: str | None = None) -> Geometry:
    if build.frame != "model":
        raise ValueError("build_geometry expects a model-frame build (run coords.to_model_frame first)")
    split_mode = split_mode or cfg.get_path("split.mode")
    rep: dict[str, Any] = {"warnings": []}

    def warn(msg: str):
        log.warning(msg)
        rep["warnings"].append(msg)

    # --- sort objects by rule
    body_objs, light_objs, unmapped, excluded = [], [], [], []
    for o in build.objects:
        rule = mapping.resolve(o.class_name)
        if rule is None:
            unmapped.append(o)
        elif rule.group == "exclude":
            excluded.append(o)
        elif rule.group == "body":
            body_objs.append((o, rule))
        else:
            light_objs.append((o, rule))
    if not body_objs:
        raise ValueError("no body objects: nothing to print (check the class mapping)")
    if unmapped:
        counts: dict[str, int] = {}
        for o in unmapped:
            counts[o.class_name] = counts.get(o.class_name, 0) + 1
        warn(f"{len(unmapped)} unmapped object(s) ({', '.join(f'{k} x{v}' for k, v in sorted(counts.items()))}): "
             f"{'placeholders added to the body' if mapping.unmapped_policy == 'body' else 'excluded; see debug placeholders'}")

    # --- alignment and recentring (double precision, before anything is built)
    A = np.eye(3)
    if cfg.get_path("align") == "auto":
        A = dominant_alignment([quat_to_matrix(o.transform.rotation) for o, _ in body_objs])
    rep["alignment_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(A) - 1) / 2, -1, 1))))
    origins = np.array([o.transform.translation for o, _ in body_objs]) @ A.T
    centre = _bbox(origins).mean(0)
    pre = np.eye(4)
    pre[:3, :3] = A
    pre[:3, 3] = -centre

    def world(o) -> np.ndarray:
        return pre @ o.matrix()

    # --- body primitives (true scale); snapping makes touching faces coincide exactly
    weld = float(cfg.get_path("weld_model_mm", 0.0))
    pos_grid = float(cfg.get_path("snap.position_cm", 0.0)) * 10.0  # game cm -> model mm
    rot_tol = float(cfg.get_path("snap.rotation_deg", 0.0))
    ref = world(body_objs[0][0])[:3, 3]
    body_parts, cache, max_moved = [], {}, 0.0
    for o, rule in body_objs:
        if rule.name not in cache:
            cache[rule.name] = body_primitive(rule, weld)
        m4, moved = snap_matrix(world(o), ref, pos_grid, rot_tol)
        max_moved = max(max_moved, moved)
        body_parts.append(transform(cache[rule.name], m4))
    rep["snap"] = {"position_grid_model_mm": pos_grid, "rotation_tol_deg": rot_tol,
                   "max_position_change_model_mm": round(max_moved, 4)}
    ph_parts = [transform(placeholder(mapping.placeholder_size), world(o)) for o in unmapped]
    if mapping.unmapped_policy == "body":
        body_parts += ph_parts
    body_union = union(body_parts)
    body_box = np.array(body_union.bounding_box()).reshape(2, 3)

    # --- light elements; the scale depends on their (exaggerated) extent, so iterate
    max_dim = float(cfg.get_path("scale.max_dimension"))
    lc = cfg["light"]
    s = fit_scale(np.ptp(body_box, axis=0), max_dim)
    for _ in range(4):
        elements = _light_elements(light_objs, world, lc, s, warn)
        _attach(elements, body_union, lc, s, rep)  # warned about once, after the scale has settled
        pts = [body_box] + [e.points() for e in elements]
        extent = np.ptp(_bbox(np.vstack(pts)), axis=0)
        s_new = fit_scale(extent, max_dim)
        if s_new >= s * (1 - 1e-4):
            s = min(s, s_new)
            break
        s = s_new
    if rep["attach"]["floating"]:
        warn(f"{rep['attach']['floating']} light element(s) not touching the body (attach={lc.get('attach')}): "
             "they would print as loose parts")
    rep["scale"] = s
    rep["model_extent_m"] = (extent / 1000.0).round(3).tolist()  # true-scale metres (1 game m = 1000 model mm)
    rep["printed_extent_mm"] = (extent * s).round(2).tolist()

    # --- hollowing
    filled = fill_voids(body_union)
    hc, dc = cfg["hollow"], cfg["dark_layer"]
    roof = hc.get("roof", "auto")
    if roof == "auto":
        roof = "pyramid" if split_mode == "one" else "flat"
    cavity = None
    core_b = None
    if hc.get("enabled", True):
        wall = float(hc["wall"]) / s
        core = filled
        if hc.get("core", "box") == "box":  # clip to the cube's own faces: protrusions stay solid
            b = core_b = dominant_box(filled)
            core = filled ^ box(b[1] - b[0], b.mean(0))
            rep["core_box_printed_mm"] = (np.ptp(b, axis=0) * s).round(3).tolist()
        cavity = erode(core, wall)
        if cavity.is_empty():
            warn(f"hollowing produced no cavity (wall {hc['wall']} mm too thick for this shape?)")
            cavity = None
        elif roof == "pyramid":
            cavity = cavity ^ pyramid_roof(cavity)
        shell = filled - cavity if cavity is not None else filled
    else:
        shell = body_union  # keep the build's own interior as-is
        void = filled - body_union
        cavity = None if void.is_empty() else void
    rep["roof"] = roof if cavity is not None else None

    dark = None
    if dc.get("enabled") and cavity is not None:
        d = float(dc["thickness"]) / s
        if hc.get("enabled", True) and d >= float(hc["wall"]) / s:
            warn("dark_layer.thickness >= hollow.wall: dark layer disabled")
        else:
            # a lining of thickness d around the cavity (also under a pyramid roof), rest stays body
            dark = shell ^ dilate(cavity, d)
            shell = shell - dark

    # --- light union, windows, inlay booleans
    light = union([solid for e in elements for solid in e.solids()])
    windows = Manifold()
    wc = cfg["windows"]
    if wc.get("mode", "none") != "none" and cavity is not None and elements:
        windows = _windows(elements, cavity, wc, s, rep, warn)
    if wc.get("mode") == "fill" and not windows.is_empty():
        # light pipes end flush with the inner surface: nothing sticks into the cavity, so lids and
        # panels stay flat (the slot itself overshoots by `margin` for a clean cut)
        light = light + (windows ^ (filled - cavity))
    body = shell - light - windows
    if dark is not None:
        dark = dark - light - windows

    # --- LED / cable opening through the bottom face
    lo = cfg["led_opening"]
    if lo.get("enabled"):
        opening = _led_opening(filled, cavity, lo, s)
        body = body - opening
        if dark is not None:
            dark = dark - opening
        if not (light ^ opening).is_empty():
            warn("LED opening cuts through light elements on the bottom face")
            light = light - opening

    # --- to printed mm, bottom at z = 0
    scale = (s, s, s)
    body, light, filled = body.scale(scale), light.scale(scale), filled.scale(scale)
    dark = dark.scale(scale) if dark is not None else None
    cavity = cavity.scale(scale) if cavity is not None else None
    placeholders = union(ph_parts).scale(scale)
    lift_z = -float(filled.bounding_box()[2])
    up = (0.0, 0.0, lift_z)
    body, light, filled = body.translate(up), light.translate(up), filled.translate(up)
    dark = dark.translate(up) if dark is not None else None
    cavity = cavity.translate(up) if cavity is not None else None
    placeholders = placeholders.translate(up)

    dropped = 0
    body, n = drop_slivers(body, 1e-3)
    dropped += n
    light, n = drop_slivers(light, 1e-3)
    dropped += n
    if dark is not None:
        dark, n = drop_slivers(dark, 1e-3)
        dropped += n
    if dropped:
        rep["slivers_removed"] = dropped

    rep.update({
        "objects": {"body": len(body_objs), "light": len(light_objs), "excluded": len(excluded), "unmapped": len(unmapped)},
        "volumes_mm3": {"body": round(body.volume(), 1), "light": round(light.volume(), 1),
                        "dark": round(dark.volume(), 1) if dark is not None else 0.0,
                        "cavity": round(cavity.volume(), 1) if cavity is not None else 0.0},
        "light_elements": [{"id": e.obj_id, "class": e.class_name, "kind": e.kind,
                            "attach_gap_mm": None if e.attach_gap is None else round(e.attach_gap * s, 3),
                            "window_depth_mm": None if e.window_depth is None else round(e.window_depth * s, 3),
                            "notes": e.notes} for e in elements],
    })
    if core_b is None:
        core_b = dominant_box(filled.scale((1 / s,) * 3).translate((0, 0, -lift_z / s)))
    core_print = core_b * s + np.array([0.0, 0.0, lift_z])
    return Geometry(s, body, light, dark, placeholders, filled, cavity, centre, A, lift_z, rep, core_print)


def _light_elements(light_objs, world, lc, s, warn) -> list[LightElement]:
    out = []
    for o, rule in light_objs:
        try:
            out.append(make_light_element(
                o, rule, world(o), min_width=float(lc["min_width"]) / s, min_thickness=float(lc["min_thickness"]) / s,
                max_segment=float(lc["spline_max_segment"]) / s, max_angle_deg=float(lc["spline_max_angle_deg"])))
        except ValueError as exc:
            warn(f"light element skipped: {exc}")
    return out


def _attach(elements, body: Manifold, lc, s, rep) -> None:
    mode = lc.get("attach", "snap")
    max_gap = float(lc["attach_max_gap"]) / s
    embed = float(lc["embed"]) / s
    floating = 0
    for e in elements:
        if isinstance(e, BoxElement) and e.inward is None:
            _choose_inward(e, body, max_gap)
        if isinstance(e, BoxElement) and e.inward is None:
            floating += 1
            e.notes.append("no surface found next to this lift: left floating")
            continue
        o, d, idx = e.back_samples()
        g = cast_rays(body, o, d, max_gap)
        per_sample = np.full(idx.max() + 1, np.inf)
        np.minimum.at(per_sample, idx, g)
        finite = per_sample[np.isfinite(per_sample)]
        e.attach_gap = float(finite.min()) if len(finite) else None
        if not len(finite):
            floating += 1
            e.notes.append(f"no surface within {lc['attach_max_gap']} mm behind it: left floating")
            continue
        e.apply_attach(per_sample, mode, embed)
        if mode == "none" and e.attach_gap is not None and e.attach_gap > 0:
            floating += 1
    rep["attach"] = {"mode": mode, "floating": floating}


def _choose_inward(e: BoxElement, body: Manifold, max_dist: float) -> None:
    """Lifts stand next to a face: pick the horizontal local axis whose ray reaches the body first."""
    best = None
    for k in (0, 1):
        for sgn in (1, -1):
            d = e.axes[:, k] * sgn
            o = e.centre + d * e.half[k]
            heights = np.linspace(-0.8, 0.8, 5) * e.half[2]
            origins = o + np.outer(heights, e.axes[:, 2])
            dist = cast_rays(body, origins, np.repeat(d[None], len(origins), 0), max_dist).min()
            if np.isfinite(dist) and (best is None or dist < best[0]):
                best = (dist, (k, sgn))
    if best is not None:
        e.inward = best[1]


def _windows(elements, cavity: Manifold, wc, s, rep, warn) -> Manifold:
    inset = float(wc["inset"]) / s
    margin = float(wc["margin"]) / s
    max_depth = float(wc["max_depth"]) / s
    overlap = 0.05 / s
    pieces, skipped = [], 0
    for e in elements:
        o, d, idx = e.back_samples()
        reach = cast_rays(cavity, o, d, max_depth)
        per_sample = np.full(idx.max() + 1, np.inf)
        np.minimum.at(per_sample, idx, reach)
        depth = per_sample + margin
        made = [m for m in e.window_solids(depth, inset, overlap) if m is not None and not m.is_empty()]
        if made:
            finite = depth[np.isfinite(depth)]
            e.window_depth = float(finite.max())
            pieces += made
        else:
            skipped += 1
            e.notes.append("no window: element too small for the inset, or cavity out of reach (windows.max_depth)")
    if skipped:
        warn(f"{skipped} light element(s) got no window (see light_elements notes)")
    rep["windows"] = {"mode": wc["mode"], "made": len(elements) - skipped, "skipped": skipped}
    return union(pieces)


def _led_opening(filled: Manifold, cavity: Manifold | None, lo, s) -> Manifold:
    x0, y0, z0, x1, y1, z1 = filled.bounding_box()
    cx, cy = (x0 + x1) / 2 + float(lo["offset"][0]) / s, (y0 + y1) / 2 + float(lo["offset"][1]) / s
    top = (cavity.bounding_box()[2] if cavity is not None else z0 + (z1 - z0) / 2) + 1.0 / s
    h = top - z0 + 2.0 / s
    w, dpt = float(lo["size"][0]) / s, float(lo["size"][1]) / s
    if lo.get("shape", "circle") == "circle":
        return Manifold.cylinder(h, w / 2, w / 2, 96).translate((cx, cy, z0 - 1.0 / s))
    return box((w, dpt, h), (cx, cy, z0 - 1.0 / s + h / 2))
