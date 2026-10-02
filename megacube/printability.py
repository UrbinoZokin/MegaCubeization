"""Phase 4 checks on the scaled, split parts.

* thin walls: from surface samples, cast a ray inward along -normal; the distance to where it
  leaves the solid is the local thickness (per material group);
* narrow gaps: cast a ray outward along +normal; a hit on any solid of the same part closer than
  the limit is a gap the nozzle can't resolve (contact between materials at ~0 distance is fine);
* supports: downward-facing triangles steeper than the overhang limit, split into "from the
  plate" (allowed) and "on the model" (internal supports, which the brief forbids);
* bed contact: area of the part's underside lying on the plate.

Flagged samples are clustered into features and reported with their location.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from manifold3d import Manifold

RNG_SEED = 7


@dataclass
class Finding:
    kind: str  # thin_wall | narrow_gap | support_on_model
    part: str
    group: str
    value_mm: float  # min thickness / min gap / overhang area
    samples: int
    location_mm: list[float]  # in the part's print coordinates
    location_assembled_mm: list[float]  # in the assembled model
    bbox_mm: list[float]
    face: str = ""

    def to_dict(self):
        return asdict(self)


def _mesh(solid: Manifold):
    m = solid.to_mesh64()
    return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts, np.int64)


def surface_samples(solid: Manifold, n_target: int, rng=None, min_altitude: float = 5e-3):
    """Every triangle's centroid (so small features are never skipped) plus area-weighted
    random points up to ``n_target``. Returns points, unit normals.

    Triangles narrower than ``min_altitude`` mm (sub-resolution strips such as the 1 micron step
    a cut leaves at a panel's inner edge) are not sampled: nothing that thin can print, and real
    thin walls always have full-size faces that do get sampled."""
    v, f = _mesh(solid)
    if not len(f):
        return np.zeros((0, 3)), np.zeros((0, 3))
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    longest = np.max(np.linalg.norm(np.stack([b - a, c - b, a - c], 1), axis=2), axis=1)
    ok = (area > 1e-12) & (2 * area / np.maximum(longest, 1e-30) >= min_altitude)
    a, b, c, n, area = a[ok], b[ok], c[ok], n[ok], area[ok]
    n = n / (2 * area[:, None])
    pts, nrm = [(a + b + c) / 3], [n]
    extra = max(0, n_target - len(area))
    if extra:
        rng = rng or np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(area), extra, p=area / area.sum())
        r1, r2 = rng.random(extra), rng.random(extra)
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        pts.append(a[idx] + (b[idx] - a[idx]) * r1[:, None] + (c[idx] - a[idx]) * r2[:, None])
        nrm.append(n[idx])
    return np.vstack(pts), np.vstack(nrm)


def _first_hit(solid: Manifold, o, d, max_dist):
    hits = solid.ray_cast(tuple(map(float, o)), tuple(map(float, o + d * max_dist)))
    if not hits:
        return None
    h = hits[0]
    return h.distance * max_dist, float(np.dot(h.normal, d))


def thickness(solid: Manifold, pts, nrm, max_dist: float, eps: float = 1e-4, max_angle_deg: float = 60.0) -> np.ndarray:
    """Local wall thickness along -normal (inf where thicker than ``max_dist``).

    Only exits through a face within ``max_angle_deg`` of parallel to the entry face count: a wall
    has two roughly parallel sides, while a ray that starts near a convex corner and leaves through
    the neighbouring face is measuring the corner, not a wall."""
    out = np.full(len(pts), np.inf)
    cos_lim = np.cos(np.radians(max_angle_deg))
    for i, (p, n) in enumerate(zip(pts, nrm)):
        hit = _first_hit(solid, p - n * eps, -n, max_dist)
        if hit is not None and hit[1] > cos_lim:  # leaving the solid through a roughly parallel face
            out[i] = hit[0] + eps
    return out


def gaps(solids: list[Manifold], pts, nrm, max_dist: float, contact: float = 1e-3) -> np.ndarray:
    """Clear distance along +normal to the next solid of the same part (inf if none close)."""
    out = np.full(len(pts), np.inf)
    eps = contact / 2
    for i, (p, n) in enumerate(zip(pts, nrm)):
        o = p + n * eps
        for s in solids:
            hit = _first_hit(s, o, n, max_dist)
            if hit is not None and hit[1] < 0 and hit[0] + eps > contact:  # entering another surface
                out[i] = min(out[i], hit[0] + eps)
    return out


def clusters(points: np.ndarray, cell: float) -> list[np.ndarray]:
    """Group points whose grid cells touch (26-neighbourhood). Returns index arrays."""
    if not len(points):
        return []
    keys = [tuple(k) for k in np.floor(points / cell).astype(np.int64)]
    by_cell: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        by_cell.setdefault(k, []).append(i)
    seen, out = set(), []
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for start in by_cell:
        if start in seen:
            continue
        seen.add(start)
        queue, members = deque([start]), []
        while queue:
            k = queue.popleft()
            members += by_cell[k]
            for o in offs:
                nb = (k[0] + o[0], k[1] + o[1], k[2] + o[2])
                if nb in by_cell and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        out.append(np.array(members))
    return out


def face_label(p_assembled, centre) -> str:
    d = np.asarray(p_assembled) - centre
    k = int(np.argmax(np.abs(d)))
    return ("-" if d[k] < 0 else "+") + "XYZ"[k] + " side"


def _findings(kind, part, group, pts, values, cell, to_assembled, centre, reduce=min):
    out = []
    for idx in clusters(pts, cell):
        p = pts[idx]
        loc = p.mean(0)
        loc_a = (to_assembled @ np.r_[loc, 1.0])[:3]
        out.append(Finding(kind, part, group, round(float(reduce(values[idx])), 3), len(idx),
                           loc.round(2).tolist(), loc_a.round(2).tolist(),
                           np.r_[p.min(0), p.max(0)].round(2).tolist(), face_label(loc_a, centre)))
    return out


def _points_in_polygons(xy: np.ndarray, polygons) -> np.ndarray:
    """Even-odd point-in-polygon test against all contours of a cross-section."""
    inside = np.zeros(len(xy), bool)
    for poly in polygons:
        px, py = poly[:, 0], poly[:, 1]
        qx, qy = np.roll(px, -1), np.roll(py, -1)
        for x, y, x2, y2 in zip(px, py, qx, qy):
            cond = (y > xy[:, 1]) != (y2 > xy[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x + (xy[:, 1] - y) * (x2 - x) / (y2 - y)
            inside ^= cond & (xy[:, 0] < xint)
    return inside


def check_part(name: str, solids: dict[str, Manifold], to_assembled: np.ndarray, assembled_centre,
               min_wall: float, min_gap: float, samples: int) -> list[Finding]:
    """Thin walls per material group, narrow gaps across all groups of one printed part."""
    findings = []
    groups = {g: s for g, s in solids.items() if s is not None and not s.is_empty()}
    total_area = sum(s.surface_area() for s in groups.values()) or 1.0
    everything = list(groups.values())
    for g, s in groups.items():
        pts, nrm = surface_samples(s, int(samples * s.surface_area() / total_area))
        t = thickness(s, pts, nrm, max_dist=min_wall * 1.5)
        thin = t < min_wall - 1e-3
        findings += _findings("thin_wall", name, g, pts[thin], t[thin], max(1.0, 2 * min_wall),
                              to_assembled, assembled_centre)
        gp = gaps(everything, pts, nrm, max_dist=min_gap * 1.5)
        narrow = gp < min_gap - 1e-3
        findings += _findings("narrow_gap", name, g, pts[narrow], gp[narrow], max(1.0, 4 * min_gap),
                              to_assembled, assembled_centre)
    return findings


def support_report(name: str, solids: dict[str, Manifold], overhang_deg: float, to_assembled, assembled_centre,
                   reach: float = 1.0, layer: float = 0.2, contact: float = 0.05,
                   bed_tol: float = 0.02) -> tuple[dict, list[Finding]]:
    """Classify the downward-facing area of a part in print orientation.

    For every triangle facing down more steeply than ``overhang_deg``:
      * resting: other material (e.g. the other filament of an inlay) lies directly below within
        ``contact`` mm, so the two print together and nothing overhangs;
      * short ledge: one layer lower there is material within ``reach`` mm horizontally, so the
        printer just extends the perimeter (small ledges on walls print fine without support);
      * support from the plate: a support column would reach the build plate (allowed);
      * support on the model: it would have to stand on the part itself. These are the internal
        supports the brief rules out, reported with locations.
    """
    cos_lim = np.cos(np.radians(overhang_deg))
    everything = [s for s in solids.values() if s is not None and not s.is_empty()]
    zmin = min(s.bounding_box()[2] for s in everything)
    whole = Manifold.batch_boolean(everything, __import__("manifold3d").OpType.Add) if len(everything) > 1 else everything[0]
    areas = {"bed_contact_mm2": 0.0, "resting_on_other_material_mm2": 0.0, "short_ledges_mm2": 0.0,
             "support_from_plate_mm2": 0.0, "support_on_model_mm2": 0.0}
    candidates = []
    for s in everything:
        v, f = _mesh(s)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        n = np.cross(b - a, c - a)
        area = np.linalg.norm(n, axis=1) / 2
        ok = area > 1e-12
        n[ok] /= (2 * area[ok])[:, None]
        cen = (a + b + c) / 3
        on_bed = ok & (n[:, 2] < -0.999) & (np.abs(cen[:, 2] - zmin) < bed_tol)
        areas["bed_contact_mm2"] += float(area[on_bed].sum())
        over = ok & (n[:, 2] < -cos_lim - 1e-6) & ~on_bed
        candidates += [(cen[i], float(area[i])) for i in np.where(over)[0]]

    sections: dict[float, list] = {}
    model_pts, model_areas = [], []
    for cen, area in candidates:
        o = cen - np.array([0.0, 0.0, 1e-4])
        seg = o[2] - zmin + 1.0
        firsts = [hs[0] for s in everything if (hs := s.ray_cast(tuple(o), (o[0], o[1], zmin - 1.0)))]
        hits = bool(firsts)
        # resting: the point just below is inside other material (first hit is an exit), or touches it
        if any(h.normal[2] < 0 or h.distance * seg < contact for h in firsts):
            areas["resting_on_other_material_mm2"] += area
            continue
        zl = round(float(cen[2] - layer), 3)
        if zl not in sections:
            sections[zl] = whole.slice(zl).offset(reach, __import__("manifold3d").JoinType.Round).to_polygons() if zl > zmin else []
        if sections[zl] and _points_in_polygons(cen[None, :2], sections[zl])[0]:
            areas["short_ledges_mm2"] += area
        elif hits:
            areas["support_on_model_mm2"] += area
            model_pts.append(cen)
            model_areas.append(area)
        else:
            areas["support_from_plate_mm2"] += area
    findings = _findings("support_on_model", name, "all", np.array(model_pts).reshape(-1, 3), np.array(model_areas),
                         2.0, to_assembled, assembled_centre, reduce=np.sum)
    return {k: round(v, 1) for k, v in areas.items()}, findings
