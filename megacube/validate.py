"""Phase 5 mesh validation, on the files actually written (what the slicer will load).

Per mesh:
* watertight: every edge is shared by exactly two triangles (no holes, no non-manifold edges);
* consistent winding: every edge is used once in each direction;
* outward normals: every closed shell has positive signed volume, and the facet normals stored
  in the STL agree with the vertex winding;
* no degenerate (zero-area) triangles;
* no self-intersections: uniform-grid broad phase, then exact-ish edge/triangle tests between
  triangles that share no vertex (plus a 2D overlap test for coplanar pairs).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

STL_RECORD = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path: str | Path):
    """Binary or ASCII STL -> (merged vertices, faces, stored facet normals)."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = int(np.frombuffer(data[80:84], "<u4")[0])
        if 84 + count * 50 == len(data):
            rec = np.frombuffer(data[84:], STL_RECORD, count)
            tris = rec["v"].astype(np.float64)
            normals = rec["n"].astype(np.float64)
            return (*_merge(tris), normals)
    text = data.decode("ascii", errors="replace").split()
    tris, normals, i = [], [], 0
    while i < len(text):
        if text[i] == "normal":
            normals.append([float(x) for x in text[i + 1:i + 4]])
            i += 4
        elif text[i] == "vertex":
            tris.append([float(x) for x in text[i + 1:i + 4]])
            i += 4
        else:
            i += 1
    tris = np.array(tris, float).reshape(-1, 3, 3)
    return (*_merge(tris), np.array(normals, float).reshape(-1, 3))


def _merge(tris: np.ndarray):
    """Exact-duplicate vertex merge (STL stores every triangle's corners separately)."""
    flat = tris.reshape(-1, 3)
    verts, inv = np.unique(flat, axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


# --------------------------------------------------------------------------- topology
def _components(faces: np.ndarray) -> np.ndarray:
    """Connected components of faces sharing an edge (union-find). Returns a label per face."""
    parent = np.arange(len(faces))

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    for a, b in zip(owner[:-1][same], owner[1:][same]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(len(faces))])


def _signed_volumes(v, f, labels):
    tri = v[f]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    out = {}
    for lab, val in zip(labels, vol):
        out[lab] = out.get(lab, 0.0) + val
    return out


# --------------------------------------------------------------------------- self-intersections
def _broad_phase(lo: np.ndarray, hi: np.ndarray, max_pairs: int = 30_000_000) -> np.ndarray:
    """Candidate pairs (i < j) whose bounding boxes share a grid cell."""
    n = len(lo)
    ext = hi - lo
    span = float(np.max(hi.max(0) - lo.min(0)))
    h = max(float(np.median(ext.max(1))) * 2.0, span / 256.0, 1e-9)
    c0 = np.floor((lo - lo.min(0)) / h).astype(np.int64)
    c1 = np.floor((hi - lo.min(0)) / h).astype(np.int64)
    counts = np.prod(c1 - c0 + 1, axis=1)
    big = counts > 512  # very large triangles: tested against everything by AABB instead
    small = np.where(~big)[0]
    cells, owners = [], []
    for i in small:  # most triangles cover 1-8 cells
        xs = np.arange(c0[i, 0], c1[i, 0] + 1)
        ys = np.arange(c0[i, 1], c1[i, 1] + 1)
        zs = np.arange(c0[i, 2], c1[i, 2] + 1)
        g = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
        cells.append(g)
        owners.append(np.full(len(g), i))
    pairs = []
    if cells:
        cells, owners = np.concatenate(cells), np.concatenate(owners)
        key = (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (cells[:, 2] * 83492791)
        order = np.argsort(key, kind="stable")
        key, owners = key[order], owners[order]
        starts = np.r_[0, np.where(np.diff(key) != 0)[0] + 1]
        ends = np.r_[starts[1:], len(key)]
        for s, e in zip(starts, ends):
            m = e - s
            if m < 2:
                continue
            ids = owners[s:e]
            a, b = np.triu_indices(m, 1)
            pairs.append(np.stack([ids[a], ids[b]], 1))
            if sum(len(p) for p in pairs) > max_pairs:
                raise MemoryError("too many candidate pairs for the self-intersection check")
    for i in np.where(big)[0]:
        overlap = np.all((lo <= hi[i]) & (hi >= lo[i]), axis=1)
        overlap[i] = False
        js = np.where(overlap)[0]
        pairs.append(np.stack([np.full(len(js), i), js], 1))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    p = np.concatenate(pairs)
    p = np.sort(p, axis=1)
    p = p[p[:, 0] != p[:, 1]]
    return np.unique(p[:, 0] * n + p[:, 1]).astype(np.int64)


def _segments_hit_triangles(p0, p1, a, b, c, tol):
    """Vectorised: does segment p0->p1 cross the interior of triangle abc (strictly)?"""
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-12 * np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d, axis=1), 1e-30)
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - a
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", d, q)
    t = inv * np.einsum("ij,ij->i", e2, q)
    return ok & (u > tol) & (v > tol) & (u + v < 1 - tol) & (t > tol) & (t < 1 - tol)


def _coplanar_overlap(A, B, tol):
    """For coplanar triangle pairs: do their interiors overlap? (2D edge crossings / containment)

    All 2D cross products are divided by the pair's squared size, so ``tol`` is relative: collinear
    or touching edges (common on flat faces) never count as crossings because of float noise."""
    n = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    k = np.argmax(np.abs(n), axis=1)
    keep = [[1, 2], [0, 2], [0, 1]]  # drop the dominant normal axis: project onto the best plane
    A2 = np.array([A[i][:, keep[k[i]]] for i in range(len(A))])
    B2 = np.array([B[i][:, keep[k[i]]] for i in range(len(B))])
    both = np.concatenate([A2, B2], axis=1)
    L2 = np.maximum(np.ptp(both, axis=1).max(1), 1e-30) ** 2  # squared size of each pair
    hit = np.zeros(len(A), bool)

    def cross2(o, p, q):
        return ((p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0])) / L2

    for i in range(3):
        for j in range(3):
            p, q = A2[:, i], A2[:, (i + 1) % 3]
            r, s = B2[:, j], B2[:, (j + 1) % 3]
            d1, d2 = cross2(r, s, p), cross2(r, s, q)
            d3, d4 = cross2(p, q, r), cross2(p, q, s)
            hit |= (d1 * d2 < -tol) & (d3 * d4 < -tol)

    def inside(P, T):
        s1, s2, s3 = cross2(T[:, 0], T[:, 1], P), cross2(T[:, 1], T[:, 2], P), cross2(T[:, 2], T[:, 0], P)
        return ((s1 > tol) & (s2 > tol) & (s3 > tol)) | ((s1 < -tol) & (s2 < -tol) & (s3 < -tol))

    hit |= inside(A2.mean(1), B2) | inside(B2.mean(1), A2)
    return hit


def self_intersections(v: np.ndarray, f: np.ndarray, rel_tol: float = 1e-9) -> tuple[int, list]:
    """Number of intersecting triangle pairs that share no vertex, plus up to 10 locations."""
    if len(f) < 2:
        return 0, []
    tri = v[f]
    lo, hi = tri.min(1), tri.max(1)
    keys = _broad_phase(lo, hi)
    if not len(keys):
        return 0, []
    n = len(f)
    i, j = keys // n, keys % n
    ov = np.all((lo[i] <= hi[j]) & (hi[i] >= lo[j]), axis=1)
    i, j = i[ov], j[ov]
    shared = (f[i][:, :, None] == f[j][:, None, :]).any(axis=(1, 2))
    i, j = i[~shared], j[~shared]
    if not len(i):
        return 0, []
    A, B = tri[i], tri[j]
    hit = np.zeros(len(i), bool)
    for T, U in ((A, B), (B, A)):
        for k in range(3):
            hit |= _segments_hit_triangles(T[:, k], T[:, (k + 1) % 3], U[:, 0], U[:, 1], U[:, 2], rel_tol)
    # coplanar pairs are invisible to the segment test: check them in 2D
    nA = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    nB = np.cross(B[:, 1] - B[:, 0], B[:, 2] - B[:, 0])
    scale = np.maximum(np.linalg.norm(nA, axis=1), 1e-30)
    dist = np.abs(np.einsum("ijk,ik->ij", B - A[:, :1, :], nA)) / scale[:, None]
    size = np.maximum(hi[i].max(1) - lo[i].min(1), 1e-9)
    coplanar = (dist.max(1) < 1e-7 * size) & (np.linalg.norm(np.cross(nA, nB), axis=1) < 1e-7 * scale * np.linalg.norm(nB, axis=1))
    if coplanar.any():
        idx = np.where(coplanar & ~hit)[0]
        if len(idx):
            hit[idx] |= _coplanar_overlap(A[idx], B[idx], 1e-9)
    where = [((A[k].mean(0) + B[k].mean(0)) / 2).round(3).tolist() for k in np.where(hit)[0][:10]]
    return int(hit.sum()), where


# --------------------------------------------------------------------------- report
@dataclass
class MeshReport:
    file: str
    triangles: int
    shells: int
    watertight: bool
    winding_consistent: bool
    outward_normals: bool
    stored_normals_match: bool
    degenerate_triangles: int
    self_intersections: int
    intersection_locations: list = field(default_factory=list)
    volume_mm3: float = 0.0

    @property
    def ok(self) -> bool:
        return (self.watertight and self.winding_consistent and self.outward_normals and self.stored_normals_match
                and self.degenerate_triangles == 0 and self.self_intersections == 0)

    def to_dict(self):
        return {**asdict(self), "ok": self.ok}


def validate_arrays(name: str, v: np.ndarray, f: np.ndarray, stored_normals: np.ndarray | None = None) -> MeshReport:
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    scale = max(float(np.ptp(v, axis=0).max()), 1e-9) if len(v) else 1.0
    degenerate = int(((area2 < 1e-14 * scale * scale) | (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])).sum())

    und = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, und_counts = np.unique(und, axis=0, return_counts=True)
    watertight = bool(len(und_counts) and np.all(und_counts == 2))
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, dir_counts = np.unique(directed, axis=0, return_counts=True)
    winding = bool(watertight and np.all(dir_counts == 1))

    labels = _components(f)
    vols = _signed_volumes(v, f, labels)
    outward = bool(winding and all(val > 0 for val in vols.values()))
    if stored_normals is not None and len(stored_normals) == len(f):
        good = area2 > 0
        unit = cross[good] / area2[good, None]
        sn = stored_normals[good]
        sn = sn / np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-30)
        stored_ok = bool(np.all(np.einsum("ij,ij->i", unit, sn) > 0.99))
    else:
        stored_ok = True
    n_int, where = self_intersections(v, f)
    return MeshReport(name, len(f), len(vols), watertight, winding, outward, stored_ok, degenerate, n_int, where,
                      round(float(sum(vols.values())), 3))


def validate_file(path: str | Path) -> MeshReport:
    v, f, normals = read_stl(path)
    return validate_arrays(str(path), v, f, normals)


def format_reports(reports: list[MeshReport]) -> str:
    lines = ["Mesh validation (watertight, consistent winding, outward normals, stored normals, "
             "degenerate triangles, self-intersections)"]
    for r in reports:
        flag = "OK  " if r.ok else "FAIL"
        lines.append(f"  {flag} {r.file}: {r.triangles} tris, {r.shells} shell(s), vol {r.volume_mm3} mm3"
                     + ("" if r.ok else f" | watertight={r.watertight} winding={r.winding_consistent} "
                        f"outward={r.outward_normals} stored_normals={r.stored_normals_match} "
                        f"degenerate={r.degenerate_triangles} self_intersections={r.self_intersections} "
                        f"{r.intersection_locations[:3]}"))
    bad = [r for r in reports if not r.ok]
    lines.append(f"  {len(reports) - len(bad)}/{len(reports)} meshes valid")
    return "\n".join(lines)
