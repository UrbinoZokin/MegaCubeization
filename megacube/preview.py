"""PNG previews: a small software rasteriser (numpy z-buffer, orthographic camera).

No OpenGL/display needed, so it runs headless and in CI. Flat shading plus dark feature edges
make the geometry readable; each material group gets its own colour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {
    "body": (178, 180, 184),
    "light": (255, 176, 32),
    "dark": (62, 64, 70),
    "placeholders": (230, 40, 200),
    "pins": (90, 160, 230),
}

VIEWS = {  # direction FROM the target TOWARD the camera, in the model frame (right-handed, Z up)
    "iso_ne": (1.0, 1.0, 0.8),
    "iso_nw": (-1.0, 1.0, 0.8),
    "iso_se": (1.0, -1.0, 0.8),
    "iso_sw": (-1.0, -1.0, 0.8),
    "iso_nw_low": (-1.0, 1.0, -0.6),
    "iso_se_low": (1.0, -1.0, -0.6),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}
FACE_VIEWS = ["top", "bottom", "+x", "-x", "+y", "-y"]


@dataclass
class MeshLayer:
    vertices: np.ndarray  # (n,3)
    faces: np.ndarray  # (m,3)
    colour: tuple[int, int, int]
    name: str = ""


def layer_from_manifold(solid, colour, name="") -> MeshLayer | None:
    if solid is None or solid.is_empty():
        return None
    mesh = solid.to_mesh64()
    return MeshLayer(np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, np.int64), colour, name)


def _camera(view_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    back = np.asarray(view_dir, float)
    back /= np.linalg.norm(back)
    fwd = -back
    world_up = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.99 else np.array([0.0, 1.0 if fwd[2] < 0 else -1.0, 0.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def _feature_edges(faces: np.ndarray, normals: np.ndarray, angle_deg: float = 25.0) -> np.ndarray:
    """Edges where adjacent faces meet at more than ``angle_deg`` (plus boundary edges)."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    i = np.where(same)[0]
    cosang = (normals[owner[i]] * normals[owner[i + 1]]).sum(1)
    sharp = e[i][cosang < np.cos(np.radians(angle_deg))]
    paired = np.zeros(len(e), bool)
    paired[i] = paired[i + 1] = True
    return np.concatenate([sharp, e[~paired]])


def render(layers: list[MeshLayer], view_dir, size=(900, 900), title: str = "", bbox=None,
           background=(255, 255, 255), edges: bool = True) -> Image.Image:
    W, H = size
    right, up, fwd = _camera(view_dir)
    layers = [lay for lay in layers if lay is not None and len(lay.faces)]
    pts_all = np.vstack([lay.vertices for lay in layers]) if bbox is None else np.asarray(bbox, float)
    proj = np.c_[pts_all @ right, pts_all @ up]
    lo, hi = proj.min(0), proj.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1e-9)
    scale = 0.86 * min(W, H) / span
    centre = (lo + hi) / 2

    zbuf = np.full((H, W), np.inf)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = background
    light_dir = -fwd * 0.6 + up * 0.5 + right * 0.35
    light_dir /= np.linalg.norm(light_dir)

    edge_jobs = []
    for lay in layers:
        v = lay.vertices
        sx = (v @ right - centre[0]) * scale + W / 2
        sy = H / 2 - (v @ up - centre[1]) * scale
        sz = v @ fwd
        tri = lay.faces
        n = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        if edges:
            edge_jobs.append((sx, sy, sz, _feature_edges(tri, n)))
        visible = (n @ fwd) < 0  # closed meshes: back faces are hidden anyway
        shade = 0.38 + 0.62 * np.clip(n @ light_dir, 0, 1)
        col = np.asarray(lay.colour, float)
        for f, sh in zip(tri[visible], shade[visible]):
            x, y, z = sx[f], sy[f], sz[f]
            x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), W - 1)
            y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())), H - 1)
            if x0 > x1 or y0 > y1:
                continue
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
            b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
            c = 1 - a - b
            inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
            if not inside.any():
                continue
            depth = a * z[0] + b * z[1] + c * z[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            win = inside & (depth < sub)
            sub[win] = depth[win]
            img[y0:y1 + 1, x0:x1 + 1][win] = (col * sh).astype(np.uint8)

    if edges:  # dark lines on feature edges, depth-tested with a small bias
        bias = 1.5 / scale
        for sx, sy, sz, ed in edge_jobs:
            for i0, i1 in ed:
                n_steps = int(max(abs(sx[i1] - sx[i0]), abs(sy[i1] - sy[i0]))) + 1
                t = np.linspace(0, 1, n_steps)
                px = np.round(sx[i0] + (sx[i1] - sx[i0]) * t).astype(int)
                py = np.round(sy[i0] + (sy[i1] - sy[i0]) * t).astype(int)
                pz = sz[i0] + (sz[i1] - sz[i0]) * t
                ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                px, py, pz = px[ok], py[ok], pz[ok]
                vis = pz <= zbuf[py, px] + bias
                img[py[vis], px[vis]] = (img[py[vis], px[vis]] * 0.35).astype(np.uint8)

    out = Image.fromarray(img)
    if title:
        ImageDraw.Draw(out).text((10, 8), title, fill=(30, 30, 30))
    return out


def contact_sheet(images: list[Image.Image], labels: list[str], cols: int = 3) -> Image.Image:
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        draw.text((x + 10, y + h - 20), label, fill=(30, 30, 30))
    return sheet


# ------------------------------------------------------------------------- pipeline sheets
def _layers(solids: dict, extra=None):
    out = [layer_from_manifold(s, COLOURS.get(g, (150, 150, 150)), g) for g, s in solids.items()]
    if extra:
        out += extra
    return [lay for lay in out if lay is not None]


def joint_sheet(parts, site: dict, size: int = 700) -> Image.Image | None:
    """Close-up of one snap joint from the coupon pieces: a section through the middle of the clip
    (barbs hooked behind the socket's lips), the pair pulled apart, and both pieces as printed."""
    by_name = {p.name: p for p in parts}
    if "coupon_clip" not in by_name or "coupon_socket" not in by_name or not site:
        return None
    clip_p, sock_p = by_name["coupon_clip"], by_name["coupon_socket"]
    tint = {"body": COLOURS["body"], "light": COLOURS["light"], "dark": COLOURS["dark"]}
    sock_tint = {"body": COLOURS["pins"], "light": COLOURS["light"], "dark": (40, 70, 110)}

    def placed(p, shift=(0.0, 0.0, 0.0)):
        return {g: s.transform(p.to_assembled[:3, :4]).translate(tuple(map(float, shift))) for g, s in p.solids.items()}

    a, u = np.asarray(site["a"], float), np.asarray(site["u"], float)
    o = np.asarray(site["origin"], float)
    off = float(np.dot(u, o) + site["clip_thickness_mm"] / 2)
    clip_a, sock_a = placed(clip_p), placed(sock_p)
    cut = [layer_from_manifold(s.trim_by_plane(tuple(map(float, u)), off), tint.get(g, (150, 150, 150)), g)
           for g, s in clip_a.items()]
    cut += [layer_from_manifold(s.trim_by_plane(tuple(map(float, u)), off), sock_tint.get(g, (120, 150, 200)), g)
            for g, s in sock_a.items()]
    v = np.cross(u, a)
    s = (size, size)
    ims = [render([lay for lay in cut if lay is not None], tuple(-u + 0.25 * v + 0.15 * a), s,
                  "section through the clip: barbs behind the socket lips (grey: side panel, blue: top/bottom)")]
    apart = _layers({g: m for g, m in clip_a.items()}) + \
        [layer_from_manifold(m, sock_tint.get(g, (120, 150, 200)), g) for g, m in placed(sock_p, a * 8.0).items()]
    ims.append(render([lay for lay in apart if lay is not None], tuple(-u + v * 0.45 - a * 0.5), s,
                      "pulled apart along the push direction (8 mm), from the cavity side: relief pocket, "
                      "prongs, socket necks"))
    a_print = clip_p.to_print[:3, :3] @ a  # the clip's direction on the print bed
    ims.append(render(_layers(clip_p.solids), tuple(a_print * 1.0 + np.array([0.35, 0.35, 0.9])), s,
                      "coupon_clip as printed (inner face on the bed, clip lying flat)"))
    ims.append(render(_layers(sock_p.solids), (0.5, -0.8, -1.0), s, "coupon_socket as printed, from below (socket on the bed)"))
    return contact_sheet(ims, ["section", "pulled apart", "clip piece", "socket piece"], cols=2)


def write_previews(out_dir, geo, parts, size: int = 700, joint: dict | None = None) -> dict[str, str]:
    """Contact sheets: assembled iso views, per-face orthographic views, cut-away, parts in print
    orientation, a snap joint close-up, and the debug view with unmapped-class placeholders."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = (size, size)
    groups = geo.groups()
    layers = _layers(groups)
    scale_note = f"scale {geo.scale:.5f} (1 game m = {geo.scale * 1000:.2f} mm)"
    files = {}

    iso = ["iso_ne", "iso_sw", "iso_nw", "iso_se_low"]
    contact_sheet([render(layers, VIEWS[v], s, f"{v}  {scale_note}") for v in iso], iso, cols=2).save(out / "assembled.png")
    files["assembled"] = "preview/assembled.png"

    labels = {"top": "top (+Z) face", "bottom": "bottom (-Z) face", "+x": "+X face", "-x": "-X face",
              "+y": "+Y face", "-y": "-Y face"}
    contact_sheet([render(layers, VIEWS[v], s, labels[v] + " (orthographic)") for v in FACE_VIEWS],
                  [labels[v] for v in FACE_VIEWS], cols=3).save(out / "faces.png")
    files["faces"] = "preview/faces.png"

    cy = sum(geo.filled.bounding_box()[i] for i in (1, 4)) / 2
    cut = {g: m.trim_by_plane((0.0, -1.0, 0.0), -cy) for g, m in groups.items()}
    render(_layers(cut), (0.35, 1.0, 0.25), s, "cut-away at the centre (front half), light pipes in orange") \
        .save(out / "cutaway.png")
    files["cutaway"] = "preview/cutaway.png"

    ims = [render(_layers(p.solids), (0.6, -1.0, 0.9), s, f"{p.name} (print orientation, bed = z0)") for p in parts]
    contact_sheet(ims, [p.name for p in parts], cols=min(3, len(ims))).save(out / "parts.png")
    files["parts"] = "preview/parts.png"

    sheet = joint_sheet(parts, joint, size) if joint else None
    if sheet is not None:
        sheet.save(out / "snap_joint.png")
        files["snap_joint"] = "preview/snap_joint.png"

    if not geo.placeholders.is_empty():
        debug = [layer_from_manifold(geo.body, (215, 216, 218), "body"),
                 layer_from_manifold(geo.placeholders, COLOURS["placeholders"], "placeholders")]
        ims = [render(debug, VIEWS[v], s, f"DEBUG: unmapped classes as placeholder boxes ({v})") for v in ("iso_ne", "top")]
        contact_sheet(ims, ["iso_ne", "top"], cols=2).save(out / "debug_unmapped.png")
        files["debug_unmapped"] = "preview/debug_unmapped.png"
    return files
