"""Phase 5 export: per-part STLs per material group, and 3MF files with the groups as parts of one object.

STL: one file per (part, group) in the part's print orientation. All groups of a part share the
same transform, so importing them together into Bambu Studio lines them up ("load as a single
object with multiple parts" when asked).

3MF (core spec): each material group is a mesh object; the part is one object made of
components referencing them, so slicers can treat the groups as parts of a single object. A
``Metadata/model_settings.config`` in the Bambu Studio / OrcaSlicer style names the parts and
assigns extruders (body 1, light 2, dark 3). That file follows Bambu's own project format but
hasn't been tested in Bambu Studio from here. If it's ignored, assign filaments per part by hand.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

EXTRUDER = {"body": 1, "light": 2, "dark": 3}
NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="text/xml"/>
</Types>
"""
RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""


def mesh_arrays(solid):
    m = solid.to_mesh64()
    return np.asarray(m.vert_properties[:, :3], float), np.asarray(m.tri_verts, np.int64)


def write_stl(solid, path: Path) -> None:
    """Binary STL with facet normals computed from the winding (outward for manifold3d output)."""
    v, f = mesh_arrays(solid)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"megacube binary STL".ljust(80, b" "))
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


def _mesh_xml(oid: int, name: str, v: np.ndarray, f: np.ndarray) -> str:
    verts = "".join(f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in v)
    tris = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
    return (f' <object id="{oid}" type="model" name="{escape(name)}"><mesh><vertices>{verts}</vertices>'
            f'<triangles>{tris}</triangles></mesh></object>\n')


def write_3mf(objects: list[tuple[str, dict]], path: Path, spacing: float = 10.0) -> None:
    """``objects``: [(object name, {group: Manifold})]. Each becomes one object made of one
    component per group; several objects are laid out in a row (``spacing`` mm apart)."""
    resources, items, settings = [], [], []
    next_id = 1
    x_cursor = 0.0
    for obj_name, groups in objects:
        comp_ids = []
        boxes = []
        for g, solid in groups.items():
            if solid is None or solid.is_empty():
                continue
            v, f = mesh_arrays(solid)
            resources.append(_mesh_xml(next_id, f"{obj_name}_{g}", v, f))
            comp_ids.append((next_id, g))
            boxes.append(np.r_[v.min(0), v.max(0)])
            next_id += 1
        if not comp_ids:
            continue
        bb = np.array(boxes)
        lo, hi = bb[:, :3].min(0), bb[:, 3:].max(0)
        dx = x_cursor - lo[0] if len(objects) > 1 else 0.0
        x_cursor += (hi[0] - lo[0]) + spacing
        comps = "".join(f'<component objectid="{cid}"/>' for cid, _ in comp_ids)
        resources.append(f' <object id="{next_id}" type="model" name="{escape(obj_name)}"><components>{comps}</components></object>\n')
        items.append(f'  <item objectid="{next_id}" transform="1 0 0 0 1 0 0 0 1 {dx:.6f} 0 0"/>\n')
        parts = "".join(
            f'  <part id="{cid}" subtype="normal_part"><metadata key="name" value="{escape(obj_name)}_{g}"/>'
            f'<metadata key="extruder" value="{EXTRUDER.get(g, 1)}"/></part>\n' for cid, g in comp_ids)
        settings.append(f' <object id="{next_id}">\n  <metadata key="name" value="{escape(obj_name)}"/>\n'
                        f'  <metadata key="extruder" value="1"/>\n{parts} </object>\n')
        next_id += 1
    model = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}">\n'
             f' <metadata name="Application">megacube</metadata>\n<resources>\n{"".join(resources)}</resources>\n'
             f'<build>\n{"".join(items)}</build>\n</model>\n')
    config = f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n{"".join(settings)}</config>\n'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", config)
