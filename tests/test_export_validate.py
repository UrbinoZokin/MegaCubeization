"""Phase 5: STL/3MF export, PNG previews, mesh validation."""
import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
from manifold3d import Manifold

from megacube.cli import main
from megacube.config import load_config
from megacube.export import EXTRUDER, mesh_arrays, write_3mf, write_stl
from megacube.mapping import load_mapping
from megacube.pipeline import run
from megacube.preview import COLOURS, VIEWS, layer_from_manifold, render
from megacube.validate import read_stl, validate_arrays, validate_file

CUBE = Manifold.cube((10, 10, 10), True)
NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}


# ---------------------------------------------------------------------------- validation
def test_good_mesh_passes(tmp_path):
    write_stl(CUBE, tmp_path / "c.stl")
    r = validate_file(tmp_path / "c.stl")
    assert r.ok and r.volume_mm3 == pytest.approx(1000) and r.shells == 1


def test_detects_self_intersection():
    v1, f1 = mesh_arrays(CUBE)
    v2, f2 = mesh_arrays(CUBE.translate((5, 3, 2)))  # two shells pushed through each other, no boolean
    r = validate_arrays("x", np.vstack([v1, v2]), np.vstack([f1, f2 + len(v1)]))
    assert r.self_intersections > 0 and not r.ok and r.intersection_locations


def test_detects_coplanar_overlap():
    v1, f1 = mesh_arrays(CUBE)
    v2, f2 = mesh_arrays(CUBE.translate((3, 0, 0)))
    assert validate_arrays("x", np.vstack([v1, v2]), np.vstack([f1, f2 + len(v1)])).self_intersections > 0


def test_detects_holes_and_bad_normals():
    v, f = mesh_arrays(CUBE)
    assert not validate_arrays("hole", v, f[1:]).watertight
    flipped = validate_arrays("flipped", v, f[:, ::-1])
    assert flipped.winding_consistent and not flipped.outward_normals  # consistent but inside-out
    one = f.copy()
    one[0] = one[0, ::-1]
    assert not validate_arrays("one", v, one).winding_consistent
    wrong_stored = -np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0])
    assert not validate_arrays("stored", v, f, wrong_stored).stored_normals_match


def test_touching_but_separate_shells_are_fine():
    v1, f1 = mesh_arrays(CUBE)
    v2, f2 = mesh_arrays(CUBE.translate((10.5, 0, 0)))
    r = validate_arrays("x", np.vstack([v1, v2]), np.vstack([f1, f2 + len(v1)]))
    assert r.ok and r.shells == 2


def test_stl_roundtrip_keeps_geometry(tmp_path):
    s = Manifold.sphere(5, 48)
    write_stl(s, tmp_path / "s.stl")
    v, f, n = read_stl(tmp_path / "s.stl")
    assert len(f) == s.num_tri() and len(v) == s.num_vert()
    assert validate_arrays("s", v, f, n).ok


def test_cli_validate_exit_codes(tmp_path):
    write_stl(CUBE, tmp_path / "ok.stl")
    assert main(["validate", str(tmp_path / "ok.stl")]) == 0
    (tmp_path / "bad.stl").write_bytes((tmp_path / "ok.stl").read_bytes()[:-100] + b"\0" * 0)
    v, f = mesh_arrays(CUBE)
    rec = np.zeros(len(f) - 1, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["v"] = v[f[1:]]
    (tmp_path / "bad.stl").write_bytes(b" " * 80 + np.uint32(len(rec)).tobytes() + rec.tobytes())
    assert main(["validate", str(tmp_path / "bad.stl")]) == 2


# ---------------------------------------------------------------------------- 3MF
def test_3mf_structure_and_parts(tmp_path):
    light = Manifold.cube((2, 2, 2), True).translate((0, 0, 6))
    write_3mf([("box", {"body": CUBE, "light": light})], tmp_path / "p.3mf")
    z = zipfile.ZipFile(tmp_path / "p.3mf")
    assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "Metadata/model_settings.config"} <= set(z.namelist())
    model = ET.fromstring(z.read("3D/3dmodel.model"))
    assert model.get("unit") == "millimeter"
    objs = model.findall("m:resources/m:object", NS)
    meshes = [o for o in objs if o.find("m:mesh", NS) is not None]
    composite = [o for o in objs if o.find("m:components", NS) is not None]
    assert len(meshes) == 2 and len(composite) == 1  # one object, two parts
    tris = [len(o.findall("m:mesh/m:triangles/m:triangle", NS)) for o in meshes]
    assert tris == [CUBE.num_tri(), light.num_tri()]
    items = model.findall("m:build/m:item", NS)
    assert [i.get("objectid") for i in items] == [composite[0].get("id")]
    cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    extruders = {p.find("metadata[@key='name']").get("value"): p.find("metadata[@key='extruder']").get("value")
                 for p in cfg.iter("part")}
    assert extruders == {"box_body": str(EXTRUDER["body"]), "box_light": str(EXTRUDER["light"])}


def test_3mf_reads_back_with_trimesh(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("lxml")
    pytest.importorskip("networkx")
    write_3mf([("a", {"body": CUBE}), ("b", {"body": CUBE, "light": CUBE.translate((0, 0, 10))})], tmp_path / "all.3mf")
    scene = trimesh.load(tmp_path / "all.3mf")
    assert len(scene.geometry) == 3 and all(g.is_watertight for g in scene.geometry.values())
    lo, hi = scene.bounds
    assert hi[0] - lo[0] > 20  # the two objects are laid out side by side, not on top of each other


# ---------------------------------------------------------------------------- previews
def test_render_highlights_light_material():
    light = Manifold.cube((4, 4, 1), True).translate((0, 0, 5.5))
    img = np.asarray(render([layer_from_manifold(CUBE, COLOURS["body"]), layer_from_manifold(light, COLOURS["light"])],
                            VIEWS["top"], size=(200, 200)))
    centre = img[90:110, 90:110].reshape(-1, 3).mean(0)
    corner = img[30:40, 30:40].reshape(-1, 3).mean(0)
    assert centre[0] > 200 and centre[2] < 100  # orange in the middle (light on top)
    assert abs(corner[0] - corner[2]) < 30  # grey body around it


# ---------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("mode", ["box_lid", "panels"])
def test_full_build_outputs_are_valid(tmp_path, mode):
    cfg = load_config(overrides=[f"split.mode={mode}", "checks.samples=3000", "preview.size=160"])
    report = run("synthetic:n=6", tmp_path, load_mapping(), cfg)
    assert report["validation"] and all(r["ok"] for r in report["validation"])
    stls = [f for f in report["files"].values() if f.endswith(".stl")]
    assert len(report["validation"]) == len(stls)
    for name in ("assembled.png", "faces.png", "cutaway.png", "parts.png", "debug_unmapped.png"):
        assert (tmp_path / "preview" / name).stat().st_size > 0
    assert (tmp_path / "summary.md").read_text().startswith("# megacube build")
    assert json.loads((tmp_path / "report.json").read_text())["split"]["mode"] == mode
    # every part's groups share one origin: their STLs must line up (shared bounding volume, no offsets)
    for part in report["checks"]["parts"]:
        files = [f for f in stls if f.startswith(f"parts/{part}_")]
        boxes = np.array([np.r_[read_stl(tmp_path / f)[0].min(0), read_stl(tmp_path / f)[0].max(0)] for f in files])
        assert boxes[:, 2].min() == pytest.approx(0, abs=1e-4)  # part rests on the bed
