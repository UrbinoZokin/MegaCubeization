"""Phase 4: printability checks, splitting, pins/sockets, print orientation."""
import itertools
import json

import numpy as np
import pytest
from manifold3d import Manifold

from megacube.config import load_config
from megacube.coords import to_model_frame
from megacube.geometry import build_geometry
from megacube.mapping import load_mapping
from megacube.pipeline import run
from megacube.printability import check_part, clusters, support_report
from megacube.split import split_geometry
from megacube.sources.synthetic import SyntheticCubeSource

MAPPING = load_mapping()
I4 = np.eye(4)


def cube(size, centre=(0, 0, 0)):
    return Manifold.cube(tuple(map(float, size)), True).translate(tuple(map(float, centre)))


def kinds(findings, kind):
    return [f for f in findings if f.kind == kind]


# ---------------------------------------------------------------------------- printability unit tests
def test_thin_plate_is_found_with_its_location():
    plate = cube((20, 20, 0.5), (5, -3, 10))
    found = kinds(check_part("t", {"body": plate}, I4, np.zeros(3), 0.8, 0.4, 3000), "thin_wall")
    assert len(found) == 1
    f = found[0]
    assert f.value_mm == pytest.approx(0.5, abs=1e-3)
    assert np.allclose(f.location_mm, (5, -3, 10), atol=1.0)


def test_thick_plate_and_cube_corners_are_fine():
    for solid in (cube((20, 20, 1.0)), cube((10, 10, 10))):
        assert kinds(check_part("t", {"body": solid}, I4, np.zeros(3), 0.8, 0.4, 3000), "thin_wall") == []


def test_narrow_gap_between_blocks():
    a, b = cube((10, 10, 10), (-5.1, 0, 0)), cube((10, 10, 10), (5.1, 0, 0))  # 0.2 mm apart
    found = kinds(check_part("t", {"body": a + b}, I4, np.zeros(3), 0.8, 0.4, 3000), "narrow_gap")
    assert found and min(f.value_mm for f in found) == pytest.approx(0.2, abs=1e-3)
    far = cube((10, 10, 10), (-5.5, 0, 0)) + cube((10, 10, 10), (5.5, 0, 0))  # 1 mm apart
    assert kinds(check_part("t", {"body": far}, I4, np.zeros(3), 0.8, 0.4, 3000), "narrow_gap") == []


def test_materials_in_contact_are_not_a_gap():
    a, b = cube((10, 10, 10), (-5, 0, 0)), cube((10, 10, 10), (5, 0, 0))
    assert kinds(check_part("t", {"body": a, "light": b}, I4, np.zeros(3), 0.8, 0.4, 3000), "narrow_gap") == []


def test_support_classification():
    post = cube((4, 4, 10), (0, 0, 5))
    table = post + cube((30, 30, 2), (0, 0, 11))  # overhanging table top: support from the plate
    rep, findings = support_report("t", {"body": table}, 45, I4, np.zeros(3))
    assert rep["support_from_plate_mm2"] > 800 and rep["support_on_model_mm2"] == 0 and not findings

    base = cube((40, 40, 2), (0, 0, 1))
    bridge = base + cube((4, 30, 10), (-18, 0, 7)) + cube((4, 30, 10), (18, 0, 7)) + cube((40, 30, 2), (0, 0, 13))
    rep, findings = support_report("t", {"body": bridge}, 45, I4, np.zeros(3))
    assert rep["support_on_model_mm2"] > 900 and findings  # ceiling over the base: internal support

    ledge = cube((10, 10, 10), (0, 0, 5)) + cube((0.6, 10, 2), (5.3, 0, 9))  # 0.6 mm ledge on a wall
    rep, _ = support_report("t", {"body": ledge}, 45, I4, np.zeros(3))
    assert rep["short_ledges_mm2"] > 0 and rep["support_from_plate_mm2"] == 0

    lower, upper = cube((10, 10, 5), (0, 0, 2.5)), cube((10, 10, 5), (0, 0, 7.5))  # two filaments stacked
    rep, _ = support_report("t", {"body": lower, "light": upper}, 45, I4, np.zeros(3))
    assert rep["resting_on_other_material_mm2"] == pytest.approx(100, rel=1e-6) and rep["support_on_model_mm2"] == 0


def test_clusters_group_nearby_points():
    pts = np.array([[0, 0, 0], [0.5, 0, 0], [10, 10, 10], [10.4, 10, 10.2]])
    groups = sorted(sorted(g.tolist()) for g in clusters(pts, 1.0))
    assert groups == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------- splitting
@pytest.fixture(scope="module", params=["box_lid", "panels"])
def split(request):
    cfg = load_config(overrides=[f"split.mode={request.param}"])
    geo = build_geometry(to_model_frame(SyntheticCubeSource().read()), MAPPING, cfg)
    parts, rep = split_geometry(geo, cfg)
    return request.param, geo, parts, rep, cfg


def assembled(part):
    return {g: s.transform(part.to_assembled[:3, :4]) for g, s in part.solids.items()}


def test_part_count_and_pins(split):
    mode, _, parts, rep, _ = split
    names = [p.name for p in parts]
    if mode == "box_lid":
        assert names == ["box", "lid"] and rep["pins"]["count"] == 4
    else:
        assert sorted(names) == sorted(["top", "bottom", "front", "back", "left", "right"])
        assert rep["pins"]["count"] == 24
    assert rep["warnings"] == []


def test_parts_lie_flat_and_fit(split):
    _, _, parts, _, cfg = split
    vol = cfg.get_path("scale.build_volume")
    for p in parts:
        bb = p.bbox()
        assert bb[2] == pytest.approx(0, abs=1e-9)
        assert np.all(bb[3:] - bb[:3] <= np.array(vol) + 1e-9)
        rep, findings = support_report(p.name, p.solids, 45, p.to_assembled, np.zeros(3))
        footprint = (bb[3] - bb[0]) * (bb[4] - bb[1])
        assert rep["bed_contact_mm2"] > 0.75 * footprint, p.name  # sits flat on its inner face
        assert rep["support_on_model_mm2"] == 0 and not findings, p.name  # no internal supports


def test_reassembled_parts_do_not_overlap(split):
    """Pins must sit inside sockets with clearance: no two parts may occupy the same space."""
    _, _, parts, _, _ = split
    solids = [(p.name, Manifold.batch_boolean(list(assembled(p).values()), __import__("manifold3d").OpType.Add))
              for p in parts]
    for (na, a), (nb, b) in itertools.combinations(solids, 2):
        assert (a ^ b).volume() == pytest.approx(0, abs=1e-4), (na, nb)


def test_pins_have_clearance_in_their_sockets(split):
    """Each pin, put back in the assembly, keeps exactly the configured clearance to its socket."""
    _, _, parts, rep, cfg = split
    c = float(cfg.get_path("split.clearance"))
    solids = {p.name: Manifold.batch_boolean(list(assembled(p).values()), __import__("manifold3d").OpType.Add)
              for p in parts}
    n = 0
    for p in parts:
        for pin, receiver in p.pins:
            assert pin.min_gap(solids[receiver], 1.0) == pytest.approx(c, abs=5e-3)  # tapered pins: a hair more
            n += 1
    assert n == rep["pins"]["count"] > 0


def test_union_of_parts_matches_the_model(split):
    mode, geo, parts, rep, _ = split
    total = sum(sum(s.volume() for s in p.solids.values()) for p in parts)
    original = sum(s.volume() for s in geo.groups().values())
    if mode == "box_lid":  # plus the four corner posts that carry the pins
        cx0, cy0, cz0, cx1, cy1, cz1 = geo.cavity.bounding_box()
        original += 4 * rep["pins"]["corner_post_mm"] ** 2 * (cz1 - cz0)
    # pins add a little, sockets remove a little: within 0.2 % of the model
    assert total == pytest.approx(original, rel=2e-3)


def test_one_piece_uses_a_pyramid_roof():
    cfg = load_config(overrides=["split.mode=one"])
    geo = build_geometry(to_model_frame(SyntheticCubeSource(features=False).read()), MAPPING, cfg)
    parts, rep = split_geometry(geo, cfg)
    assert geo.report["roof"] == "pyramid" and len(parts) == 1
    sup, findings = support_report("cube", parts[0].solids, 45, I4, np.zeros(3))
    assert sup["support_on_model_mm2"] == 0 and not findings


def test_one_piece_flat_roof_is_flagged():
    cfg = load_config(overrides=["split.mode=one", "hollow.roof=flat"])
    geo = build_geometry(to_model_frame(SyntheticCubeSource(features=False).read()), MAPPING, cfg)
    parts, rep = split_geometry(geo, cfg)
    assert any("internal supports" in w for w in rep["warnings"])
    sup, findings = support_report("cube", parts[0].solids, 45, I4, np.zeros(3))
    assert sup["support_on_model_mm2"] > 10000 and findings


def test_pipeline_writes_parts_and_checks(tmp_path):
    report = run("synthetic:n=4", tmp_path, MAPPING, load_config(overrides=["split.mode=panels", "checks.samples=4000"]))
    assert (tmp_path / "printability.txt").exists()
    assert sorted(report["checks"]["parts"]) == sorted(["top", "bottom", "front", "back", "left", "right"])
    assert all((tmp_path / f).exists() for f in report["files"].values())
    assert json.loads((tmp_path / "report.json").read_text())["split"]["pins"]["count"] == 24
