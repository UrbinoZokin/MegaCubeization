"""Phase 3: geometry generation (primitives, sweeps, hollowing, light inlays, windows)."""
import json
import math

import numpy as np
import pytest
from manifold3d import Manifold

from megacube.cli import main
from megacube.config import load_config
from megacube.coords import to_model_frame
from megacube.geometry import build_geometry, dominant_alignment
from megacube.mapping import load_mapping
from megacube.model import Transform, matrix_to_quat, quat_to_matrix
from megacube.primitives import body_primitive, swept_solids, union
from megacube.sources.synthetic import SyntheticCubeSource
from megacube.sweep import hermite, hermite_derivative, profile_frames, rect_rings, sample_spline, tube_folds, tube_mesh

MAPPING = load_mapping()


def mesh_stats(solid):
    m = solid.to_mesh64()
    v, f = np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
    return v, f, area


def build(overrides=(), source=None):
    raw = source if source is not None else SyntheticCubeSource().read()
    return build_geometry(to_model_frame(raw), MAPPING, load_config(overrides=list(overrides)))


@pytest.fixture(scope="module")
def default():
    return build()


# ---------------------------------------------------------------------------- splines and sweeps
def test_hermite_endpoints_and_tangents():
    p0, t0, p1, t1 = map(np.array, ([0, 0, 0.], [3, 1, 0.], [5, 2, 1.], [0, 4, 2.]))
    assert np.allclose(hermite(p0, t0, p1, t1, [0, 1]), [p0, p1])
    assert np.allclose(hermite_derivative(p0, t0, p1, t1, [0, 1]), [t0, t1])


def test_sampling_respects_max_segment():
    P, T = sample_spline([[0, 0, 0], [100, 0, 0]], [[100, 0, 0]] * 2, [[100, 0, 0]] * 2, 7.0, 5)
    assert np.allclose(P[:, 1:], 0) and np.allclose(T, [1, 0, 0])
    assert np.diff(P[:, 0]).max() <= 7.0 + 1e-9 and P[-1, 0] == pytest.approx(100)


def test_quarter_arc_tangents_fit_a_circle():
    r = 800.0
    k = 4 * math.tan(math.pi / 8) * r
    P, _ = sample_spline([[0, 0, 0], [r, r, 0]], [[k, 0, 0], [0, k, 0]], [[k, 0, 0], [0, k, 0]], 5.0, 1.0)
    radius = np.linalg.norm(P - [0, r, 0], axis=1)
    assert np.abs(radius - r).max() < 1e-3 * r


def straight_rings(length=1000.0, width=120.0, thick=20.0, n=11):
    P = np.c_[np.linspace(0, length, n), np.zeros(n), np.zeros(n)]
    U, S = profile_frames(np.tile([1.0, 0, 0], (n, 1)), (0, 0, 1))
    return rect_rings(P, U, S, width / 2, 0.0, -thick)


def test_tube_mesh_is_a_clean_closed_solid():
    tube = tube_mesh(straight_rings())
    assert tube.volume() == pytest.approx(1000 * 120 * 20, rel=1e-9)
    assert tube.num_tri() == 10 * 8 + 4 and tube.genus() == 0


def test_tight_bend_falls_back_to_convex_pieces():
    # U-turn of radius 50 with a 300 wide profile: the inner side would fold through itself
    t = np.linspace(0, math.pi, 25)
    P = np.c_[50 * np.sin(t), 50 - 50 * np.cos(t), np.zeros_like(t)]
    T = np.c_[np.cos(t), np.sin(t), np.zeros_like(t)]
    U, S = profile_frames(T, (0, 0, 1))
    rings = rect_rings(P, U, S, 150.0, 0.0, -20.0)
    assert tube_folds(rings)
    solids = swept_solids(rings, overlap=0.1)
    assert len(solids) > 1 and union(solids).volume() > 0


# ---------------------------------------------------------------------------- primitives
def test_box_wedge_and_wall_offsets():
    ramp = body_primitive(MAPPING.resolve("Build_Ramp_8x4_01_C"))
    assert ramp.volume() == pytest.approx(8000 * 8000 * 4000 / 2)
    x0, y0, z0, x1, y1, z1 = ramp.bounding_box()
    assert (x0, z0, x1, z1) == pytest.approx((-4000, -2000, 4000, 2000))
    wall = body_primitive(MAPPING.resolve("Build_Wall_8x4_01_C"))
    assert wall.bounding_box()[2] == pytest.approx(0) and wall.bounding_box()[5] == pytest.approx(4000)  # origin at the bottom
    qp = body_primitive(MAPPING.resolve("Build_QuarterPipe_C"))
    assert 0 < qp.volume() < 8000 * 8000 * 4000


def test_ramp_high_end_is_at_local_minus_x():
    ramp = body_primitive(MAPPING.resolve("Build_Ramp_8x2_01_C"))
    v, _, _ = mesh_stats(ramp)
    assert v[v[:, 2] > 999][:, 0].max() == pytest.approx(-4000)  # top vertices only at x = -4000


def test_asymmetric_hull_is_mirrored_into_the_model_frame():
    """corner_up is high at the game's (-X, -Y) corner. Game -Y is model +Y, so in model-local
    coordinates the apex must be at (-X, +Y): the shape keeps its handedness."""
    rule = MAPPING.resolve("Build_Ramp_UpCorner_Concrete_8x4_C")
    v, _, _ = mesh_stats(body_primitive(rule))
    apex = v[np.argmax(v[:, 2])]
    assert apex[0] == pytest.approx(-4000) and apex[1] == pytest.approx(4000)


# ---------------------------------------------------------------------------- integration
def test_scale_fits_the_target(default):
    ext = default.report["printed_extent_mm"]
    assert max(ext) == pytest.approx(240.0, abs=1e-6)
    lo, hi = np.array(default.body.bounding_box()).reshape(2, 3), np.array(default.light.bounding_box()).reshape(2, 3)
    total = np.maximum(hi, np.array(default.body.bounding_box()[3:])) - np.minimum(lo, np.array(default.body.bounding_box()[:3]))
    assert total.max() <= 240.0 + 1e-6


def test_groups_are_clean_solids_that_do_not_overlap(default):
    assert len(default.body.decompose()) == 1
    assert len(default.light.decompose()) == default.report["objects"]["light"] == 7
    for solid in (default.body, default.light):
        _, _, area = mesh_stats(solid)
        assert (area < 1e-6).sum() == 0
        assert solid.status().name == "NoError"
    assert (default.body ^ default.light).volume() == pytest.approx(0, abs=1e-4)  # mm3: nothing


def test_shell_thickness_and_core_box(default):
    fx0, fy0, fz0, fx1, fy1, _ = default.filled.bounding_box()
    cx0, cy0, cz0, cx1, cy1, cz1 = default.cavity.bounding_box()
    core = default.report["core_box_printed_mm"]
    assert (cx0 - fx0, fx1 - cx1, cy0 - fy0, fy1 - cy1, cz0 - fz0) == pytest.approx((3.0,) * 5, abs=1e-6)
    assert cz1 == pytest.approx(core[2] - 3.0, abs=1e-3)  # under the cube's own top face, not the wall on it


def test_every_light_element_is_attached_and_lit(default):
    assert default.report["attach"]["floating"] == 0
    assert default.report["windows"] == {"mode": "fill", "made": 7, "skipped": 0}
    for part in default.light.decompose():  # each light pipe reaches the cavity, flush with its surface
        assert (part ^ default.cavity).volume() == pytest.approx(0, abs=1e-6)
        assert part.min_gap(default.cavity, 1.0) < 1e-6  # touches the cavity wall where the pipe ends


def test_led_opening_goes_through_the_bottom(default):
    cx = sum(default.filled.bounding_box()[i] for i in (0, 3)) / 2
    cy = sum(default.filled.bounding_box()[i] for i in (1, 4)) / 2
    probe = Manifold.cube((2, 2, 2), True).translate((cx, cy, 1.5))
    assert (default.body ^ probe).volume() == pytest.approx(0, abs=1e-9)


def test_unmapped_objects_become_debug_placeholders(default):
    assert default.report["objects"]["unmapped"] == 2
    assert default.placeholders.volume() > 0
    assert (default.placeholders - default.body).volume() > 0  # not part of the printed body
    assert any("unmapped" in w for w in default.report["warnings"])


def test_unmapped_policy_body_prints_placeholders(tmp_path, default):
    p = tmp_path / "classes.yaml"
    p.write_text(MAPPING_TEXT.replace("policy: exclude", "policy: body"))
    from megacube.mapping import load_mapping as lm
    g = build_geometry(to_model_frame(SyntheticCubeSource().read()), lm(p), load_config())
    assert (g.placeholders - g.body).volume() == pytest.approx(0, abs=1e-6)  # now part of the body


def test_window_modes():
    hole = build(["windows.mode=hole"])
    none = build(["windows.mode=none"])
    assert (hole.light ^ hole.cavity).volume() == pytest.approx(0, abs=1e-9)  # nothing pokes into the cavity
    assert hole.body.genus() >= 7  # but the shell has the slots
    assert (none.light ^ none.cavity).volume() == pytest.approx(0, abs=1e-9)
    assert none.body.genus() == 0  # only the LED opening: no slots
    assert "windows" not in none.report


def test_dark_layer_lines_the_cavity():
    g = build(["dark_layer.enabled=true"])
    assert g.dark is not None and g.dark.volume() > 0
    for a, b in ((g.body, g.dark), (g.dark, g.light), (g.body, g.light)):
        assert (a ^ b).volume() == pytest.approx(0, abs=1e-4)  # mm3: sub-micron cleanup tolerance
    dx0, dy0, _, dx1, dy1, _ = g.dark.bounding_box()
    cx0, cy0, _, cx1, cy1, _ = g.cavity.bounding_box()
    assert (cx0 - dx0, dx1 - cx1, cy0 - dy0, dy1 - cy1) == pytest.approx((0.8,) * 4, abs=1e-6)


def test_pyramid_roof_is_self_supporting():
    g = build(["hollow.roof=pyramid"])
    v, f, area = mesh_stats(g.cavity)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # cavity faces pointing up are the ceiling seen from inside: none may be flatter than 45 degrees
    assert n[:, 2].max() <= math.cos(math.radians(45)) + 1e-6


def test_attach_none_leaves_pole_belts_floating(default):
    g = build(["light.attach=none"])
    belt_gaps = [e["attach_gap_mm"] for e in g.report["light_elements"] if e["class"] == "Build_ConveyorBeltMk5_C"]
    assert belt_gaps[0] > 3  # 1 m pole height minus belt thickness, at ~4.6 mm per game metre
    assert g.light.bounding_box()[5] > default.light.bounding_box()[5]  # belts stay up on their poles
    # the light pipes are still inlaid in the shell, but no longer touch the floating belts
    assert len(g.light.decompose()) > 7
    assert any("not touching the body" in w for w in g.report["warnings"])


def test_auto_alignment_undoes_a_rotated_build():
    raw = SyntheticCubeSource(features=False).read()
    yaw = quat_to_matrix((0, 0, math.sin(math.radians(15)), math.cos(math.radians(15))))  # 30 degree yaw
    for o in raw.objects:
        r = yaw @ quat_to_matrix(o.transform.rotation)
        o.transform = Transform(matrix_to_quat(r), tuple(yaw @ np.array(o.transform.translation)), o.transform.scale)
    g = build(["align=auto"], source=raw)
    assert g.report["alignment_deg"] == pytest.approx(30, abs=1e-6)
    ext = g.report["printed_extent_mm"]
    assert ext == pytest.approx([240.0, 240.0, 240.0], abs=1e-6)  # faces axis-aligned again


def test_dominant_alignment_identity_for_aligned_input():
    assert np.allclose(dominant_alignment([np.eye(3)] * 5), np.eye(3))


def test_cli_build(tmp_path):
    out = tmp_path / "out"
    assert main(["build", "synthetic:n=6", "-o", str(out), "--set", "hollow.wall=2.5", "--no-previews",
                 "--set", "checks.samples=3000"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["geometry"]["scale"] > 0 and (out / "assembled" / "body.stl").stat().st_size > 0
    assert (out / "assembled" / "debug_placeholders.stl").exists() and (out / "coverage.txt").exists()


MAPPING_TEXT = open(MAPPING.path).read()
