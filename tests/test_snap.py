"""Snap-fit panels: clip/socket/key geometry, wall raise, box cavity behind relief, dodging light."""
import dataclasses

import numpy as np
import pytest
from manifold3d import Manifold

from megacube.config import load_config
from megacube.coords import to_model_frame
from megacube.export import mesh_arrays, write_stl
from megacube.geometry import Geometry, build_geometry, erode, fix_pinches, inscribed_box
from megacube.mapping import load_mapping
from megacube.primitives import box
from megacube.snap import KeySpec, SnapSpec, required_wall
from megacube.sources.synthetic import SyntheticCubeSource
from megacube.split import split_geometry
from megacube.validate import validate_arrays, validate_file

MAPPING = load_mapping()
CFG = load_config()
SPEC = SnapSpec.from_config(CFG)


def B(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box(hi - lo, (lo + hi) / 2)


# ---------------------------------------------------------------------------- local geometry
def test_clip_dimensions_follow_the_strain_limit():
    lever = SPEC.in_length + SPEC.catch_at + SPEC.land / 2
    assert 1.5 * SPEC.prong * SPEC.deflection / lever ** 2 == pytest.approx(SPEC.max_strain)
    assert SPEC.deflection == pytest.approx(SPEC.barb - SPEC.clearance)
    assert SPEC.required_wall() == pytest.approx(max(SPEC.depth, SPEC.thickness + SPEC.relief) + SPEC.skin)
    assert required_wall(CFG) == pytest.approx(3.4)
    assert required_wall(load_config(overrides=["split.panel_joint=pins"])) is None
    assert required_wall(CFG, "box_lid") is None
    assert SPEC.problems(0.4) == []
    weak = dataclasses.replace(SPEC, barb=0.3)
    assert any("won't hold" in p for p in weak.problems(0.4))
    jammed = dataclasses.replace(SPEC, split=0.6)
    assert any("too narrow" in p for p in jammed.problems(0.4))
    assert 5 < SPEC.insertion_force_n() < 40  # a firm press, not a hammer


def test_clip_engages_and_is_retained():
    w, c = SPEC.required_wall(), SPEC.clearance
    clip = SPEC.clip()
    carrier = B((-20, -15, 0), (0, 15, w)) - SPEC.pocket() + clip
    receiver = B((0, -15, -20), (w, 15, 10)) - SPEC.socket()
    assert clip.status().name == "NoError" and len(clip.decompose()) == 2  # two prongs
    assert (carrier ^ receiver).volume() == pytest.approx(0, abs=1e-9)
    assert clip.min_gap(receiver, 1.0) == pytest.approx(c, abs=1e-6)
    # pulled back by less than the clearance it's still free; by more, the barbs hit the lips
    assert (clip.translate((-0.9 * c, 0, 0)) ^ receiver).volume() == pytest.approx(0, abs=1e-12)
    assert (clip.translate((-1.5 * c, 0, 0)) ^ receiver).volume() > 1e-3
    # on the way in the barbs are wider than the neck: the prongs have to bend to pass it
    assert (clip.translate((-SPEC.catch_at - SPEC.land / 2, 0, 0)) ^ receiver).volume() > 1e-3
    # the prongs bend freely: past their root, the pocket leaves `relief` beside and above them
    free = B((-SPEC.in_length + 0.5, -15, -1), (0, 15, 10))
    walls = (B((-20, -15, 0), (0, 15, w)) - SPEC.pocket()) ^ free
    assert (clip ^ free).min_gap(walls, 1.0) == pytest.approx(SPEC.relief, abs=1e-6)
    # neither the socket nor the pocket breaks through the outer surface
    assert w - SPEC.depth >= SPEC.skin - 1e-9 and w - SPEC.thickness - SPEC.relief >= SPEC.skin - 1e-9


def test_key_slides_along_its_groove():
    k = KeySpec.from_config(CFG)
    receiver = B((0, -30, -10), (3.4, 30, 10)) - k.groove(-31, 31)
    tongue = k.tongue()
    assert tongue.min_gap(receiver, 1.0) == pytest.approx(k.clearance, abs=1e-6)
    for dv in (-12, -4, 4, 12):  # free to slide along the seam
        assert (tongue.translate((0, dv, 0)) ^ receiver).volume() == pytest.approx(0, abs=1e-12)
    assert (tongue.translate((-0.3, 0, 0)) ^ receiver).volume() == pytest.approx(0, abs=1e-12)  # not across it
    assert (tongue.translate((0, 0, 0.3)) ^ receiver).volume() > 0


# ---------------------------------------------------------------------------- cavity and wall
@pytest.mark.parametrize("name, cut, expect", [
    ("plain", None, ((3, 3, 3), (97, 97, 97))),
    ("top dent", ((40, 40, 97), (60, 60, 101)), ((3, 3, 3), (97, 97, 94))),
    ("edge notch", ((90, -1, 98), (101, 101, 101)), ((3, 3, 3), (97, 97, 95))),
    ("dent near an edge", ((97.5, 3, 3), (101, 6, 6)), ((3, 3, 3), (94.5, 97, 97))),
])
def test_inscribed_box_sits_behind_every_dent(name, cut, expect):
    core = B((0, 0, 0), (100, 100, 100))
    if cut is not None:
        core = core - B(*cut)
    eroded = erode(core, 3.0)
    b = inscribed_box(eroded)
    assert b == pytest.approx(np.array(expect, float), abs=1e-9), name
    assert (B(b[0], b[1]) - eroded).volume() == pytest.approx(0, abs=1e-9)


def test_inscribed_box_handles_a_groove_around_the_sides():
    core = B((0, 0, 0), (100, 100, 100)) - (B((-1, -1, 45), (101, 101, 55)) - B((2, 2, 40), (98, 98, 60)))
    b = inscribed_box(erode(core, 3.0))  # a ring: all four sides move in, the height stays
    assert b == pytest.approx(np.array([(5, 5, 3), (95, 95, 97)], float), abs=1e-9)


@pytest.fixture(scope="module")
def relief():
    geo = build_geometry(to_model_frame(SyntheticCubeSource(relief=True).read()), MAPPING, CFG)
    return geo, split_geometry(geo, CFG)


def test_cavity_is_a_box_behind_the_relief(relief):
    geo, _ = relief
    assert geo.cavity.num_tri() == 12  # flat inner faces
    sides = geo.report["wall"]["per_side_mm"]
    assert all(lo >= geo.wall - 1e-6 for lo, _, _ in sides.values())
    assert sides["+x"][1] > geo.wall + 1.0  # the recessed foundation pushes the whole +X face back
    assert sides["-x"][2] > geo.wall + 2.0  # offset foundations stand out of the face
    lo, hi = np.array(geo.cavity.bounding_box()).reshape(2, 3)
    grown = B(lo - geo.wall + 1e-6, hi + geo.wall - 1e-6)  # nowhere thinner than the wall
    assert (grown - geo.filled).volume() == pytest.approx(0, abs=1e-6)


def test_follow_cavity_keeps_the_dents():
    cfg = load_config(overrides=["hollow.cavity=follow", "split.mode=box_lid"])
    geo = build_geometry(to_model_frame(SyntheticCubeSource(relief=True).read()), MAPPING, cfg)
    assert geo.cavity.num_tri() > 12


def test_relief_panels_are_valid(relief):
    geo, (parts, rep) = relief
    assert geo.report["edge_contacts_joined"] > 0  # the checkerboard's corners touch
    assert (rep["snap"]["clips"], rep["snap"]["keys"]) == (16, 8) and not rep["warnings"]
    for p in parts:
        for g, s in p.solids.items():
            v, f = mesh_arrays(s)
            r = validate_arrays(f"{p.name}_{g}", v, f)
            assert r.ok, r


def test_wall_is_raised_for_snap_clips_only():
    raw = SyntheticCubeSource(features=False).read()
    g = build_geometry(to_model_frame(raw), MAPPING, load_config(overrides=["hollow.wall=2.4"]))
    assert g.wall == pytest.approx(3.4) and g.report["wall"]["configured_mm"] == 2.4 and g.report["wall"]["raised_for"]
    g = build_geometry(to_model_frame(raw), MAPPING, load_config(overrides=["hollow.wall=2.4", "split.mode=box_lid"]))
    assert g.wall == 2.4 and "raised_for" not in g.report["wall"]


def test_fix_pinches_joins_edge_contacts(tmp_path):
    a, b = B((0, 0, 0), (10, 10, 10)), B((10, 10, 0), (20, 20, 10))  # touching along the z edge at (10, 10)
    both = a + b
    write_stl(both, tmp_path / "pinched.stl")
    assert not validate_file(tmp_path / "pinched.stl").watertight  # the edge has four triangles in an STL
    fixed, n = fix_pinches(both)
    assert n > 0 and fixed.volume() == pytest.approx(both.volume(), abs=1e-3)
    write_stl(fixed, tmp_path / "fixed.stl")
    assert validate_file(tmp_path / "fixed.stl").ok
    assert fix_pinches(a) == (a, 0)


# ---------------------------------------------------------------------------- dodging light
def shell(light=None, size=100.0, wall=3.4):
    outer = B((0, 0, 0), (size,) * 3)
    cavity = B((wall,) * 3, (size - wall,) * 3)
    light = light if light is not None else Manifold()
    body = outer - cavity - light
    return Geometry(1.0, body, light, None, Manifold(), outer, cavity, np.zeros(3), np.eye(3), 0.0, {},
                    np.array([(0.0,) * 3, (size,) * 3]), wall)


def split_shell(light):
    geo = shell(light)
    parts, rep = split_geometry(geo, CFG)
    lit = sum(p.solids["light"].volume() for p in parts if "light" in p.solids and p.role == "part")
    assert lit == pytest.approx(light.volume(), rel=1e-9)  # no connector cut into the light pipe
    return parts, rep


def test_clips_slide_along_the_edge_around_a_light_pipe():
    x0 = 3.4 + (100 - 2 * 3.4) / 4  # where the front panel's first top clip would go
    pipe = B((x0 - 2.5, 0, 86), (x0 + 2.5, 3.4, 95))  # through the front wall, near the top edge
    parts, rep = split_shell(pipe)
    sn = rep["snap"]
    assert sn["clips"] == 16 and sn["clips_moved_to_dodge_light"] >= 1 and not rep["warnings"]
    front_top = [s["at_mm"][0] for s in sn["sites"] if (s["panel"], s["into"], s["kind"]) == ("front", "top", "clip")]
    assert min(abs(x - x0) for x in front_top) > 5


def test_clips_blocked_along_a_whole_edge_are_reported():
    strip = B((3, 0, 88), (97, 3.4, 95))  # light along the whole top edge of the front panel
    parts, rep = split_shell(strip)
    assert rep["snap"]["clips"] == 14 and len(rep["snap"]["clips_not_placed"]) == 2
    assert any("not placed" in w for w in rep["warnings"])


def test_keys_move_up_when_the_groove_would_cut_light():
    pipe = B((0.5, 0, 20), (6, 3.4, 30))  # through the front wall at the left seam, low down
    parts, rep = split_shell(pipe)
    keys = [s for s in rep["snap"]["sites"] if (s["kind"], s["panel"], s["into"]) == ("key", "left", "front")]
    assert len(keys) == 2 and min(s["at_mm"][2] for s in keys) > 30 + KeySpec.from_config(CFG).length / 2
    assert not rep["snap"]["seams_without_keys"]
