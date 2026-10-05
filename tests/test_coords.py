"""Unit conversion and orientation: game (cm, left-handed, Z-up) -> model (mm, right-handed, Z-up)."""
import numpy as np
import pytest

from megacube.cli import main
from megacube.coords import local_to_model, quat_to_model, to_model_frame, transform_to_model, vec_to_model
from megacube.model import Build, Transform, matrix_to_quat, quat_to_matrix
from megacube.sources.base import open_source, select_objects
from megacube.sources.synthetic import SyntheticCubeSource

RNG = np.random.default_rng(1234)


def random_quat():
    q = RNG.normal(size=4)
    return tuple(q / np.linalg.norm(q))


def test_vectors_and_quaternions():
    assert vec_to_model((1, 2, 3)) == (10, -20, 30)
    assert quat_to_model((0.1, 0.2, 0.3, 0.9)) == (-0.1, 0.2, -0.3, 0.9)


def test_matrix_quat_roundtrip():
    for _ in range(50):
        q = random_quat()
        q2 = matrix_to_quat(quat_to_matrix(q))
        assert np.allclose(quat_to_matrix(q2), quat_to_matrix(q), atol=1e-12)


def test_converting_points_equals_converting_transform_and_local_geometry():
    """model(R v + t) == R' (S v * 10) + t': we may convert transforms and local shapes separately."""
    for _ in range(50):
        t = Transform(random_quat(), tuple(RNG.uniform(-1e5, 1e5, 3)), tuple(RNG.uniform(0.5, 2, 3)))
        v = RNG.uniform(-800, 800, (10, 3))
        world_game = (t.matrix() @ np.c_[v, np.ones(10)].T).T[:, :3]
        expected = np.array([vec_to_model(p) for p in world_game])
        tm = transform_to_model(t)
        got = (tm.matrix() @ np.c_[local_to_model(v), np.ones(10)].T).T[:, :3]
        assert np.allclose(got, expected, atol=1e-6)
        assert np.isclose(np.linalg.det(quat_to_matrix(tm.rotation)), 1.0)  # still a proper rotation


def test_yaw_turns_the_same_way_for_the_observer():
    """+90 degree yaw in the game turns +X toward +Y = the observer's right (left-handed frame).
    In the model it must turn +X toward the observer's right too, which in a right-handed Z-up frame
    is forward x up = -Y."""
    yaw90 = (0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    assert np.allclose(quat_to_matrix(yaw90) @ [1, 0, 0], [0, 1, 0])
    right_model = np.cross([1, 0, 0], [0, 0, 1])
    assert np.allclose(quat_to_matrix(quat_to_model(yaw90)) @ [1, 0, 0], right_model)
    assert np.allclose(right_model, [0, -1, 0])


@pytest.fixture(scope="module")
def cube():
    raw = SyntheticCubeSource(n=6).read()
    return raw, to_model_frame(raw)


def test_marker_sign_lands_on_correct_face(cube):
    """The asymmetric marker sign is on the game's -Y face (the observer's LEFT when facing +X),
    near the +X edge, in the lower half. Not mirrored means: still on the observer's left (model +Y),
    still near +X, still low, and still facing outward."""
    raw, model = cube
    half_mm = 6 * 800 / 2 * 10
    marker = next(o for o in model.objects if o.props.get("role") == "orientation-marker")
    x, y, z = marker.transform.translation
    left_model = np.cross([0, 0, 1], [1, 0, 0])  # up x forward = observer's left in a right-handed frame
    assert np.allclose(left_model, [0, 1, 0])
    assert y > half_mm  # on the +Y (left) face, just outside the surface
    assert x > 0.5 * half_mm and z < 0  # near the +X edge, lower half
    # the sign's front (local +Y in the game) must point away from the cube
    front_game = quat_to_matrix(next(o for o in raw.objects if o.props.get("role")).transform.rotation) @ [0, 1, 0]
    front_model = quat_to_matrix(marker.transform.rotation) @ (local_to_model([0, 1, 0]) / 10)
    assert np.allclose(front_game, [0, -1, 0]) and np.allclose(front_model, [0, 1, 0])


def test_l_belt_still_turns_right(cube):
    """The L-shaped top belt runs +X and then turns to the game's +Y, a right turn. After conversion
    it must still be a right turn, i.e. end heading toward model -Y."""
    _, model = cube
    belt = next(o for o in model.objects if o.class_name == "Build_ConveyorBeltMk3_C")
    R = quat_to_matrix(belt.transform.rotation)
    first = R @ np.asarray(belt.spline[0].leave_tangent)
    last = R @ np.asarray(belt.spline[-1].arrive_tangent)
    first, last = first / np.linalg.norm(first), last / np.linalg.norm(last)
    assert np.allclose(first, [1, 0, 0], atol=1e-9)
    assert np.allclose(last, np.cross(first, [0, 0, 1]), atol=1e-9)  # forward x up = right


def test_units_cube_size(cube):
    _, model = cube
    pts = np.array([o.transform.translation for o in model.objects if o.class_name.startswith("Build_Foundation")])
    # foundation centres are 0.5 m inside the 48 m cube's outer faces -> +/- 23.5 m = 23500 mm
    assert np.isclose(pts.max(), 23500) and np.isclose(pts.min(), -23500)


def test_json_roundtrip(tmp_path, cube):
    _, model = cube
    path = tmp_path / "b.json"
    model.save(path)
    again = Build.load(path)
    assert again.frame == "model" and len(again.objects) == len(model.objects)
    a = next(o for o in again.objects if o.spline)
    b = next(o for o in model.objects if o.spline)
    assert np.allclose(np.array([p.location for p in a.spline]), np.array([p.location for p in b.spline]))
    assert np.allclose(a.transform.rotation, b.transform.rotation)


def test_select_by_bbox(cube):
    raw, _ = cube
    top_only = select_objects(raw, bbox=(-3000, -3000, 2300, 3000, 3000, 3000))
    assert all(o.transform.translation[2] >= 2300 for o in top_only.objects)
    assert top_only.source["selection"]["kept"] == len(top_only.objects) < len(raw.objects)


def test_normalize_cli(tmp_path):
    out = tmp_path / "build.json"
    assert main(["normalize", "synthetic:n=4", "-o", str(out)]) == 0
    b = Build.load(out)
    assert b.frame == "model" and b.source["kind"] == "synthetic"


def test_sav_is_phase1(tmp_path):
    with pytest.raises(NotImplementedError):
        open_source(str(tmp_path / "x.sav")).read()
