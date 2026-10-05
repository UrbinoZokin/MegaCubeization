"""Phase 2: class -> primitive mapping and coverage."""
from pathlib import Path

import pytest

from megacube.mapping import MappingError, coverage, load_mapping
from megacube.model import Build, BuildObject, Transform
from megacube.sources.synthetic import SyntheticCubeSource

FIXTURE = Path(__file__).parent / "fixtures" / "public_save_classes.tsv"


@pytest.fixture(scope="module")
def mapping():
    return load_mapping()


@pytest.mark.parametrize("cls, rule, group", [
    ("Build_Foundation_8x1_01_C", "foundation_8x1", "body"),
    ("Build_Foundation_8x2_01_C", "foundation_8x2", "body"),
    ("Build_Foundation_8x4_01_C", "foundation_8x4", "body"),
    ("Build_Foundation_Asphalt_8x4_C", "foundation_8x4", "body"),
    ("Build_Foundation_ConcretePolished_8x2_2_C", "foundation_8x2", "body"),
    ("Build_Foundation_Metal_8x1_C", "foundation_8x1", "body"),
    ("Build_FoundationGlass_01_C", "foundation_glass", "body"),
    ("Build_Foundation_Frame_01_C", "foundation_frame", "body"),
    ("Build_Ramp_8x2_01_C", "ramp_8x2", "body"),
    ("Build_Ramp_Polished_8x4_C", "ramp_8x4", "body"),
    ("Build_Ramp_Diagonal_8x1_01_C", "ramp_diagonal", "body"),  # must not be caught by ramp_8x1
    ("Build_RampDouble_Concrete_8x2_C", "ramp_double_8x2", "body"),
    ("Build_InvertedRamp_Concrete_8x4_C", "ramp_inverted_8x4", "body"),
    ("Build_Ramp_UpCorner_Metal_8x2_C", "ramp_corner_up", "body"),
    ("Build_QuarterPipe_Asphalt_8x4_C", "quarter_pipe", "body"),
    ("Build_QuarterPipeCorner_01_C", "quarter_pipe_corners", "body"),
    ("Build_Wall_8x4_01_C", "wall_8x4", "body"),
    ("Build_Wall_Window_8x4_02_C", "wall_window_8x4", "body"),  # specific window rule wins
    ("Build_Wall_Concrete_8x4_Window_01_C", "wall_window_8x4", "body"),
    ("Build_Wall_Door_8x4_01_Steel_C", "wall_8x4", "body"),
    ("Build_Wall_Orange_8x1_C", "wall_8x1", "body"),
    ("Build_Wall_Frame_01_C", "wall_frame", "body"),
    ("Build_Wall_Orange_FlipTris_8x4_C", "wall_tris_8x4", "body"),
    ("Build_ConveyorBeltMk6_C", "conveyor_belt", "light"),
    ("Build_ConveyorLiftMk1_C", "conveyor_lift", "light"),
    ("Build_StandaloneWidgetSign_SmallVeryWide_C", "sign_label_4m", "light"),
    ("Build_StandaloneWidgetSign_Square_Tiny_C", "sign_square_05m", "light"),
    ("Build_ConveyorPoleStackable_C", "conveyor_supports", "exclude"),
])
def test_required_families_resolve(mapping, cls, rule, group):
    r = mapping.resolve(cls)
    assert r is not None and (r.name, r.group) == (rule, group)


def test_unknown_class_is_unmapped(mapping):
    assert mapping.resolve("Build_PowerPoleMk1_C") is None
    assert mapping.resolve("Build_Wall_10_C") is None  # modded


def test_point_sets_and_profiles_are_resolved(mapping):
    ramp = mapping.resolve("Build_Ramp_8x4_01_C")
    assert len(ramp.params["points"]) == 6 and ramp.params["size"] == [800, 800, 400]
    qp = mapping.resolve("Build_QuarterPipe_C")
    assert isinstance(qp.params["profile"], list) and len(qp.params["profile"]) >= 4


def test_public_vocabulary_has_no_unmapped_structural_family(mapping):
    """Every vanilla foundation/ramp/wall/frame/window/belt/lift/sign class seen in public saves
    (or listed by the parser) must hit a rule. Foundation passthroughs (holes) are not foundations."""
    import re
    fam = re.compile(r"^Build_(Foundation(?!Passthrough)|FoundationGlass|Ramp|RampDouble|RampInverted|InvertedRamp|Wall_|"
                     r"SteelWall|Stair|QuarterPipe|DownQuarterPipe|ConveyorBelt|ConveyorLift|StandaloneWidgetSign|Flat_Frame)")
    names = [ln.split("\t")[0] for ln in FIXTURE.read_text().splitlines() if ln and not ln.startswith("#")]
    missing = [n for n in names if fam.match(n) and mapping.resolve(n) is None]
    assert missing == []


def test_coverage_report_on_synthetic(mapping):
    rep = coverage(SyntheticCubeSource().read(), mapping)
    d = rep.to_dict()
    assert d["total_objects"] == 231
    assert d["by_group"] == {"body": 218, "light": 7, "exclude": 4, "UNMAPPED": 2}
    assert d["unmapped_classes"] == {"Build_PowerPoleMk1_C": 1, "Build_Wall_10_C": 1}
    assert abs(d["mapped_fraction"] - 229 / 231) < 1e-4
    text = rep.format_text()
    assert "WARNING: 2 unmapped class(es)" in text and "Build_Wall_10_C" in text


def _write(tmp_path, body):
    p = tmp_path / "classes.yaml"
    p.write_text(body)
    return p


@pytest.mark.parametrize("body, msg", [
    ("rules:\n - {name: a, match: x, group: nope, primitive: box, size: [1,1,1]}\n", "group must be"),
    ("rules:\n - {name: a, match: x, group: body, primitive: box}\n", "needs 'size'"),
    ("rules:\n - {name: a, match: x, group: body, primitive: hull, size: [1,1,1], points: nope}\n", "unknown point set"),
    ("rules:\n - {name: a, match: '(', group: body, primitive: box, size: [1,1,1]}\n", "bad regex"),
    ("rules:\n - {name: a, match: x, group: body, primitive: box, size: [1,1,1]}\n"
     " - {name: a, match: y, group: body, primitive: box, size: [1,1,1]}\n", "duplicate"),
])
def test_validation_errors(tmp_path, body, msg):
    with pytest.raises(MappingError, match=msg):
        load_mapping(_write(tmp_path, body))


def test_first_match_wins(tmp_path):
    m = load_mapping(_write(tmp_path, "rules:\n"
                            " - {name: specific, match: 'Build_X_C', group: light, primitive: panel, size: [1,1], thickness: 1}\n"
                            " - {name: generic, match: 'Build_.*_C', group: body, primitive: box, size: [1,1,1]}\n"))
    assert m.resolve("Build_X_C").name == "specific" and m.resolve("Build_Y_C").name == "generic"


def test_coverage_counts_every_object(mapping):
    objs = [BuildObject(f"o{i}", c, Transform()) for i, c in enumerate(["Build_A_C"] * 3 + ["Build_Foundation_8x4_01_C"])]
    rep = coverage(Build(objs), mapping)
    assert rep.total == 4 and rep.count("UNMAPPED") == 3 and rep.count("body") == 1
