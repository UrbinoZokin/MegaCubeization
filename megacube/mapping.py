"""Class -> primitive mapping (Phase 2): load, validate, resolve, and report coverage."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .model import Build

DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / "config" / "classes.yaml"

GROUPS = ("body", "light", "exclude")
VERIFIED = ("measured", "partial", "unverified")
# required parameters per primitive (beyond name/match/group)
PRIMITIVE_PARAMS = {
    "box": ("size",),
    "hull": ("size", "points"),
    "extrude": ("size", "profile"),
    "belt": ("width", "thickness"),
    "lift": ("footprint",),
    "panel": ("size", "thickness"),
}


class MappingError(ValueError):
    pass


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    group: str
    primitive: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    verified: str = "unverified"
    notes: str = ""

    def offset(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.params.get("offset", (0.0, 0.0, 0.0)))


@dataclass
class Mapping:
    rules: list[Rule]
    unmapped_policy: str = "exclude"
    placeholder_size: tuple[float, float, float] = (100.0, 100.0, 100.0)
    path: str = ""
    _cache: dict[str, Rule | None] = field(default_factory=dict, repr=False)

    def resolve(self, class_name: str) -> Rule | None:
        if class_name not in self._cache:
            self._cache[class_name] = next((r for r in self.rules if r.pattern.fullmatch(class_name)), None)
        return self._cache[class_name]


def load_mapping(path: str | Path | None = None) -> Mapping:
    path = Path(path) if path else DEFAULT_CLASSES
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or "rules" not in doc:
        raise MappingError(f"{path}: expected a mapping with a 'rules' list")
    point_sets = doc.get("point_sets", {}) or {}
    profiles = doc.get("profiles", {}) or {}
    unmapped = doc.get("unmapped", {}) or {}
    policy = unmapped.get("policy", "exclude")
    if policy not in ("exclude", "body"):
        raise MappingError(f"{path}: unmapped.policy must be 'exclude' or 'body', got {policy!r}")

    rules, names = [], set()
    for i, raw in enumerate(doc["rules"]):
        where = f"{path}: rule #{i + 1} ({raw.get('name', '?')})"
        for key in ("name", "match", "group"):
            if key not in raw:
                raise MappingError(f"{where}: missing '{key}'")
        if raw["name"] in names:
            raise MappingError(f"{where}: duplicate rule name")
        names.add(raw["name"])
        if raw["group"] not in GROUPS:
            raise MappingError(f"{where}: group must be one of {GROUPS}")
        verified = raw.get("verified", "unverified")
        if verified not in VERIFIED:
            raise MappingError(f"{where}: verified must be one of {VERIFIED}")
        try:
            pattern = re.compile(raw["match"])
        except re.error as exc:
            raise MappingError(f"{where}: bad regex: {exc}") from exc
        params = {k: v for k, v in raw.items() if k not in ("name", "match", "group", "primitive", "verified", "notes")}
        primitive = raw.get("primitive")
        if raw["group"] != "exclude":
            if primitive not in PRIMITIVE_PARAMS:
                raise MappingError(f"{where}: primitive must be one of {sorted(PRIMITIVE_PARAMS)}")
            for key in PRIMITIVE_PARAMS[primitive]:
                if key not in params:
                    raise MappingError(f"{where}: primitive '{primitive}' needs '{key}'")
            if primitive == "hull" and isinstance(params["points"], str):
                if params["points"] not in point_sets:
                    raise MappingError(f"{where}: unknown point set {params['points']!r}")
                params["points"] = point_sets[params["points"]]
            if primitive == "extrude" and isinstance(params["profile"], str):
                if params["profile"] not in profiles:
                    raise MappingError(f"{where}: unknown profile {params['profile']!r}")
                params["profile"] = profiles[params["profile"]]
        rules.append(Rule(raw["name"], pattern, raw["group"], primitive, params, verified, raw.get("notes", "")))
    return Mapping(rules, policy, tuple(float(v) for v in unmapped.get("placeholder_size", (100, 100, 100))), str(path))


# ---------------------------------------------------------------------------------- coverage
@dataclass
class ClassLine:
    class_name: str
    count: int
    rule: str | None
    group: str  # body | light | exclude | UNMAPPED
    verified: str | None


@dataclass
class Coverage:
    total: int
    lines: list[ClassLine]
    policy: str

    def count(self, group: str) -> int:
        return sum(line.count for line in self.lines if line.group == group)

    @property
    def mapped_fraction(self) -> float:
        """Share of objects matched by a rule (printed or deliberately excluded)."""
        return 1.0 - self.count("UNMAPPED") / self.total if self.total else 1.0

    @property
    def printed_fraction(self) -> float:
        return (self.count("body") + self.count("light")) / self.total if self.total else 0.0

    def unmapped(self) -> list[ClassLine]:
        return [line for line in self.lines if line.group == "UNMAPPED"]

    def unverified_in_use(self) -> list[ClassLine]:
        return [line for line in self.lines if line.group in ("body", "light") and line.verified != "measured"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_objects": self.total,
            "mapped_fraction": round(self.mapped_fraction, 4),
            "printed_fraction": round(self.printed_fraction, 4),
            "by_group": {g: self.count(g) for g in ("body", "light", "exclude", "UNMAPPED")},
            "unmapped_policy": self.policy,
            "unmapped_classes": {line.class_name: line.count for line in self.unmapped()},
            "not_fully_verified_rules_in_use": sorted({f"{line.rule} ({line.verified})" for line in self.unverified_in_use()}),
            "classes": [vars(line) for line in self.lines],
        }

    def format_text(self) -> str:
        out = [f"Coverage: {self.total} objects in {len(self.lines)} classes"]
        for g, label in (("body", "body (opaque)"), ("light", "light (translucent)"),
                         ("exclude", "excluded by rule"), ("UNMAPPED", "UNMAPPED")):
            n = self.count(g)
            pct = 100.0 * n / self.total if self.total else 0.0
            out.append(f"  {label:<22}{n:7d}  {pct:6.2f}%")
        out.append(f"  mapped by a rule: {100 * self.mapped_fraction:.2f}%   printed: {100 * self.printed_fraction:.2f}%")
        out.append("")
        out.append(f"  {'count':>7}  {'group':<9} {'rule':<26} {'verified':<11} class")
        for line in self.lines:
            out.append(f"  {line.count:7d}  {line.group:<9} {line.rule or '-':<26} {line.verified or '-':<11} {line.class_name}")
        if self.unmapped():
            action = "excluded from print, drawn as placeholders in the debug output" if self.policy == "exclude" \
                else "printed as placeholder boxes in the body"
            out.append("")
            out.append(f"WARNING: {len(self.unmapped())} unmapped class(es), {self.count('UNMAPPED')} object(s) ({action}):")
            for line in self.unmapped():
                out.append(f"  {line.count:7d}  {line.class_name}")
        if self.unverified_in_use():
            out.append("")
            out.append("Note: rules in use whose dimensions/pivots are not fully verified (check the previews):")
            for name in sorted({f"{line.rule} ({line.verified})" for line in self.unverified_in_use()}):
                out.append(f"  - {name}")
        return "\n".join(out)


def coverage(build: Build, mapping: Mapping) -> Coverage:
    lines = []
    for cls, n in build.class_counts().items():
        rule = mapping.resolve(cls)
        if rule is None:
            lines.append(ClassLine(cls, n, None, "UNMAPPED", None))
        else:
            lines.append(ClassLine(cls, n, rule.name, rule.group, None if rule.group == "exclude" else rule.verified))
    order = {"UNMAPPED": 0, "light": 1, "body": 2, "exclude": 3}
    lines.sort(key=lambda line: (order[line.group], -line.count, line.class_name))
    return Coverage(len(build.objects), lines, mapping.unmapped_policy)
