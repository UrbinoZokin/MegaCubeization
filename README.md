# MegaCubeization

Turns a Satisfactory build (a big floating cube of foundations/walls with belts and signs on its
faces) into a printable, back-lit display model for a Bambu X1C with AMS:

* an **opaque body** (a hollow shell with an LED/cable opening in the bottom face),
* a translucent **light body** (belts, lifts, signs) inlaid into it, with light pipes through the
  shell so an LED inside lights every element,
* optionally a dark inner lining against light bleed.

All geometry is generated procedurally from boxes, wedges, swept profiles and panels, using only
the class, transform and spline data of the build. No game meshes, textures or assets are used.

## Status

| Phase | State |
|---|---|
| 0: format research | done (public sources only): [`docs/phase0_research.md`](docs/phase0_research.md) |
| 1: parse the real file | **waiting for your files and go-ahead.** The parser interface, unit conversion and intermediate JSON are done |
| 2: class → primitive mapping | done: [`config/classes.yaml`](config/classes.yaml), coverage report |
| 3: geometry | done |
| 4: scaling, printability checks, splitting | done |
| 5: export, previews, validation | done |

Phases 2–5 run on a **synthetic test cube** (`synthetic`), not your build.

## Install

```bash
pip install -e ".[dev]"     # numpy, scipy, manifold3d, trimesh, pyyaml, pillow (+ pytest)
```

## Use

```bash
megacube build synthetic -o out                    # everything, default split: box + lid
megacube build synthetic -o out --set split.mode=panels --set hollow.wall=2.4
megacube coverage synthetic                        # class mapping coverage report
megacube normalize synthetic -o build.json         # Phase 1 output format (mm, right-handed)
megacube validate out/parts/*.stl                  # mesh checks for any STL
megacube sniff megaprint_calculator.cbp            # identify an unknown file's encoding
```

Inputs: `synthetic` (options: `synthetic:n=8`), or a `megacube-build` JSON in either frame. `.sav`
and `.cbp` readers come in Phase 1.

### Outputs (`out/`)

| File | What |
|---|---|
| `summary.md` | one-page summary: coverage, scale, parts, checks, validation, warnings |
| `parts/<part>_<body,light,dark>.stl` | per part, per material, in print orientation; all files of a part share one origin, so import them together into Bambu Studio and accept "load as a single object with multiple parts" |
| `parts/<part>.3mf`, `parts/all_parts.3mf` | the materials as parts of one object, with Bambu-style extruder metadata (body 1, light 2, dark 3). **Not yet tested in Bambu Studio** |
| `assembled/*.stl` | the whole model in place (bottom at z = 0) |
| `assembled/debug_placeholders.stl`, `preview/debug_unmapped.png` | unmapped classes as placeholder boxes |
| `preview/*.png` | iso views, six orthographic face views, cut-away, parts in print orientation |
| `coverage.txt`, `printability.txt`, `validation.txt`, `report.json` | details |

### Split modes (`split.mode`)

* `one`: single print, bottom down. The cavity gets a 45° pyramid roof so it needs no internal
  supports. Light pipes through the roof get long; elements over the middle of the top face may get none.
* `box_lid`: box printed upright, plus a flat lid printed inner face down. Corner posts carry round pins,
  and the lid has sockets with clearance.
* `panels`: six panels printed inner face down. Diamond-section pins (self-supporting 45° flanks)
  on the joint edges, sockets in the neighbouring panel.

## Configuration

* [`config/pipeline.yaml`](config/pipeline.yaml): every parameter (target size 240 mm, shell
  thickness, minimum printed belt/sign size, attach mode, windows, dark layer, LED opening, pins,
  check limits), overridable with `--set key=value` or `--config my.yaml`.
* [`config/classes.yaml`](config/classes.yaml): class → primitive rules (regex, size, offset,
  group), each marked `measured` / `partial` / `unverified`.

## How it works

1. **Parse** (Phase 1, pending) → `Build` in the game frame (cm, left-handed, Z-up).
   `coords.to_model_frame` mirrors Y and converts to mm. Rotations map (x,y,z,w) → (-x,y,-z,w).
   Tests check that an asymmetric marker lands on the correct face and a right turn stays a right turn.
2. **Map** each class to a primitive (`mapping.py`). Unknown classes are counted, warned about and
   drawn as placeholders, never silently dropped.
3. **Geometry** (`geometry.py`, manifold3d): snap pieces to a 1 cm grid, then batch union. Belts
   are swept along the game's Hermite spline as one tube mesh. Light elements are scaled up to printable
   minimums and moved onto the surface by raycast. The shell is hollowed inside the cube's own face planes
   (protrusions stay solid), and windows/light pipes go behind every light element. Booleans keep body,
   light and dark from ever overlapping.
4. **Split and check** (`split.py`, `printability.py`): thin walls (< 0.8 mm) and gaps (< 0.4 mm)
   are reported with locations. Overhangs are classed as plate supports (allowed) or supports on the
   model (internal, reported). Every part is checked against the build volume.
5. **Export and validate** (`export.py`, `validate.py`, `preview.py`): binary STL and 3MF. Every
   written STL is re-read and checked for watertightness, winding, outward normals, degenerate
   triangles and self-intersections.

## Tests

```bash
pytest -q        # ~110 tests, ~75 s
```

Phase 0 research scripts (public sample saves only): [`research/`](research/README.md).
