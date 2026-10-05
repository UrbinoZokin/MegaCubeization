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
| 4: scaling, printability checks, splitting | done; snap-fit locking panels added |
| 5: export, previews, validation | done |

Phases 2–5 run on a **synthetic test cube** (`synthetic`), not your build.

## Install

```bash
pip install -e ".[dev]"     # numpy, scipy, manifold3d, trimesh, pyyaml, pillow (+ pytest)
```

## Use

```bash
megacube build synthetic -o out                    # everything; default split: six snap-fit panels
megacube build synthetic:relief=1 -o out           # test cube with textured (offset-foundation) side faces
megacube build synthetic -o out --set split.mode=box_lid --set hollow.wall=2.4
megacube coverage synthetic                        # class mapping coverage report
megacube normalize synthetic -o build.json         # Phase 1 output format (mm, right-handed)
megacube validate out/parts/*.stl                  # mesh checks for any STL
megacube sniff megaprint_calculator.cbp            # identify an unknown file's encoding
```

Inputs: `synthetic` (options: `synthetic:n=8`, `synthetic:relief=1`), or a `megacube-build` JSON in
either frame. `.sav` and `.cbp` readers come in Phase 1. `relief=1` gives the side faces a checkerboard of
offset foundations, one recessed foundation and signs around the border of the base layer. That layout is
invented to test the pipeline; your real faces may differ.

### Outputs (`out/`)

| File | What |
|---|---|
| `summary.md` | one-page summary: coverage, scale, parts, checks, validation, warnings |
| `parts/<part>_<body,light,dark>.stl` | per part, per material, in print orientation; all files of a part share one origin, so import them together into Bambu Studio and accept "load as a single object with multiple parts" |
| `parts/<part>.3mf`, `parts/all_parts.3mf` | the materials as parts of one object, with Bambu-style extruder metadata (body 1, light 2, dark 3). **Not yet tested in Bambu Studio** |
| `parts/snap_coupon.3mf`, `parts/coupon_*.stl` | two small test pieces cut from the real panels around one clip: print these first |
| `assembled/*.stl` | the whole model in place (bottom at z = 0) |
| `assembled/debug_placeholders.stl`, `preview/debug_unmapped.png` | unmapped classes as placeholder boxes |
| `preview/*.png` | iso views, six orthographic face views, cut-away, parts in print orientation, `snap_joint.png` close-up of one clip |
| `coverage.txt`, `printability.txt`, `validation.txt`, `report.json` | details |

### Split modes (`split.mode`)

* `panels` (default): six panels, each printed inner face down. `split.panel_joint` picks the joints:
  * `snap` (default): the panels lock together, see below;
  * `pins`: diamond-section alignment pins (self-supporting 45° flanks) and sockets; glue the panels.
* `box_lid`: box printed upright, plus a flat lid printed inner face down. Corner posts carry round pins,
  and the lid has sockets with clearance.
* `one`: single print, bottom down. The cavity gets a 45° pyramid roof so it needs no internal
  supports. Light pipes through the roof get long, with knife edges where they meet it; elements over
  the middle of the top face may get none.

### Snap-fit panels

Top and bottom panels are full size. Front and back stand between them, and left and right sit
between front and back.

* **Clips** (2 per edge): on the top and bottom edges of the four side panels. A clip is a flat split
  arrowhead lying on the bed, so its prongs bend sideways along the layers, the strong direction. The
  prongs start about 5 mm inside the panel, in a relief pocket open to the cavity. That makes them long
  enough to bend at 2 % strain (`split.snap.max_strain`; PLA ~0.02, PETG ~0.03). The socket in the top
  or bottom panel is a neck, then a wider chamber. The barbs hook behind the neck's 0.8 mm lips with a
  0.45 mm catch.
* **Keys** (2 per seam): short tongues on the left and right panels' vertical edges. They slide down
  grooves in the front and back panels' inner faces and keep those seams flush.
* Every connector keeps `split.clearance` (0.15 mm) to its socket once assembled. The tests check
  this, and that the reassembled parts don't overlap.
* Clips and keys slide along their edge to avoid light pipes and holes. Clips that can't be placed are
  reported. A key groove that would cut light moves its keys up, or the seam is reported.
* The clips need a 3.4 mm wall: 2.6 mm socket plus 0.8 mm skin. `hollow.wall` is raised to that
  automatically and the summary says so. Relief on a face only adds thickness: the cavity is a box set
  back behind the deepest point of each face, so every panel's inner face is flat. The summary lists
  wall thickness per side.

**Print** every panel inner face down with **supports off**: the pocket roofs and socket ceilings are
short bridges. Keep the slicer's elephant-foot compensation on, because the socket necks are in the
first layers. **Print `parts/snap_coupon.3mf` first** (two 20 mm pieces cut from the real panels) and
push them together. If the fit is too tight, raise `split.clearance` or lower `split.snap.barb`. If it's
too loose, do the opposite.

**Assemble** (every clip engages with the same downward push):
1. Lay the bottom panel on the table, inner face up.
2. Press the front and back panels down onto it until all their bottom clips click.
3. Slide the left and right panels down between them. Their keys run in the front/back grooves, and
   their bottom clips click into the bottom panel.
4. Put the LED in through the bottom opening (it stays reachable through it).
5. Press the top panel down evenly until its 8 clips click. Expect about 14 N per clip (a rough PLA
   estimate), so roughly 110 N for the whole top. The catches are square and don't release: test-fit
   the coupon first.

Check visually: the clips' relief pockets and sockets leave 0.8–1.3 mm of skin to the outside. With a
light-coloured body filament, they may show as faint glow spots when the LED is on. A dark or opaque
body filament avoids that.

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
   (protrusions stay solid). The cavity is the largest box that keeps the full wall behind every dent,
   and windows/light pipes go behind every light element. Booleans keep body, light and dark from ever
   overlapping. Pieces that touch only along an edge (offset foundations corner to corner) are joined
   with a 4 µm rod, so the STL stays manifold.
4. **Split and check** (`split.py`, `snap.py`, `printability.py`): thin walls (< 0.8 mm) and gaps
   (< 0.4 mm) are reported with locations. Designed small features (the clip barbs' 0.4 mm lands) are
   listed separately. Overhangs are classed as short ledges, bridges (flat ceilings ≤ 6 mm between
   walls), plate supports (allowed) or supports on the model (internal, reported). Every part is checked
   against the build volume.
5. **Export and validate** (`export.py`, `validate.py`, `preview.py`): binary STL and 3MF. Every
   written STL is re-read and checked for watertightness, winding, outward normals, degenerate
   triangles and self-intersections.

## Tests

```bash
pytest -q        # ~130 tests, ~75 s
```

Phase 0 research scripts (public sample saves only): [`research/`](research/README.md).
