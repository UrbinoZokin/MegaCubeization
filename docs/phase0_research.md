# Phase 0: format investigation (research half)

Status: **research only**. The megaprint and `.sav` aren't available yet, so nothing here
describes *your* files. Every per-object fact below was either read from parser source code or
measured on **public sample saves** shipped in the parser's own test folder (5 saves, 1.0 and 1.1,
170,306 placed objects). Reproduce with the scripts in [`research/`](../research/README.md).

## 1. Preliminary recommendation (to confirm once the files arrive)

| Decision | Recommendation | Why |
|---|---|---|
| Input path | **The `.sav`**, selecting the cube's objects by bounding box (or by blueprint-proxy reference if it was placed as a blueprint) | The `.sav` format has two maintained, open parsers; the megaprint format has no public spec or parser (§3.2) |
| Parser | **`@etothepii/satisfactory-file-parser`** (Node/TypeScript, MIT), wrapped by a small extraction script that emits compact JSON for Python | Supports U6–U1.2 saves, doesn't crash on unknown/modded data (keeps raw bytes instead), and parses 0.5–6 MB saves in 0.5–2 s here |
| Fallback / cross-check | GreyHak `sat_sav_parse` (Python, GPL-3.0) | Good second opinion; but GPL, pinned to specific save versions, and it asserts unit scale on lightweight buildables (see §2) |
| Megaprint | Run `megacube sniff <file>` first to identify the container/encoding, then decide | I won't guess the structure of a file I haven't seen |

The Node dependency follows your rule ("Node only if the best parser requires it, wrapped as a
small script called from Python"). There's no maintained Python parser with an equally permissive
license and version range.

## 2. Parser libraries

| Library | Lang | License | Version checked | Formats | Game versions | Notes |
|---|---|---|---|---|---|---|
| `satisfactory-file-parser` (unscoped) | — | — | — | — | — | **Does not exist** on npm (`404`). The real package is scoped ↓ |
| [`@etothepii/satisfactory-file-parser`](https://github.com/etothepii4/satisfactory-file-parser) | TS/JS | MIT | 4.1.2 (2026‑07‑26) | `.sav`, `.sbp`, `.sbpcfg` (read **and** write) | U6/U7 "mostly", U8, 1.0, 1.1, 1.2 | Only dependency: `pako`. `throwErrors:false` (default since 3.3) keeps unparseable properties as `rawBytes`/`trailingData` instead of aborting. **Verified** by running it on 5 public saves (save versions 46 and 52) |
| [GreyHak `sat_sav_parse`](https://github.com/GreyHak/sat_sav_parse) | Python | GPL‑3.0 | `bf490d3` (2026‑08‑15) | `.sav`, `.sbp`, `.sbpcfg` | current release: save versions 52/53/58/59/60 (v1.1.0.4–v1.2.2.1); older via older releases | Usable as a library (`readFullSaveFile`). Detects SCIM-edited saves. Raises on lightweight buildables with non-unit scale or non-empty material/skin refs |
| `satisfactory-save-reader` (PyPI) | Python | GPL‑3.0 | 0.8.2 (2024‑05) | `.sav` | pre-1.0 | Its README says it fails on big saves. **Not suitable** |
| `sav2json`, `satisfactory-json` (npm), `ficsit-toolkit` (Go) | — | — | — | `.sav` | old (pre-1.0) | Not evaluated further |
| **Megaprint (`.cbp`) parser** | — | — | — | — | — | **None found** on npm, PyPI or GitHub search. Neither parser above mentions megaprints (checked by grepping their sources) |

## 3. File formats

### 3.1 `.sav` (from parser source; confirmed by running on the samples)

```
file = header + body chunks
header:   saveHeaderType (13 = 1.0, 14 = 1.1+), saveVersion (46 = 1.0, 51–53 = 1.1, 58–60 = 1.2),
          buildVersion, [saveName (1.1+)], mapName, mapOptions, sessionName, playDuration,
          saveDateTime (.NET ticks), sessionVisibility, editorObjectVersion, modMetadata,
          isModdedSave, saveIdentifier, isPartitionedWorld, saveDataHash, isCreativeModeEnabled
chunks:   repeated { 0x9E2A83C1 (UE package tag), 0x22222222, 0x00, maxChunkSize,
                     0x03000000 (zlib), compressedSize (u64), uncompressedSize (u64), [repeated],
                     zlib stream }
body:     u64 size, [SaveObjectVersionData (1.1+)], partition/grid table,
          N sub-levels + persistent level, each:
            object headers: actor  = class path, level, instance name, needTransform,
                                     rotation quat (4×float32), position (3×float32),
                                     scale (3×float32), wasPlacedInLevel
                            component = class path, level, instance name, parent actor
            object data:    Unreal property-tag stream + class-specific trailing data
          then collectables, destroyed actors, unresolved references
```

**Important for this project:** since 1.0, most structural pieces (foundations, walls, ramps,
beams, …) are *not* individual actors. They're stored inside one object,
`/Script/FactoryGame.FGLightweightBuildableSubsystem`, as per-class instance arrays with a
**double-precision** transform (actors use float32). In the samples that holds 47 structural classes
(about 140k of the 170k placed objects). The extractor has to read both places.

SCIM-edited saves deviate slightly from what the game writes (extra trailing bytes, missing
final arrays). GreyHak's parser detects this. The etothepii parser tolerates it.

### 3.2 Megaprint (SCIM)

What public sources say: SCIM's "download selected" exports a file named
`megaprint_calculator.cbp`, which is imported back by dragging it onto SCIM's import/export
screen. That's all I could establish:

- no public spec, no open-source reader, and neither parser supports it;
- `satisfactory-calculator.com`, `satisfactory.wiki.gg`, `steamcommunity.com`, `supercraft.host` and
  `answeroverflow.com` were **blocked by this environment's network policy**, so I couldn't read
  SCIM's own help pages or forum threads (see §7);
- third-party "file extension" sites claim `.cbp` is zlib-compressed, but they're generic SEO
  pages. I'm **not** relying on that.

Plan: `megacube sniff <file>` (implemented) reports magic bytes, whether the file is
zlib/gzip/deflate/UE-chunked/JSON/base64, and what's inside after decoding, without assuming a
schema. Based on that we pick: (a) parse the megaprint directly if it's simple (e.g. JSON),
or (b) use the `.sav` path.

### 3.3 Blueprints (`.sbp` + `.sbpcfg`)

For reference only: both parsers read them, and they use the same object/property encoding as saves.
Not one of your inputs.

## 4. Fields available per object (verified on the public samples)

| Field | Where | Notes |
|---|---|---|
| Class name | `typePath`, e.g. `/Game/FactoryGame/Buildable/Building/Foundation/Build_Foundation_8x4_01.Build_Foundation_8x4_01_C` | same for actors and lightweight instances. Modded classes use other roots (`/MoreDecorations/...`, `/FicsitWiremod/...`) |
| Transform | actor header (float32) or lightweight instance (float64) | `rotation` quaternion (x,y,z,w), `translation` (cm), `scale3d`. Scale was ≠ 1 for only 9/170,306 samples, but is supported anyway |
| Belt spline | property `mSplineData` = array of `SplinePointData{Location, ArriveTangent, LeaveTangent}` | **actor-local** cm; first point is at the origin; 2–7 points per belt (2 is most common). The first point's `ArriveTangent` is a unit vector (unused by the curve) |
| Lift | property `mTopTransform{Rotation, Translation}` | relative to the actor; height = `Translation.z`, can be **negative** (downward lifts); seen 150–1200 cm |
| Sign size | **not stored** | size is implied by the class: 10 vanilla classes `Build_StandaloneWidgetSign_{Small, SmallWide, SmallVeryWide, Medium, Large, Huge, Portrait, Square, Square_Small, Square_Tiny}_C`. Layout name in `mSoftActivePrefabLayout` |
| Sign content | `mPrefabTextElementSaveData`, `mPrefabIconElementSaveData`, `mForegroundColor`, `mBackgroundColor`, `mAuxilaryColor`, `mEmissive` | text could be embossed later; not used now |
| Paint (actors) | `mCustomizationData{SwatchDesc, PatternDesc, MaterialDesc, SkinDesc, ColorSlot, OverrideColorData…}` | samples only had `SwatchDesc` populated |
| Paint (lightweight) | `usedSwatchSlot`, `usedMaterial`, `usedPattern`, `usedSkin`, `primaryColor`, `secondaryColor`, `usedPaintFinish`, `patternRotation` | |
| Swatch → actual colour | `BP_BuildableSubsystem.mColorSlots_Data` (array of `FactoryCustomizationColorSlot`) | lives in the `.sav`; probably **not** in a megaprint, so the optional AMS-colour phase likely needs the `.sav` |
| Beam length | lightweight `instanceSpecificData` (`BuildableBeamLightweightData`) | |
| Sign pole height | `mHeight` | |
| Blueprint origin | lightweight `blueprintProxy`, actor `mBlueprintProxy` | useful for selecting the cube if it was placed as a blueprint |

## 5. Geometry conventions measured from public saves (not your file)

The mapping config needs pivots and axes, which no file stores. Measured from how pieces snap together
in the 5 sample saves (`python research/conventions.py`):

| Convention | Evidence (sample counts) | Status |
|---|---|---|
| Foundations 8×N: footprint 800×800 cm, heights 100/200/400 | neighbour offset 800 (211k pairs); stacking 100/200/400 | **measured** |
| Foundation origin = **centre** of its bounding box | a wall on a 1/2/4 m foundation sits +50/+100/+200 cm (= half-height) above the foundation origin (2942/12/217 walls) | **measured** |
| Wall origin = **bottom centre**, centred on the foundation edge line | same data; horizontal offset exactly 400 | **measured** |
| Wall width along local **Y**, thickness along local X | 3171/3171 walls on foundation edges | **measured** |
| Wall heights 400 (8×4) / 100 (8×1) | stacked walls at +400 / +100 | **measured** |
| Ramp origin = centre of its bounding box; footprint 800×800 | same-height foundation neighbours at dz = 0 | **measured** |
| Ramp **high end at local −X** | foundation flush with ramp top lies at −X: 829/963 (8×4), 559/592 (8×2) | **measured** |
| Belt spline height above surface | 100 cm (59.5k points) or 300 cm (12.3k): **belts on poles float above the floor** | measured, but your cube may differ |
| Sign front normal = local +Y (pointing away from the wall), up = local +Z | 56/70 and 6/6 wall-mounted signs | **measured** |
| Sign origin is horizontally centred | two 4 m signs side by side at y = ±200 on one 8 m wall | **measured** (for SmallVeryWide) |
| Sign origin is 50 cm from the wall's centre plane | 62 signs | measured; can't separate wall thickness from sign offset |
| Wall thickness, belt cross-section, sign panel sizes/thickness, sign vertical origin, lift footprint, double/inverted/corner ramp shapes, quarter-pipe profiles | none in save data | **unverified**: defaults in `config/classes.yaml` are marked `verified: false` |

Practical effect: at the likely print scale (about 0.003), the unverified *offsets* are sub-0.2 mm.
The unverified *sizes* (e.g. sign panels) are visible, so they need checking before final prints.

## 6. Object count by class

Can't be produced without your file. The tooling is ready: `megacube coverage <build.json>` prints
the per-class count and mapping coverage.

## 7. Open questions for you

1. **Files:** which game version saved the `.sav`? Any mods? (Real saves often contain modded classes.
   The mapping logs and placeholders them, but they need rules.)
2. **Megaprint:** do you want to use it at all, given the `.sav` contains the cube? Did you paste
   the cube *into* the save from the megaprint via SCIM?
3. **Selection:** roughly where is the cube in the world (centre + size), so I can select it by bounding box?
4. **Belts:** on poles (floating about 1 m above the face in the data), or directly on the surface?
   Default: drop them onto the face (`attach: snap`).
5. **Network:** if you want me to read SCIM's own pages or the wiki (e.g. to confirm sign sizes),
   add `satisfactory-calculator.com` and `satisfactory.wiki.gg` to this environment's allowed
   domains. That's under Network access in the environment's settings (cloud environment menu in the
   session title bar → Edit). See <https://code.claude.com/docs/en/claude-code-on-the-web>.

## Sources

- npm registry: `@etothepii/satisfactory-file-parser` (versions, license, deps); `satisfactory-file-parser` → 404
- <https://github.com/etothepii4/satisfactory-file-parser> (README, GUIDE, CHANGELOG, `src/test/*.sav` used as samples)
- <https://github.com/GreyHak/sat_sav_parse> (`sav_parse.py`: header, chunking, lightweight subsystem, SCIM detection)
- <https://pypi.org/project/satisfactory-save-reader/>
- Search snippets of SCIM community posts mentioning `megaprint_calculator.cbp` (pages themselves blocked):
  <https://steamcommunity.com/app/526870/discussions/0/4511002848506268148/>,
  <https://satisfactory-calculator.com/en/megaprints/index/details/id/1397/name/The+Ultra-HUB>
