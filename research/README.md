# Phase 0 research scripts

These scripts produced the measurements in [`docs/phase0_research.md`](../docs/phase0_research.md).
They run on the **public** sample saves in the parser's repository, not on your files.

```bash
# 1. parser + public sample saves
npm install @etothepii/satisfactory-file-parser@4.1.2
git clone --depth 1 https://github.com/etothepii4/satisfactory-file-parser.git sfp
#    (samples used: Release-032, Dunarr-019, Dunarr-076, 264_ohne_Mods, 269 in sfp/src/test/)

# 2. dump transforms, belt splines and lift heights
node research/export_transforms.js sfp/src/test/Dunarr-019.sav dumps/Dunarr-019.json

# 3. measure snapping/pivot/axis conventions
python research/conventions.py dumps/*.json
```

`export_transforms.js` is research tooling, not the Phase 1 extractor. Phase 1 waits for your
go-ahead after the Phase 0 report. It will reuse the same parser calls.
