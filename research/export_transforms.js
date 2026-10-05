// Phase 0 research helper: dump building transforms (and belt/lift geometry data) from a
// Satisfactory .sav into compact JSON for the Python analysis in conventions.py.
//
// Used only on PUBLIC sample saves from the parser's own test folder, never on the user's file.
//
//   npm install @etothepii/satisfactory-file-parser@4.1.2
//   node export_transforms.js <save.sav> <out.json>
//
// Output: {"buildables": [[class, tx, ty, tz, qx, qy, qz, qw, sx, sy, sz, storage], ...],
//          "conveyors": [{c, t, pts?, top?}, ...]}
const fs = require('fs');
const { Parser } = require('@etothepii/satisfactory-file-parser');

const [, , inPath, outPath] = process.argv;
const buf = fs.readFileSync(inPath);
const save = Parser.ParseSave('research', buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), { throwErrors: false });
const objects = Object.values(save.levels).flatMap(level => level.objects);

const className = typePath => typePath.split('.').pop();
const buildables = [];
const conveyors = [];
const push = (cls, t, storage) => buildables.push([
  cls, t.translation.x, t.translation.y, t.translation.z,
  t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w,
  t.scale3d.x, t.scale3d.y, t.scale3d.z, storage]);
const vec = p => [p.x, p.y, p.z];

for (const o of objects) {
  if (o.typePath === '/Script/FactoryGame.FGLightweightBuildableSubsystem') {
    // 1.0+: foundations, walls, ramps, beams ... are stored here, not as actors.
    for (const b of o.specialProperties.buildables) {
      const cls = className(b.typeReference ? b.typeReference.pathName : b.typePath);
      for (const inst of b.instances) push(cls, inst.transform, 'lightweight');
    }
    continue;
  }
  if (!o.transform || !/^\/Game\/FactoryGame\/Buildable\//.test(o.typePath)) continue;
  const cls = className(o.typePath);
  push(cls, o.transform, 'actor');
  if (/^Build_ConveyorBeltMk\d+_C$/.test(cls) && o.properties.mSplineData) {
    const pts = o.properties.mSplineData.values.map(v => [
      vec(v.properties.Location.value), vec(v.properties.ArriveTangent.value), vec(v.properties.LeaveTangent.value)]);
    conveyors.push({ c: cls, t: o.transform, pts });
  } else if (/^Build_ConveyorLiftMk\d+_C$/.test(cls) && o.properties.mTopTransform) {
    conveyors.push({ c: cls, t: o.transform, top: vec(o.properties.mTopTransform.value.properties.Translation.value) });
  }
}

fs.writeFileSync(outPath, JSON.stringify({ save: inPath.split('/').pop(), saveVersion: save.header.saveVersion, buildables, conveyors }));
console.log(`${inPath.split('/').pop()}: saveVersion ${save.header.saveVersion}, ${buildables.length} buildables, ${conveyors.length} belts/lifts`);
