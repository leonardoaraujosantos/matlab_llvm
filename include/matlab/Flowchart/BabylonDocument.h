#pragma once

//===----------------------------------------------------------------------===//
// Shared Babylon.js document writer.
//
// Emits the self-contained HTML player given a pre-built scene JSON string. The
// HTML shell, the engine <script> tags, and the inline viewer logic are
// identical for every producer of a Babylon scene:
//   - the block-diagram path  (lib/Flowchart/EmitBabylon.cpp, -emit-mflowlink-babylon)
//   - the command-line sim3d path (runtime/toolbox/sim3d/runtime_sim3d.cpp)
// so they share this one writer to stay byte-for-byte in sync (the
// sim3d-matlab-api OpenSpec change, design D3). Header-only so the runtime can
// include it without linking the compiler libraries.
//
// The scene JSON contract (built by each producer) is:
//   { "world":{...}, "times":[...], "lights":[...], "cameras":[...],
//     "actors":[ {"id","name","shape","size","radius","height","color",
//                 "emissive","opacity",..., "keys":[[tx,ty,tz,rx,ry,rz,sx,sy,sz,...]]} ] }
// The viewer maps the right-handed Z-up metres frame onto Babylon by parenting
// the scene under a root rotated -90 deg about X. The rotation triple
// [rx,ry,rz] is standard intrinsic roll-pitch-yaw about model [X,Y,Z]
// (a cart-pole whose cart moves along X tilts with [0,theta,0]); the same
// applies to URDF link/joint origin rpy.
//===----------------------------------------------------------------------===//

#include <ostream>
#include <string>

namespace matlab {
namespace babylon {

// Parameters for the document shell (everything outside the scene JSON).
struct DocParams {
  std::string Title;                              // shown in <title> + header div
  std::string CdnBase = "https://cdn.babylonjs.com";
  // When non-empty, the Babylon engine bundle CONTENTS are inlined (a fully
  // network-free artifact) instead of the CDN <script>. The caller reads the
  // file; this writer only embeds the bytes.
  std::string InlineEngine;
  bool WorldPhysics = false;                      // emit a physics engine script
  std::string Engine = "havok";                   // "havok" | "ammo"
};

// Write the full HTML document to OS. `SceneJson` is the complete scene object
// (no trailing newline required). Never fails (the inline-engine bytes are
// supplied by the caller), so there is no error path here.
inline void writeDocument(std::ostream &OS, const std::string &SceneJson,
                          const DocParams &P) {
  OS << "<!doctype html>\n<html lang=\"en\">\n<head>\n";
  OS << "<meta charset=\"utf-8\">\n";
  OS << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n";
  OS << "<title>mflowLink 3-D — " << P.Title << "</title>\n";
  OS << "<style>\n"
        "html,body{margin:0;height:100%;background:#111;color:#ddd;"
        "font-family:system-ui,sans-serif;overflow:hidden}\n"
        "#c{width:100%;height:100%;display:block;touch-action:none}\n"
        "#hud{position:fixed;left:12px;right:12px;bottom:12px;display:flex;"
        "gap:10px;align-items:center;background:rgba(0,0,0,.55);padding:8px 12px;"
        "border-radius:8px}\n"
        "#hud button{background:#2b6cb0;color:#fff;border:0;border-radius:5px;"
        "padding:6px 12px;cursor:pointer}\n"
        "#scrub{flex:1}\n#t{min-width:84px;font-variant-numeric:tabular-nums}\n"
        "#title{position:fixed;left:12px;top:10px;font-size:13px;opacity:.8}\n"
        "</style>\n</head>\n<body>\n";
  OS << "<canvas id=\"c\"></canvas>\n";
  OS << "<div id=\"title\">" << P.Title << " — mflowLink 3-D animation</div>\n";
  OS << "<div id=\"hud\"><button id=\"play\">Pause</button>"
        "<input id=\"scrub\" type=\"range\" min=\"0\" max=\"100\" value=\"0\">"
        "<span id=\"t\">t = 0.000</span></div>\n";
  OS << "<script id=\"scene\" type=\"application/json\">\n";
  OS << SceneJson << "\n</script>\n";
  // Engine: inline a user-provided bundle for a fully network-free artifact,
  // else reference the pinned CDN.
  if (!P.InlineEngine.empty()) {
    OS << "<script>\n" << P.InlineEngine << "\n</script>\n";
  } else {
    OS << "<script src=\"" << P.CdnBase << "/babylon.js\"></script>\n";
    OS << "<script src=\"" << P.CdnBase
       << "/loaders/babylonjs.loaders.min.js\"></script>\n";
  }
  // Viewer-side physics engine (Tier 4) — only when the world enables physics.
  if (P.WorldPhysics) {
    if (P.Engine == "ammo")
      OS << "<script src=\"https://cdn.jsdelivr.net/npm/ammo.js@0.0.10/builds/"
            "ammo.js\"></script>\n";
    else
      OS << "<script src=\"" << P.CdnBase
         << "/havok/HavokPhysics_umd.js\"></script>\n";
  }
  OS << "<script>\n";
  OS << R"JS(
const DATA = JSON.parse(document.getElementById('scene').textContent);
const canvas = document.getElementById('c');
const engine = new BABYLON.Engine(canvas, true, {preserveDrawingBuffer:true, stencil:true});
const scene = new BABYLON.Scene(engine);
scene.useRightHandedSystem = true;
const bg = DATA.world.background || [0.07,0.08,0.1];
scene.clearColor = new BABYLON.Color4(bg[0], bg[1], bg[2], 1);

// Model frame: right-handed, Z-up, metres. Babylon RH is Y-up, so parent the
// whole scene under a root rotated -90 deg about X to lift model +Z to world up.
const root = new BABYLON.TransformNode('zup', scene);
root.rotation = new BABYLON.Vector3(-Math.PI/2, 0, 0);

const vp = DATA.world.viewpoint || [8,8,6];
const camera = new BABYLON.ArcRotateCamera('cam', -Math.PI/3, Math.PI/3,
  Math.hypot(vp[0],vp[1],vp[2]) || 14, BABYLON.Vector3.Zero(), scene);
camera.wheelDeltaPercentage = 0.01;
camera.attachControl(canvas, true);

// Lights: a dim hemispheric fill always, plus each configured signal_light3d.
const fill = new BABYLON.HemisphericLight('fill', new BABYLON.Vector3(0,1,0), scene);
fill.intensity = (DATA.lights && DATA.lights.length) ? 0.35 : 0.85;
function v3(p){ return new BABYLON.Vector3(p[0], p[1], p[2]); }
for (const L of (DATA.lights || [])) {
  let light;
  if (L.type === 'point')      light = new BABYLON.PointLight(L.id, v3(L.position), scene);
  else if (L.type === 'spot')  light = new BABYLON.SpotLight(L.id, v3(L.position), v3(L.direction), Math.PI/3, 2, scene);
  else                         light = new BABYLON.DirectionalLight(L.id, v3(L.direction), scene);
  light.intensity = L.intensity;
  light.diffuse = new BABYLON.Color3(L.color[0], L.color[1], L.color[2]);
  light.parent = root;
}

const PHYS = !!DATA.world.physics; // viewer-side Havok/Ammo (visualization only)
let groundMesh = null;
if (DATA.world.showGround) {
  groundMesh = BABYLON.MeshBuilder.CreateGround('ground', {width:40, height:40}, scene);
  const gm = new BABYLON.StandardMaterial('gm', scene);
  gm.diffuseColor = new BABYLON.Color3(0.18,0.19,0.22); groundMesh.material = gm;
  // Without physics, parent to the Z-up root and lay it in the model XY plane.
  // With physics, keep Babylon's native Y-up ground so rigid bodies rest on it.
  if (!PHYS) { groundMesh.parent = root; groundMesh.rotation.x = Math.PI/2; }
}
if (DATA.world.showAxes) new BABYLON.AxesViewer(scene, 2);
// Map a model (Z-up) vector to Babylon (Y-up) world coords.
function toBabylon(p){ return new BABYLON.Vector3(p[0], p[2], p[1]); }

function mkColor(c){ return new BABYLON.Color3(c[0], c[1], c[2]); }
function buildMesh(a){
  const s = a.size || [1,1,1];
  switch (a.shape) {
    case 'sphere':   return BABYLON.MeshBuilder.CreateSphere(a.id,{diameter:2*a.radius},scene);
    case 'cylinder': return BABYLON.MeshBuilder.CreateCylinder(a.id,{diameter:2*a.radius,height:a.height},scene);
    case 'cone':     return BABYLON.MeshBuilder.CreateCylinder(a.id,{diameterTop:0,diameterBottom:2*a.radius,height:a.height},scene);
    case 'capsule':  return BABYLON.MeshBuilder.CreateCapsule(a.id,{radius:a.radius,height:a.height},scene);
    case 'plane':    return BABYLON.MeshBuilder.CreateGround(a.id,{width:s[0],height:s[1]},scene);
    case 'box':
    default:         return BABYLON.MeshBuilder.CreateBox(a.id,{width:s[0],height:s[1],depth:s[2]},scene);
  }
}

// URDF rigging (Tier 3b): build one node per link, attach via joints, and
// return the movable joints so applyFrame can rotate them from jointAngles.
function buildUrdf(a, baseNode){
  const nodes = {};
  for (const L of a.urdf.links) {
    const n = new BABYLON.TransformNode(a.id+'_'+L.name, scene);
    let vis;
    if (L.geom === 'cylinder') vis = BABYLON.MeshBuilder.CreateCylinder(n.name+'_v',{diameter:2*L.radius,height:L.length},scene);
    else if (L.geom === 'sphere') vis = BABYLON.MeshBuilder.CreateSphere(n.name+'_v',{diameter:2*L.radius},scene);
    else { const b=L.box||[0.1,0.1,0.1]; vis = BABYLON.MeshBuilder.CreateBox(n.name+'_v',{width:b[0],height:b[1],depth:b[2]},scene); }
    vis.parent = n;
    const vo = L.vorigin||[0,0,0,0,0,0];
    vis.position.set(vo[0],vo[1],vo[2]);
    vis.rotationQuaternion = BABYLON.Quaternion.RotationYawPitchRoll(vo[4],vo[3],vo[5]);
    const mat = new BABYLON.StandardMaterial(n.name+'_m', scene);
    mat.diffuseColor = mkColor(a.color || [0.7,0.7,0.78]); vis.material = mat;
    nodes[L.name] = n;
  }
  const roots = new Set(Object.keys(nodes));
  const joints = [];
  for (const J of a.urdf.joints) {
    const cn = nodes[J.child], pn = nodes[J.parent];
    if (!cn || !pn) continue;
    cn.parent = pn;
    cn.position.set(J.origin[0],J.origin[1],J.origin[2]);
    const baseQ = BABYLON.Quaternion.RotationYawPitchRoll(J.origin[4],J.origin[3],J.origin[5]);
    cn.rotationQuaternion = baseQ.clone();
    roots.delete(J.child);
    if (J.q >= 0) joints.push({node:cn, axis:new BABYLON.Vector3(J.axis[0],J.axis[1],J.axis[2]), q:J.q, base:baseQ});
  }
  for (const r of roots) nodes[r].parent = baseNode;
  return joints;
}
const urdfRigs = {}; // actor id -> [movable joints]

const meshByKey = {}; // both id and name resolve to the animated node
for (const a of DATA.actors) {
  let m;
  if (a.text !== undefined) {
    // Billboarded text label (Tier 6.3) via a DynamicTexture on a plane.
    m = BABYLON.MeshBuilder.CreatePlane(a.id, {width:3, height:0.9}, scene);
    m.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
    const dt = new BABYLON.DynamicTexture(a.id+'_dt', {width:512, height:150}, scene, true);
    dt.hasAlpha = true;
    dt.drawText(a.text, null, 100, 'bold 80px sans-serif', '#ffffff', 'transparent', true);
    const tm = new BABYLON.StandardMaterial(a.id+'_tm', scene);
    tm.diffuseTexture = dt; tm.opacityTexture = dt; tm.emissiveColor = mkColor(a.color || [1,1,1]);
    tm.backFaceCulling = false; m.material = tm;
  } else if (a.urdf) {
    m = new BABYLON.TransformNode(a.id, scene);
    urdfRigs[a.id] = buildUrdf(a, m);
  } else if (a.mesh) {
    // glTF/GLB import (Tier 3): animate a TransformNode; the loaded geometry
    // is parented to it once it streams in from the inline data URL.
    m = new BABYLON.TransformNode(a.id, scene);
    BABYLON.SceneLoader.ImportMesh('', '', a.mesh, scene, (meshes) => {
      for (const mm of meshes) if (!mm.parent) mm.parent = m;
    }, null, null, a.meshExt || '.glb');
  } else {
    m = buildMesh(a);
    const mat = new BABYLON.StandardMaterial(a.id+'_m', scene);
    mat.diffuseColor = mkColor(a.color || [0.6,0.6,0.6]);
    if (a.emissive) mat.emissiveColor = mkColor(a.emissive);
    if (a.opacity !== undefined && a.opacity < 1) mat.alpha = a.opacity;
    m.material = mat;
  }
  m.rotationQuaternion = BABYLON.Quaternion.Identity();
  meshByKey[a.id] = m; if (a.name) meshByKey[a.name] = m;
}

// Tier 4 — viewer physics. Physics actors live in Babylon world frame (not the
// Z-up root) so the engine integrates them directly; they are excluded from the
// recorded-timeline animation (physics is visualization-only, never a golden).
const physicsActors = [];
const physicsSet = new Set();
if (PHYS) {
  for (const a of DATA.actors) {
    if (!a.physics) continue;
    const m = meshByKey[a.id];
    m.parent = null;
    const k0 = (a.keys && a.keys[0]) || [0,0,0,0,0,0,1,1,1];
    m.position = toBabylon([k0[0], k0[1], k0[2]]);
    m.scaling.set(k0[6], k0[7], k0[8]);
    physicsActors.push(a); physicsSet.add(a.id);
  }
}
// Resolve parenting by parent NAME (or id); children compose via the scene
// graph, so a child's recorded transform is its local frame.
for (const a of DATA.actors) {
  const m = meshByKey[a.id];
  m.parent = (a.parent && meshByKey[a.parent]) ? meshByKey[a.parent] : root;
}

// Camera: first configured signal_camera3d wins. follow → track an actor.
const camCfg = (DATA.cameras && DATA.cameras[0]) || null;
let followMesh = null;
if (camCfg) {
  camera.fov = camCfg.fov || 0.8;
  const tgt = camCfg.target || [0,0,0];
  if (camCfg.mode === 'follow' && camCfg.follow && meshByKey[camCfg.follow]) {
    followMesh = meshByKey[camCfg.follow];
  } else {
    const p = camCfg.position || [8,8,6];
    camera.setPosition(new BABYLON.Vector3(p[0], p[2], p[1])); // model Z-up
    camera.setTarget(new BABYLON.Vector3(tgt[0], tgt[2], tgt[1]));
  }
}

const N = DATA.times.length;
function applyFrame(i){
  i = Math.max(0, Math.min(N-1, i|0));
  for (const a of DATA.actors) {
    if (physicsSet.has(a.id)) continue; // physics-driven in the viewer
    const k = a.keys[i]; if (!k) continue;
    const m = meshByKey[a.id];
    m.position.set(k[0], k[1], k[2]);
    m.rotationQuaternion = BABYLON.Quaternion.RotationYawPitchRoll(k[4], k[3], k[5]);
    m.scaling.set(k[6], k[7], k[8]);
    // URDF joints: rotate each movable joint about its axis by jointAngles[q].
    const rig = urdfRigs[a.id];
    if (rig) for (const J of rig) {
      const ang = k[9 + J.q] || 0;
      J.node.rotationQuaternion = J.base.multiply(BABYLON.Quaternion.RotationAxis(J.axis, ang));
    }
  }
  if (followMesh) camera.setTarget(followMesh.getAbsolutePosition());
  const tEl = document.getElementById('t');
  tEl.textContent = 't = ' + (DATA.times[i]||0).toFixed(3);
}

// Playback: real-time against the recorded sample times, looping.
let playing = N > 1, frame = 0, acc = 0;
const scrub = document.getElementById('scrub');
scrub.max = String(Math.max(0, N-1));
const playBtn = document.getElementById('play');
playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; };
scrub.oninput = () => { playing = false; playBtn.textContent='Play'; frame = +scrub.value; applyFrame(frame); };

// Tier 4 — initialize the viewer physics engine (async WASM) and seed bodies.
// Visualization-only: results never re-enter the model (design D3).
async function initPhysics(){
  if (!PHYS) return;
  let plugin = null;
  try {
    if (DATA.world.engine === 'ammo' && typeof Ammo !== 'undefined') {
      const ammo = await Ammo(); plugin = new BABYLON.AmmoJSPlugin(true, ammo);
    } else if (typeof HavokPhysics !== 'undefined') {
      const hk = await HavokPhysics(); plugin = new BABYLON.HavokPlugin(true, hk);
    }
  } catch (e) { console.warn('physics init failed:', e); }
  if (!plugin) return;
  const g = DATA.world.gravity || [0,0,-9.81];
  scene.enablePhysics(toBabylon(g), plugin);
  const ST = BABYLON.PhysicsShapeType;
  for (const a of physicsActors) {
    const m = meshByKey[a.id];
    const t = (a.collisionShape === 'sphere') ? ST.SPHERE
            : (a.collisionShape === 'mesh' || a.collisionShape === 'convexHull') ? ST.CONVEX_HULL
            : ST.BOX;
    new BABYLON.PhysicsAggregate(m, t,
      {mass: a.mass||0, restitution: a.restitution||0, friction: (a.friction!==undefined?a.friction:0.5)}, scene);
  }
  if (groundMesh) new BABYLON.PhysicsAggregate(groundMesh, ST.BOX, {mass:0, friction:0.8, restitution:0.1}, scene);
}
initPhysics();

applyFrame(0);
let last = performance.now();
engine.runRenderLoop(() => {
  const now = performance.now(), dt = (now-last)/1000; last = now;
  if (playing && N > 1) {
    acc += dt * (DATA.world.pacingRate || 1);
    const span = (DATA.times[N-1]-DATA.times[0]) || (N*0.02);
    const tNow = DATA.times[0] + ((acc % span));
    let i = 0; while (i < N-1 && DATA.times[i+1] <= tNow) i++;
    frame = i; scrub.value = String(frame); applyFrame(frame);
  }
  scene.render();
});
window.addEventListener('resize', () => engine.resize());
)JS";
  OS << "\n</script>\n</body>\n</html>\n";
}

} // namespace babylon
} // namespace matlab
