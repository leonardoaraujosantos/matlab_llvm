//===----------------------------------------------------------------------===//
// mflow-3d-animation — Babylon.js scene + timeline emitter.
// See include/matlab/Flowchart/EmitBabylon.h.
//===----------------------------------------------------------------------===//

#include "matlab/Flowchart/EmitBabylon.h"

#include "matlab/Flowchart/MflowLinkModel.h"
#include "matlab/Flowchart/MflowLinkSim.h"

#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace matlab {
namespace flowchart {

namespace {

const std::string *param(const MflBlock &B, const char *Key) {
  auto It = B.Params.find(Key);
  return It == B.Params.end() ? nullptr : &It->second;
}

std::string paramOr(const MflBlock &B, const char *Key, const std::string &Def) {
  const std::string *S = param(B, Key);
  return S ? *S : Def;
}

bool paramBool(const MflBlock &B, const char *Key, bool Def) {
  const std::string *S = param(B, Key);
  if (!S) return Def;
  return *S == "true" || *S == "1" || *S == "yes";
}

double paramNum(const MflBlock &B, const char *Key, double Def) {
  const std::string *S = param(B, Key);
  if (!S) return Def;
  try { return std::stod(*S); } catch (...) { return Def; }
}

// Parse a "x,y,z" / "x y z" param into a JSON array literal "[x,y,z]"; returns
// the given fallback (already a JSON array string) when the param is absent.
std::string vecJson(const MflBlock &B, const char *Key, const char *Fallback) {
  const std::string *S = param(B, Key);
  if (!S) return Fallback;
  std::vector<double> V;
  std::string Tok, Str = *S;
  Str.push_back(',');
  for (char C : Str) {
    if (C == ',' || C == ' ' || C == '\t' || C == ';' || C == '[' || C == ']') {
      if (!Tok.empty()) {
        try { V.push_back(std::stod(Tok)); } catch (...) {}
        Tok.clear();
      }
    } else {
      Tok.push_back(C);
    }
  }
  std::ostringstream OS;
  OS << "[";
  for (size_t I = 0; I < V.size(); ++I) OS << (I ? "," : "") << V[I];
  OS << "]";
  return OS.str();
}

std::string jsonStr(const std::string &S) {
  std::string Out = "\"";
  for (char C : S) {
    switch (C) {
    case '"': Out += "\\\""; break;
    case '\\': Out += "\\\\"; break;
    case '\n': Out += "\\n"; break;
    case '\r': Out += "\\r"; break;
    case '\t': Out += "\\t"; break;
    default: Out += C;
    }
  }
  Out += "\"";
  return Out;
}

} // namespace

bool emitMflowLinkBabylon(const MflowLinkModel &M, const MflowLinkSim &Sim,
                          std::ostream &OS, std::string &Err,
                          const BabylonEmitOptions &Opts) {
  // Locate the scene config.
  const MflBlock *World = nullptr;
  for (const auto &B : M.Blocks)
    if (B.Kind == "signal_world3d") { World = &B; break; }
  if (!World) {
    Err = "-emit-mflowlink-babylon: model has no signal_world3d scene block";
    return false;
  }

  // Index the recorded log columns by name → column index.
  const auto &Names = Sim.logColumnNames();
  const auto &Cols = Sim.logColumns();
  std::map<std::string, size_t> ColIdx;
  for (size_t I = 0; I < Names.size(); ++I) ColIdx[Names[I]] = I;

  // The shared time vector is taken from the first recorded column (every
  // column shares the same sample times).
  std::vector<double> Times;
  if (!Cols.empty())
    for (const auto &P : Cols.front()) Times.push_back(P.T);

  // Build the actor list + per-actor keyframe arrays.
  static const char *Comp[9] = {"tx", "ty", "tz", "rx", "ry",
                                 "rz", "sx", "sy", "sz"};
  std::ostringstream Actors;
  size_t ActorCount = 0;
  for (const auto &B : M.Blocks) {
    if (B.Kind != "signal_actor3d") continue;
    if (ActorCount) Actors << ",\n";
    std::string Name = paramOr(B, "name", B.Id);
    Actors << "    {";
    Actors << "\"id\":" << jsonStr(B.Id);
    Actors << ",\"name\":" << jsonStr(Name);
    Actors << ",\"shape\":" << jsonStr(paramOr(B, "shape", "box"));
    Actors << ",\"size\":" << vecJson(B, "size", "[1,1,1]");
    Actors << ",\"radius\":" << paramNum(B, "radius", 0.5);
    Actors << ",\"height\":" << paramNum(B, "height", 1.0);
    Actors << ",\"color\":" << vecJson(B, "color", "[0.6,0.6,0.6]");
    Actors << ",\"emissive\":" << vecJson(B, "emissive", "[0,0,0]");
    Actors << ",\"opacity\":" << paramNum(B, "opacity", 1.0);
    if (const std::string *P = param(B, "parent"))
      Actors << ",\"parent\":" << jsonStr(*P);
    // Tier-4 (viewer physics) hints — emitted now so the scene contract is
    // stable; honoured by the viewer once physics lands.
    if (paramBool(B, "physics", false)) {
      Actors << ",\"physics\":true";
      Actors << ",\"mass\":" << paramNum(B, "mass", 1.0);
      Actors << ",\"friction\":" << paramNum(B, "friction", 0.5);
      Actors << ",\"restitution\":" << paramNum(B, "restitution", 0.2);
      Actors << ",\"collisionShape\":"
             << jsonStr(paramOr(B, "collisionShape", "box"));
    }
    // Keyframes: nine components per recorded time. Falls back to the static
    // identity/param transform when the actor produced no log columns.
    Actors << ",\"keys\":[";
    size_t Rows = Times.size();
    for (size_t R = 0; R < Rows; ++R) {
      Actors << (R ? "," : "") << "[";
      for (int C = 0; C < 9; ++C) {
        auto It = ColIdx.find(B.Id + "[" + Comp[C] + "]");
        double V = (C >= 6) ? 1.0 : 0.0; // scale defaults to 1
        if (It != ColIdx.end() && R < Cols[It->second].size())
          V = Cols[It->second][R].Value;
        Actors << (C ? "," : "") << V;
      }
      Actors << "]";
    }
    Actors << "]}";
    ++ActorCount;
  }

  // Lights (Tier 2) — static config blocks read by the viewer.
  std::ostringstream Lights;
  size_t LightCount = 0;
  for (const auto &B : M.Blocks) {
    if (B.Kind != "signal_light3d") continue;
    if (LightCount) Lights << ",";
    Lights << "{\"id\":" << jsonStr(B.Id);
    Lights << ",\"type\":" << jsonStr(paramOr(B, "type", "directional"));
    Lights << ",\"color\":" << vecJson(B, "color", "[1,1,1]");
    Lights << ",\"intensity\":" << paramNum(B, "intensity", 0.8);
    Lights << ",\"position\":" << vecJson(B, "position", "[0,0,10]");
    Lights << ",\"direction\":" << vecJson(B, "direction", "[-0.5,-0.5,-1]");
    Lights << "}";
    ++LightCount;
  }

  // Cameras (Tier 2) — static viewpoint or follow-actor.
  std::ostringstream Cameras;
  size_t CameraCount = 0;
  for (const auto &B : M.Blocks) {
    if (B.Kind != "signal_camera3d") continue;
    if (CameraCount) Cameras << ",";
    Cameras << "{\"id\":" << jsonStr(B.Id);
    Cameras << ",\"mode\":" << jsonStr(paramOr(B, "mode", "static"));
    Cameras << ",\"position\":" << vecJson(B, "position", "[8,8,6]");
    Cameras << ",\"target\":" << vecJson(B, "target", "[0,0,0]");
    if (const std::string *F = param(B, "follow"))
      Cameras << ",\"follow\":" << jsonStr(*F);
    Cameras << ",\"fov\":" << paramNum(B, "fov", 0.8);
    Cameras << "}";
    ++CameraCount;
  }

  // The scene JSON.
  std::ostringstream Scene;
  Scene << "{\n";
  Scene << "  \"world\":{";
  Scene << "\"gravity\":" << vecJson(*World, "gravity", "[0,0,-9.81]");
  Scene << ",\"viewpoint\":" << vecJson(*World, "viewpoint", "[8,8,6]");
  Scene << ",\"engine\":" << jsonStr(paramOr(*World, "engine", "havok"));
  Scene << ",\"physics\":" << (paramBool(*World, "physics", false) ? "true"
                                                                    : "false");
  Scene << ",\"showGround\":"
        << (paramBool(*World, "showGround", true) ? "true" : "false");
  Scene << ",\"showAxes\":"
        << (paramBool(*World, "showAxes", true) ? "true" : "false");
  Scene << ",\"background\":" << vecJson(*World, "background", "[0.07,0.08,0.1]");
  Scene << "},\n";
  Scene << "  \"times\":[";
  for (size_t I = 0; I < Times.size(); ++I)
    Scene << (I ? "," : "") << Times[I];
  Scene << "],\n";
  Scene << "  \"lights\":[" << Lights.str() << "],\n";
  Scene << "  \"cameras\":[" << Cameras.str() << "],\n";
  Scene << "  \"actors\":[\n" << Actors.str() << "\n  ]\n";
  Scene << "}";

  const std::string &Cdn = Opts.CdnBase;

  // The document. Scene JSON + all viewer logic are inline; only the engine is
  // CDN-referenced. The viewer maps the model's right-handed Z-up metres frame
  // onto Babylon by parenting everything under a root rotated -90° about X.
  OS << "<!doctype html>\n<html lang=\"en\">\n<head>\n";
  OS << "<meta charset=\"utf-8\">\n";
  OS << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n";
  OS << "<title>mflowLink 3-D — " << M.EntryName << "</title>\n";
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
  OS << "<div id=\"title\">" << M.EntryName << " — mflowLink 3-D animation</div>\n";
  OS << "<div id=\"hud\"><button id=\"play\">Pause</button>"
        "<input id=\"scrub\" type=\"range\" min=\"0\" max=\"100\" value=\"0\">"
        "<span id=\"t\">t = 0.000</span></div>\n";
  OS << "<script id=\"scene\" type=\"application/json\">\n";
  OS << Scene.str() << "\n</script>\n";
  OS << "<script src=\"" << Cdn << "/babylon.js\"></script>\n";
  OS << "<script src=\"" << Cdn << "/loaders/babylonjs.loaders.min.js\"></script>\n";
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

if (DATA.world.showGround) {
  const g = BABYLON.MeshBuilder.CreateGround('ground', {width:40, height:40}, scene);
  g.parent = root; g.rotation.x = Math.PI/2; // ground lies in model XY plane
  const gm = new BABYLON.StandardMaterial('gm', scene);
  gm.diffuseColor = new BABYLON.Color3(0.18,0.19,0.22); g.material = gm;
}
if (DATA.world.showAxes) new BABYLON.AxesViewer(scene, 2);

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

const meshByKey = {}; // both id and name resolve to the mesh
for (const a of DATA.actors) {
  const m = buildMesh(a);
  const mat = new BABYLON.StandardMaterial(a.id+'_m', scene);
  mat.diffuseColor = mkColor(a.color || [0.6,0.6,0.6]);
  if (a.emissive) mat.emissiveColor = mkColor(a.emissive);
  if (a.opacity !== undefined && a.opacity < 1) mat.alpha = a.opacity;
  m.material = mat;
  m.rotationQuaternion = BABYLON.Quaternion.Identity();
  meshByKey[a.id] = m; if (a.name) meshByKey[a.name] = m;
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
    const k = a.keys[i]; if (!k) continue;
    const m = meshById[a.id];
    m.position.set(k[0], k[1], k[2]);
    m.rotationQuaternion = BABYLON.Quaternion.RotationYawPitchRoll(k[5], k[4], k[3]);
    m.scaling.set(k[6], k[7], k[8]);
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

applyFrame(0);
let last = performance.now();
engine.runRenderLoop(() => {
  const now = performance.now(), dt = (now-last)/1000; last = now;
  if (playing && N > 1) {
    acc += dt;
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
  (void)Opts;
  return true;
}

} // namespace flowchart
} // namespace matlab
