//===----------------------------------------------------------------------===//
// mflow-3d-animation — Babylon.js scene + timeline emitter.
// See include/matlab/Flowchart/EmitBabylon.h.
//===----------------------------------------------------------------------===//

#include "matlab/Flowchart/EmitBabylon.h"

#include "matlab/Flowchart/MflowLinkModel.h"
#include "matlab/Flowchart/MflowLinkSim.h"

#include <fstream>
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

// Read a binary file fully into `Out`. Returns false if it cannot be opened.
bool readFile(const std::string &Path, std::string &Out) {
  std::ifstream In(Path, std::ios::binary);
  if (!In) return false;
  std::ostringstream SS;
  SS << In.rdbuf();
  Out = SS.str();
  return true;
}

std::string base64(const std::string &In) {
  static const char *T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string Out;
  Out.reserve(((In.size() + 2) / 3) * 4);
  size_t I = 0;
  for (; I + 2 < In.size(); I += 3) {
    unsigned N = (unsigned char)In[I] << 16 | (unsigned char)In[I + 1] << 8 |
                 (unsigned char)In[I + 2];
    Out += T[(N >> 18) & 63]; Out += T[(N >> 12) & 63];
    Out += T[(N >> 6) & 63];  Out += T[N & 63];
  }
  if (I < In.size()) {
    unsigned N = (unsigned char)In[I] << 16;
    if (I + 1 < In.size()) N |= (unsigned char)In[I + 1] << 8;
    Out += T[(N >> 18) & 63];
    Out += T[(N >> 12) & 63];
    Out += (I + 1 < In.size()) ? T[(N >> 6) & 63] : '=';
    Out += '=';
  }
  return Out;
}

// glTF/GLB mesh → an inline `data:` URL. The extension picks the MIME so
// Babylon's loader plugin is selected correctly for the data URL.
std::string meshDataUrl(const std::string &Bytes, const std::string &Ext) {
  std::string Mime = (Ext == ".glb") ? "model/gltf-binary" : "model/gltf+json";
  return "data:" + Mime + ";base64," + base64(Bytes);
}

//===----------------------------------------------------------------------===//
// Minimal URDF parser (Tier 3b). Handles the common subset: <link> with one
// <visual> (<geometry> box/cylinder/sphere + optional <origin>), and <joint>
// (type, <parent>, <child>, <origin>, <axis>). Mesh geometry and multiple
// visuals are out of scope — a link with no parsed primitive renders as a small
// box placeholder. Good enough to articulate a robot from its joint signals via
// the viewer scene graph; FK matches the URDF kinematics by construction.
//===----------------------------------------------------------------------===//

struct UrdfGeom {
  std::string type = "box"; // box | cylinder | sphere
  double box[3] = {0.1, 0.1, 0.1};
  double radius = 0.05, length = 0.1;
  double vorigin[6] = {0, 0, 0, 0, 0, 0}; // xyz + rpy
};
struct UrdfLink { std::string name; UrdfGeom geom; };
struct UrdfJoint {
  std::string name, type, parent, child;
  double origin[6] = {0, 0, 0, 0, 0, 0};
  double axis[3] = {0, 0, 1};
};
struct UrdfModel {
  std::vector<UrdfLink> links;
  std::vector<UrdfJoint> joints;
};

// Extract attribute `Name` from a tag-text region [Beg, End).
std::string xmlAttr(const std::string &S, size_t Beg, size_t End,
                    const std::string &Name) {
  std::string Key = Name + "=";
  size_t P = S.find(Key, Beg);
  if (P == std::string::npos || P >= End) return "";
  P += Key.size();
  if (P >= S.size() || (S[P] != '"' && S[P] != '\'')) return "";
  char Q = S[P++];
  size_t E = S.find(Q, P);
  if (E == std::string::npos || E > End) return "";
  return S.substr(P, E - P);
}

void parseTriple(const std::string &S, double *Out, int N) {
  std::string Tok;
  int I = 0;
  std::string Str = S;
  Str.push_back(' ');
  for (char C : Str) {
    if (C == ' ' || C == '\t' || C == ',') {
      if (!Tok.empty()) {
        if (I < N) { try { Out[I] = std::stod(Tok); } catch (...) {} ++I; }
        Tok.clear();
      }
    } else Tok.push_back(C);
  }
}

// Find the inner extent of the element starting at `<Tag` from `From`; returns
// the open-tag end and sets CloseAt to the matching `</Tag>` (or the self-close
// `/>`), or npos when not found.
size_t findElement(const std::string &S, const std::string &Tag, size_t From,
                   size_t &OpenEnd, size_t &CloseAt) {
  size_t Open = S.find("<" + Tag, From);
  if (Open == std::string::npos) return std::string::npos;
  OpenEnd = S.find('>', Open);
  if (OpenEnd == std::string::npos) return std::string::npos;
  if (OpenEnd > 0 && S[OpenEnd - 1] == '/') { CloseAt = Open; return Open; }
  CloseAt = S.find("</" + Tag, OpenEnd);
  return Open;
}

void parseOrigin(const std::string &S, size_t Beg, size_t End, double *Out6) {
  size_t OE, CA;
  size_t O = findElement(S.substr(0, End), "origin", Beg, OE, CA);
  if (O == std::string::npos || O >= End) return;
  std::string xyz = xmlAttr(S, O, OE, "xyz");
  std::string rpy = xmlAttr(S, O, OE, "rpy");
  if (!xyz.empty()) parseTriple(xyz, Out6, 3);
  if (!rpy.empty()) parseTriple(rpy, Out6 + 3, 3);
}

bool parseUrdf(const std::string &S, UrdfModel &M) {
  // Links.
  size_t Pos = 0;
  while (true) {
    size_t OE, CA;
    size_t O = findElement(S, "link", Pos, OE, CA);
    if (O == std::string::npos) break;
    size_t Inner = (CA == O) ? OE : CA; // self-closing vs container
    UrdfLink L;
    L.name = xmlAttr(S, O, OE, "name");
    // First <visual>'s geometry + origin (within this link's extent).
    size_t VOE, VCA;
    size_t V = findElement(S, "visual", OE, VOE, VCA);
    if (V != std::string::npos && V < Inner) {
      size_t VEnd = (VCA == V) ? VOE : VCA;
      parseOrigin(S, VOE, VEnd, L.geom.vorigin);
      size_t GOE, GCA;
      size_t G = findElement(S, "geometry", VOE, GOE, GCA);
      if (G != std::string::npos && G < VEnd) {
        size_t GEnd = (GCA == G) ? GOE : GCA;
        size_t boe, boc;
        if (findElement(S, "box", GOE, boe, boc) < GEnd) {
          L.geom.type = "box";
          parseTriple(xmlAttr(S, S.find("<box", GOE), boe, "size"), L.geom.box, 3);
        } else {
          size_t coe, coc;
          if (findElement(S, "cylinder", GOE, coe, coc) < GEnd) {
            L.geom.type = "cylinder";
            size_t cp = S.find("<cylinder", GOE);
            L.geom.radius = std::atof(xmlAttr(S, cp, coe, "radius").c_str());
            L.geom.length = std::atof(xmlAttr(S, cp, coe, "length").c_str());
          } else {
            size_t soe, soc;
            if (findElement(S, "sphere", GOE, soe, soc) < GEnd) {
              L.geom.type = "sphere";
              size_t sp = S.find("<sphere", GOE);
              L.geom.radius = std::atof(xmlAttr(S, sp, soe, "radius").c_str());
            }
          }
        }
      }
    }
    M.links.push_back(L);
    Pos = (CA == O) ? OE + 1 : CA + 6;
  }
  // Joints.
  Pos = 0;
  while (true) {
    size_t OE, CA;
    size_t O = findElement(S, "joint", Pos, OE, CA);
    if (O == std::string::npos) break;
    size_t Inner = (CA == O) ? OE : CA;
    UrdfJoint J;
    J.name = xmlAttr(S, O, OE, "name");
    J.type = xmlAttr(S, O, OE, "type");
    size_t poe, pca;
    size_t P = findElement(S, "parent", OE, poe, pca);
    if (P != std::string::npos && P < Inner) J.parent = xmlAttr(S, P, poe, "link");
    size_t coe, cca;
    size_t C = findElement(S, "child", OE, coe, cca);
    if (C != std::string::npos && C < Inner) J.child = xmlAttr(S, C, coe, "link");
    parseOrigin(S, OE, Inner, J.origin);
    size_t aoe, aca;
    size_t A = findElement(S, "axis", OE, aoe, aca);
    if (A != std::string::npos && A < Inner) {
      std::string ax = xmlAttr(S, A, aoe, "xyz");
      if (!ax.empty()) parseTriple(ax, J.axis, 3);
    }
    M.joints.push_back(J);
    Pos = (CA == O) ? OE + 1 : CA + 7;
  }
  return !M.links.empty();
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
  std::string MeshErr;
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
    // glTF/GLB mesh import — embed the asset inline as a data URL (Tier 3).
    if (const std::string *MeshP = param(B, "mesh")) {
      std::string Rel = *MeshP;
      std::string Ext;
      auto Dot = Rel.rfind('.');
      if (Dot != std::string::npos) Ext = Rel.substr(Dot);
      std::string Path = (!Opts.ModelDir.empty() && !Rel.empty() &&
                          Rel.front() != '/')
                             ? Opts.ModelDir + "/" + Rel
                             : Rel;
      std::string Bytes;
      if (!readFile(Path, Bytes)) {
        MeshErr = "signal_actor3d \"" + B.Id + "\": cannot read mesh \"" + Rel +
                  "\" (resolved to " + Path + ")";
        break;
      }
      Actors << ",\"mesh\":" << jsonStr(meshDataUrl(Bytes, Ext));
      Actors << ",\"meshExt\":" << jsonStr(Ext.empty() ? ".glb" : Ext);
    }
    // URDF import (Tier 3b): parse the link/joint tree and emit it; the viewer
    // builds one node per link and rotates each joint by jointAngles[qIndex].
    if (const std::string *UrdfP = param(B, "urdf")) {
      std::string Path = (!Opts.ModelDir.empty() && !UrdfP->empty() &&
                          UrdfP->front() != '/')
                             ? Opts.ModelDir + "/" + *UrdfP
                             : *UrdfP;
      std::string Xml;
      UrdfModel U;
      if (!readFile(Path, Xml) || !parseUrdf(Xml, U)) {
        MeshErr = "signal_actor3d \"" + B.Id + "\": cannot read/parse URDF \"" +
                  *UrdfP + "\" (resolved to " + Path + ")";
        break;
      }
      auto arr6 = [](std::ostringstream &O, const double *V, int N) {
        O << "[";
        for (int i = 0; i < N; ++i) O << (i ? "," : "") << V[i];
        O << "]";
      };
      Actors << ",\"urdf\":{\"links\":[";
      for (size_t li = 0; li < U.links.size(); ++li) {
        const auto &L = U.links[li];
        Actors << (li ? "," : "") << "{\"name\":" << jsonStr(L.name)
               << ",\"geom\":" << jsonStr(L.geom.type)
               << ",\"box\":"; arr6(Actors, L.geom.box, 3);
        Actors << ",\"radius\":" << L.geom.radius
               << ",\"length\":" << L.geom.length << ",\"vorigin\":";
        arr6(Actors, L.geom.vorigin, 6);
        Actors << "}";
      }
      Actors << "],\"joints\":[";
      int QIdx = 0;
      for (size_t ji = 0; ji < U.joints.size(); ++ji) {
        const auto &J = U.joints[ji];
        bool Movable = J.type != "fixed";
        Actors << (ji ? "," : "") << "{\"name\":" << jsonStr(J.name)
               << ",\"type\":" << jsonStr(J.type)
               << ",\"parent\":" << jsonStr(J.parent)
               << ",\"child\":" << jsonStr(J.child) << ",\"origin\":";
        arr6(Actors, J.origin, 6);
        Actors << ",\"axis\":"; arr6(Actors, J.axis, 3);
        Actors << ",\"q\":" << (Movable && QIdx < 12 ? QIdx : -1) << "}";
        if (Movable && QIdx < 12) ++QIdx;
      }
      Actors << "]}";
    }
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
    // Column refs: 9 transform (scale defaults to 1) + any joint angles q1..qN.
    std::vector<long> Ref;
    std::vector<double> Def;
    for (int C = 0; C < 9; ++C) {
      auto It = ColIdx.find(B.Id + "[" + Comp[C] + "]");
      Ref.push_back(It != ColIdx.end() ? (long)It->second : -1);
      Def.push_back(C >= 6 ? 1.0 : 0.0);
    }
    for (int Q = 1; Q <= 12; ++Q) {
      auto It = ColIdx.find(B.Id + "[q" + std::to_string(Q) + "]");
      if (It == ColIdx.end()) break;
      Ref.push_back((long)It->second);
      Def.push_back(0.0);
    }
    Actors << ",\"keys\":[";
    size_t Rows = Times.size();
    for (size_t R = 0; R < Rows; ++R) {
      Actors << (R ? "," : "") << "[";
      for (size_t C = 0; C < Ref.size(); ++C) {
        double V = Def[C];
        if (Ref[C] >= 0 && R < Cols[Ref[C]].size()) V = Cols[Ref[C]][R].Value;
        Actors << (C ? "," : "") << V;
      }
      Actors << "]";
    }
    Actors << "]}";
    ++ActorCount;
  }
  if (!MeshErr.empty()) { Err = MeshErr; return false; }

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
  // Viewer-side physics engine (Tier 4) — only when the world enables physics.
  bool WorldPhysics = paramBool(*World, "physics", false);
  std::string Engine = paramOr(*World, "engine", "havok");
  if (WorldPhysics) {
    if (Engine == "ammo")
      OS << "<script src=\"https://cdn.jsdelivr.net/npm/ammo.js@0.0.10/builds/"
            "ammo.js\"></script>\n";
    else
      OS << "<script src=\"" << Cdn << "/havok/HavokPhysics_umd.js\"></script>\n";
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
    vis.rotationQuaternion = BABYLON.Quaternion.RotationYawPitchRoll(vo[5],vo[4],vo[3]);
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
    const baseQ = BABYLON.Quaternion.RotationYawPitchRoll(J.origin[5],J.origin[4],J.origin[3]);
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
  if (a.urdf) {
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
    m.rotationQuaternion = BABYLON.Quaternion.RotationYawPitchRoll(k[5], k[4], k[3]);
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
