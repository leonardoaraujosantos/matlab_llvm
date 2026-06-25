//===----------------------------------------------------------------------===//
// mflow-3d-animation — Babylon.js scene + timeline emitter.
// See include/matlab/Flowchart/EmitBabylon.h.
//===----------------------------------------------------------------------===//

#include "matlab/Flowchart/EmitBabylon.h"

#include "matlab/Flowchart/BabylonDocument.h"
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
  std::string Mime;
  if (Ext == ".glb") Mime = "model/gltf-binary";
  else if (Ext == ".gltf") Mime = "model/gltf+json";
  else if (Ext == ".stl") Mime = "model/stl";
  else if (Ext == ".obj") Mime = "model/obj";
  else Mime = "application/octet-stream";
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
    // Text annotation (Tier 6.3) — a billboarded label actor.
    if (const std::string *Txt = param(B, "text"))
      Actors << ",\"text\":" << jsonStr(*Txt);
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
  Scene << ",\"pacingRate\":" << paramNum(*World, "pacingRate", 1.0);
  Scene << "},\n";
  Scene << "  \"times\":[";
  for (size_t I = 0; I < Times.size(); ++I)
    Scene << (I ? "," : "") << Times[I];
  Scene << "],\n";
  Scene << "  \"lights\":[" << Lights.str() << "],\n";
  Scene << "  \"cameras\":[" << Cameras.str() << "],\n";
  Scene << "  \"actors\":[\n" << Actors.str() << "\n  ]\n";
  Scene << "}";

  // The document. Scene JSON + all viewer logic are inline; only the engine is
  // CDN-referenced. The viewer maps the model's right-handed Z-up metres frame
  // onto Babylon by parenting everything under a root rotated -90° about X. The
  // shell + viewer JS are shared with the command-line sim3d path via
  // BabylonDocument.h so both stay byte-for-byte identical.
  babylon::DocParams Doc;
  Doc.Title = M.EntryName;
  Doc.CdnBase = Opts.CdnBase;
  Doc.WorldPhysics = paramBool(*World, "physics", false);
  Doc.Engine = paramOr(*World, "engine", "havok");
  // Inline a user-provided engine bundle (`--babylon-inline`) for a fully
  // network-free artifact, else the writer references the pinned CDN.
  if (!Opts.InlineEnginePath.empty()) {
    if (!readFile(Opts.InlineEnginePath, Doc.InlineEngine)) {
      Err = "--babylon-inline: cannot read engine bundle " + Opts.InlineEnginePath;
      return false;
    }
  }
  babylon::writeDocument(OS, Scene.str(), Doc);
  (void)Opts;
  return true;
}


} // namespace flowchart
} // namespace matlab
