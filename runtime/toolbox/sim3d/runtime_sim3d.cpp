//===----------------------------------------------------------------------===//
// Simulink-3D-Animation command-line surface — `sim3d.*` runtime.
//
// Backs the MATLAB-side handle classes `sim3d_World` / `sim3d_Actor` (parsed
// from `sim3d.World` / `sim3d.Actor`; see runtime/toolbox/sim3d/sim3d_classdefs.m
// and the parser fold in lib/Parse/Parser.cpp). The classdef method bodies are
// one-liners that forward the receiver `obj` (and matrix/scalar args) into the
// `matlab_sim3d_*` entries below — the same System-Object convention as the DSP
// toolbox (runtime/toolbox/dsp/runtime_dsp.cpp).
//
// All scene + timeline state lives here, keyed by the handle object pointer
// (stable across calls for a given handle). `run(world, dt)` records one
// keyframe of every added actor's current transform; `export(world, path)`
// builds the scene JSON and writes a self-contained Babylon.js HTML player via
// the shared writer (include/matlab/Flowchart/BabylonDocument.h), byte-for-byte
// the same shell + viewer as the block-diagram `-emit-mflowlink-babylon` path.
//===----------------------------------------------------------------------===//

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include "matlab/Flowchart/BabylonDocument.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---- runtime object/string property accessors (defined in matlab_runtime) --
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void       *matlab_obj_get_string(matlab_obj *o, const char *name, int64_t len);

namespace {

// Mirror of the runtime string descriptor (char* + length, not zero-terminated).
struct Sim3dString { char *data; int64_t len; };
std::string toStr(const void *s) {
  if (!s) return {};
  const Sim3dString *p = reinterpret_cast<const Sim3dString *>(s);
  if (!p->data || p->len <= 0) return {};
  return std::string(p->data, p->data + static_cast<size_t>(p->len));
}

// Copy a matrix property / argument into a fixed-width vector, padding with the
// element-wise defaults. A scalar broadcasts to every slot (so `Scale = 2`
// fills [2 2 2]).
void readVec(const matlab_mat *m, double *out, int n, const double *def) {
  for (int i = 0; i < n; ++i) out[i] = def ? def[i] : 0.0;
  if (!m || !m->data) return;
  int64_t len = m->rows * m->cols;
  if (len == 1) { for (int i = 0; i < n; ++i) out[i] = m->data[0]; return; }
  for (int i = 0; i < n && i < len; ++i) out[i] = m->data[i];
}

struct Actor {
  std::string id, name, shape{"box"};
  double size[3]  = {1, 1, 1};
  double radius = 0.5, height = 1.0;
  double color[3]    = {0.6, 0.6, 0.6};
  double emissive[3] = {0, 0, 0};
  double opacity = 1.0;
  // Current transform: translation, rotation (rpy), scale.
  double t[3] = {0, 0, 0};
  double r[3] = {0, 0, 0};
  double s[3] = {1, 1, 1};
  // Recorded keyframes: nine components [tx,ty,tz,rx,ry,rz,sx,sy,sz] per run().
  std::vector<std::array<double, 9>> keys;
};

struct World {
  std::vector<matlab_obj *> actorOrder; // registration order
  std::vector<double> times;
  bool opened = false;
};

std::map<matlab_obj *, World> g_worlds;
std::map<matlab_obj *, Actor> g_actors;

// Lazily fetch/register the actor record for a handle object. Name/Shape are
// filled in by matlab_sim3d_actor_new (the constructor); a lazily-created record
// (e.g. from a setter that ran before actor_new) gets a synthetic id.
Actor &actorOf(matlab_obj *o) {
  auto It = g_actors.find(o);
  if (It != g_actors.end()) return It->second;
  Actor A;
  A.id = "actor" + std::to_string(g_actors.size() + 1);
  return g_actors.emplace(o, std::move(A)).first->second;
}

std::string jsonStr(const std::string &S) {
  std::string Out = "\"";
  for (char C : S) {
    switch (C) {
    case '"':  Out += "\\\""; break;
    case '\\': Out += "\\\\"; break;
    case '\n': Out += "\\n";  break;
    case '\r': Out += "\\r";  break;
    case '\t': Out += "\\t";  break;
    default:   Out += C;
    }
  }
  return Out + "\"";
}

void arr(std::ostringstream &O, const double *V, int N) {
  O << "[";
  for (int I = 0; I < N; ++I) O << (I ? "," : "") << V[I];
  O << "]";
}

} // namespace

extern "C" {

// world = sim3d.World(): register an empty scene for this handle.
void *matlab_sim3d_world_new(void *world_v) {
  if (world_v) g_worlds[reinterpret_cast<matlab_obj *>(world_v)] = World{};
  return world_v;
}

// actor = sim3d.Actor(name, shape): register the actor. name/shape arrive as
// MATLAB string objects ({char* data; int64 len}).
void *matlab_sim3d_actor_new(void *actor_v, void *name_v, void *shape_v) {
  if (!actor_v) return actor_v;
  Actor &A = actorOf(reinterpret_cast<matlab_obj *>(actor_v));
  std::string nm = toStr(name_v), sh = toStr(shape_v);
  if (!nm.empty()) { A.name = nm; A.id = nm; }
  if (!sh.empty()) {
    // Validate against the supported primitive set; fall back to a box.
    if (sh != "box" && sh != "sphere" && sh != "cylinder" && sh != "plane" &&
        sh != "cone" && sh != "capsule") {
      fprintf(stderr,
              "sim3d.Actor: unsupported shape '%s' (expected one of "
              "box/sphere/cylinder/plane/cone/capsule); using 'box'\n",
              sh.c_str());
      sh = "box";
    }
    A.shape = sh;
  }
  return actor_v;
}

// a.Translation = v  /  a.Rotation = v  /  a.Scale = v  (Dependent setters).
void matlab_sim3d_set_translation(void *actor_v, matlab_mat *v) {
  if (!actor_v) return;
  static const double def[3] = {0, 0, 0};
  readVec(v, actorOf(reinterpret_cast<matlab_obj *>(actor_v)).t, 3, def);
}
void matlab_sim3d_set_rotation(void *actor_v, matlab_mat *v) {
  if (!actor_v) return;
  static const double def[3] = {0, 0, 0};
  readVec(v, actorOf(reinterpret_cast<matlab_obj *>(actor_v)).r, 3, def);
}
void matlab_sim3d_set_scale(void *actor_v, matlab_mat *v) {
  if (!actor_v) return;
  static const double def[3] = {1, 1, 1};
  readVec(v, actorOf(reinterpret_cast<matlab_obj *>(actor_v)).s, 3, def);
}
void matlab_sim3d_set_color(void *actor_v, matlab_mat *v) {
  if (!actor_v) return;
  static const double def[3] = {0.6, 0.6, 0.6};
  readVec(v, actorOf(reinterpret_cast<matlab_obj *>(actor_v)).color, 3, def);
}
void matlab_sim3d_set_size(void *actor_v, matlab_mat *v) {
  if (!actor_v) return;
  static const double def[3] = {1, 1, 1};
  readVec(v, actorOf(reinterpret_cast<matlab_obj *>(actor_v)).size, 3, def);
}

// Getters return the stored transform as a 1x3 row (get.Translation etc.).
matlab_mat *matlab_sim3d_get_translation(void *actor_v) {
  matlab_mat *m = mat_alloc(1, 3);
  if (actor_v) { const double *t = actorOf(reinterpret_cast<matlab_obj *>(actor_v)).t;
                 m->data[0] = t[0]; m->data[1] = t[1]; m->data[2] = t[2]; }
  return m;
}
matlab_mat *matlab_sim3d_get_rotation(void *actor_v) {
  matlab_mat *m = mat_alloc(1, 3);
  if (actor_v) { const double *r = actorOf(reinterpret_cast<matlab_obj *>(actor_v)).r;
                 m->data[0] = r[0]; m->data[1] = r[1]; m->data[2] = r[2]; }
  return m;
}
matlab_mat *matlab_sim3d_get_scale(void *actor_v) {
  matlab_mat *m = mat_alloc(1, 3);
  if (actor_v) { const double *s = actorOf(reinterpret_cast<matlab_obj *>(actor_v)).s;
                 m->data[0] = s[0]; m->data[1] = s[1]; m->data[2] = s[2]; }
  return m;
}

// add(world, actor): register the actor into the world's draw order.
void matlab_sim3d_add(void *world_v, void *actor_v) {
  if (!world_v || !actor_v) return;
  auto *w = reinterpret_cast<matlab_obj *>(world_v);
  auto *a = reinterpret_cast<matlab_obj *>(actor_v);
  actorOf(a);
  World &W = g_worlds[w];
  for (auto *e : W.actorOrder) if (e == a) return;
  W.actorOrder.push_back(a);
}

// open(world): begin recording (clears any prior timeline).
void matlab_sim3d_open(void *world_v) {
  if (!world_v) return;
  World &W = g_worlds[reinterpret_cast<matlab_obj *>(world_v)];
  W.opened = true;
  W.times.clear();
  for (auto *a : W.actorOrder) g_actors[a].keys.clear();
}

// run(world, dt): record one keyframe of every actor's current transform.
void matlab_sim3d_run(void *world_v, double dt) {
  if (!world_v) return;
  World &W = g_worlds[reinterpret_cast<matlab_obj *>(world_v)];
  if (!W.opened) {
    fprintf(stderr, "sim3d: run() called before open(); ignoring\n");
    return;
  }
  if (W.actorOrder.empty())
    fprintf(stderr, "sim3d: run() on a world with no actors\n");
  // First sample at t=0, then advance by dt each call.
  W.times.push_back(W.times.empty() ? 0.0 : W.times.back() + dt);
  for (auto *a : W.actorOrder) {
    Actor &A = g_actors[a];
    A.keys.push_back({A.t[0], A.t[1], A.t[2], A.r[0], A.r[1], A.r[2],
                      A.s[0], A.s[1], A.s[2]});
  }
}

// close(world): no-op finaliser (kept for API symmetry / future flushing).
void matlab_sim3d_close(void *world_v) { (void)world_v; }

// export(world, path): build the scene JSON and write the Babylon HTML player.
// `path` arrives as a MATLAB string object ({char* data; int64 len}).
void matlab_sim3d_export(void *world_v, void *path_v) {
  if (!world_v) return;
  auto *w = reinterpret_cast<matlab_obj *>(world_v);
  auto WIt = g_worlds.find(w);
  if (WIt == g_worlds.end()) return;
  World &W = WIt->second;
  std::string Path = toStr(path_v);
  if (Path.empty()) Path = "sim3d_scene.html";
  std::string Title = "sim3d";

  std::ostringstream Actors;
  size_t Count = 0;
  for (auto *a : W.actorOrder) {
    Actor &A = g_actors[a];
    // Name/Shape come from the constructor (string args). Geometry/material
    // live as plain numeric object properties; read their latest values
    // straight off the handle at export time.
    static const double cdef[3] = {0.6, 0.6, 0.6};
    static const double sdef[3] = {1, 1, 1};
    readVec(matlab_obj_get_mat(a, "Color", 5), A.color, 3, cdef);
    readVec(matlab_obj_get_mat(a, "Size", 4), A.size, 3, sdef);
    if (Count) Actors << ",\n";
    Actors << "    {";
    Actors << "\"id\":" << jsonStr(A.id);
    Actors << ",\"name\":" << jsonStr(A.name.empty() ? A.id : A.name);
    Actors << ",\"shape\":" << jsonStr(A.shape);
    Actors << ",\"size\":"; arr(Actors, A.size, 3);
    Actors << ",\"radius\":" << A.radius;
    Actors << ",\"height\":" << A.height;
    Actors << ",\"color\":"; arr(Actors, A.color, 3);
    Actors << ",\"emissive\":"; arr(Actors, A.emissive, 3);
    Actors << ",\"opacity\":" << A.opacity;
    Actors << ",\"keys\":[";
    for (size_t R = 0; R < A.keys.size(); ++R) {
      Actors << (R ? "," : "") << "[";
      for (int C = 0; C < 9; ++C) Actors << (C ? "," : "") << A.keys[R][C];
      Actors << "]";
    }
    Actors << "]}";
    ++Count;
  }

  std::ostringstream Scene;
  Scene << "{\n";
  Scene << "  \"world\":{\"gravity\":[0,0,-9.81],\"viewpoint\":[8,8,6]"
           ",\"engine\":\"havok\",\"physics\":false,\"showGround\":true"
           ",\"showAxes\":true,\"background\":[0.07,0.08,0.1]"
           ",\"pacingRate\":1},\n";
  Scene << "  \"times\":[";
  for (size_t I = 0; I < W.times.size(); ++I) Scene << (I ? "," : "") << W.times[I];
  Scene << "],\n";
  Scene << "  \"lights\":[],\n";
  Scene << "  \"cameras\":[],\n";
  Scene << "  \"actors\":[\n" << Actors.str() << "\n  ]\n";
  Scene << "}";

  std::ofstream Out(Path, std::ios::binary);
  if (!Out) return;
  matlab::babylon::DocParams Doc;
  Doc.Title = Title;
  matlab::babylon::writeDocument(Out, Scene.str(), Doc);
}

} // extern "C"
