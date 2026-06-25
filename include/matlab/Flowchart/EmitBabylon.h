#pragma once

//===----------------------------------------------------------------------===//
// mflow-3d-animation — `matlabc -emit-mflowlink-babylon model.mflow`.
//
// Lowers a signal-flow `.mflow` 3-D scene (one `signal_world3d` + any number of
// `signal_actor3d` blocks) plus the recorded per-step transform timeline to a
// single self-contained HTML document that renders and plays the animation in
// Babylon.js. The scene-graph, the keyframe timeline, and all viewer logic are
// embedded inline; the Babylon/Havok engine itself is referenced from a pinned
// CDN by default (override the base URL with `--babylon-cdn <url>`), with full
// asset vendoring left as a packaging follow-on.
//
// Authoritative dynamics stay in `MflowLinkSim` (the timeline is the recorded
// log); Havok in the viewer is visualization-only. See
// `docs/mflowlink_3d_animation_roadmap.md` and the `mflow-3d-animation`
// OpenSpec change.
//===----------------------------------------------------------------------===//

#include <iosfwd>
#include <string>

namespace matlab {
namespace flowchart {

struct MflowLinkModel;
class MflowLinkSim;

struct BabylonEmitOptions {
  // Base CDN host for the Babylon engine (core, loaders, Havok). The default
  // pins the public Babylon CDN; `--babylon-cdn <url>` overrides it (e.g. to a
  // self-hosted mirror for air-gapped use).
  std::string CdnBase = "https://cdn.babylonjs.com";
  // Directory of the source .mflow, used to resolve a `signal_actor3d` relative
  // `mesh` path (glTF/GLB) for embedding. Empty ⇒ resolve against the CWD.
  std::string ModelDir;
  // Tier-6 placeholder — when set, the viewer auto-plays for headless capture.
  bool Record = false;
};

// Emit the self-contained HTML to `OS`. The simulation must already have run
// (`Sim.runToCompletion()`), so the transform timeline is populated. Returns
// false and fills `Err` on a structural problem (e.g. no `signal_world3d`).
bool emitMflowLinkBabylon(const MflowLinkModel &M, const MflowLinkSim &Sim,
                          std::ostream &OS, std::string &Err,
                          const BabylonEmitOptions &Opts = {});

} // namespace flowchart
} // namespace matlab
