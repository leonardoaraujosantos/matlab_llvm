// §17.5 #6 — cross-dialect composition: runtime helper that lets a
// compiled MATLAB program invoke a baked signal-flow simulation.
//
// MATLAB call site:
//
//   y = mflowlink_run('model.mflow');
//
// lowers (via Sema builtin registration + LowerTensorOps dispatch)
// to `matlab_mflowlink_run` defined below. The function loads the
// .mflow at the given path, lowers it through `SignalFlowLowering`,
// drives `MflowLinkSim::runToCompletion`, and returns a single-row
// `matlab_mat` whose columns are the final values of every logged
// signal in the simulation (in the order the IR's LogNames_ table
// reports). A missing or malformed file is reported to stderr and
// returns a 1×0 empty matrix.

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Flowchart/Loader.h"
#include "matlab/Flowchart/MflowLinkModel.h"
#include "matlab/Flowchart/MflowLinkSim.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// matlab_string + matlab_mat live in matlab_runtime.cpp. We
// re-declare the bits we touch here so this TU stays decoupled
// from the runtime header.
extern "C" {
struct matlab_string_s {
  char *data;
  long long len;
};
typedef struct matlab_string_s matlab_string;

struct matlab_mat;
// matlab_runtime.cpp's `matlab_mat_from_buf(buf, m, n)` constructs
// an m×n descriptor copying the user buffer. We hand it the row of
// final logged-signal values.
matlab_mat *matlab_mat_from_buf(const double *buf, double m, double n);
matlab_mat *matlab_empty_mat(void);
}

extern "C" matlab_mat *matlab_mflowlink_run(matlab_string *Path) {
  if (!Path || !Path->data || Path->len <= 0) {
    std::fprintf(stderr,
                 "mflowlink_run: empty / null path argument\n");
    return matlab_empty_mat();
  }
  std::string PathStr(Path->data, static_cast<size_t>(Path->len));
  matlab::SourceManager SM;
  matlab::DiagnosticEngine Diag(SM);
  auto Doc = matlab::flowchart::loadMflowFromPath(SM, PathStr, Diag);
  if (!Doc) {
    Diag.printAll();
    std::fprintf(stderr, "mflowlink_run: failed to load \"%s\"\n",
                 PathStr.c_str());
    return matlab_empty_mat();
  }
  if (!Doc->isSignalFlow()) {
    std::fprintf(stderr,
                 "mflowlink_run: \"%s\" is not a signal-flow .mflow\n",
                 PathStr.c_str());
    return matlab_empty_mat();
  }
  auto Model = matlab::flowchart::lowerSignalFlow(*Doc, Diag);
  Diag.printAll();
  if (!Model) {
    std::fprintf(stderr,
                 "mflowlink_run: lowering failed for \"%s\"\n",
                 PathStr.c_str());
    return matlab_empty_mat();
  }
  matlab::flowchart::MflowLinkSim Sim(*Model);
  Sim.runToCompletion();
  auto Final = Sim.currentLoggedOutputs();
  std::vector<double> Buf;
  Buf.reserve(Final.size());
  for (auto &P : Final) Buf.push_back(P.second);
  return matlab_mat_from_buf(Buf.data(), 1.0,
                             static_cast<double>(Buf.size()));
}
