#include "matlab/Flowchart/MflowLinkSim.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// MflowLinkSim — the Tier-C continuous-time simulation interpreter.
//
// One in-process pass per major step:
//   1. Resolve every block's input from `Out_` (the previous-step
//      outputs for loop-breakers; this step's outputs for everything
//      already evaluated, since `M_.ExecOrder` is topo-sorted with
//      loop-breakers' outgoing edges dropped).
//   2. Call its evaluator, writing `Out_` and, if a state derivative
//      buffer is being filled, the per-block dx/dt slice.
//   3. The integrator (classic RK4) calls evalAll four times to
//      collect k1..k4, advances Y_ by Σ wi·ki·h.
//
// Inputs are assumed scalar `double` throughout Tier C — sufficient
// for the lowpass / pid_tracking demos. Vector signals + bus types
// are Tier H.
//===----------------------------------------------------------------------===//

namespace matlab::flowchart {

namespace {

// "1, 2, 1" → {1, 2, 1}. Trims whitespace; silently drops malformed
// tokens (the loader has already validated structural shape).
std::vector<double> parsePoly(const std::string &S) {
  std::vector<double> Out;
  std::stringstream SS(S);
  std::string Tok;
  while (std::getline(SS, Tok, ',')) {
    size_t A = Tok.find_first_not_of(" \t");
    if (A == std::string::npos) continue;
    size_t B = Tok.find_last_not_of(" \t");
    try {
      Out.push_back(std::stod(Tok.substr(A, B - A + 1)));
    } catch (...) {
    }
  }
  return Out;
}

const std::string *paramS(const MflBlock &B, const char *Key) {
  auto It = B.Params.find(Key);
  return It == B.Params.end() ? nullptr : &It->second;
}

} // namespace

double MflowLinkSim::paramD(const MflBlock &B, const char *Key, double Def) {
  auto It = B.Params.find(Key);
  if (It == B.Params.end()) return Def;
  try {
    return std::stod(It->second);
  } catch (...) {
    return Def;
  }
}

//===----------------------------------------------------------------------===//
// Construction — pre-compute everything per-block we'd otherwise
// reparse on every step.
//===----------------------------------------------------------------------===//

MflowLinkSim::MflowLinkSim(const MflowLinkModel &M) : M_(M) {
  const size_t N = M_.Blocks.size();
  Out_.assign(N, 0.0);
  Inputs_.assign(N, {});
  StateOffset_.assign(N, 0);
  TFCache_.assign(N, {});

  std::unordered_map<std::string, size_t> IdxOf;
  for (size_t I = 0; I < N; ++I) IdxOf[M_.Blocks[I].Id] = I;

  // Per-block continuous-state offset.
  size_t Off = 0;
  for (size_t I = 0; I < N; ++I) {
    StateOffset_[I] = Off;
    Off += static_cast<size_t>(M_.Blocks[I].ContStateCount);
  }
  Y_.assign(M_.ContStateCount, 0.0);

  // Input wiring. Sum / Product blocks read in1, in2, …; every other
  // single-input block reads "in".
  for (auto &E : M_.Edges) {
    auto FI = IdxOf.find(E.FromBlock);
    auto TI = IdxOf.find(E.ToBlock);
    if (FI == IdxOf.end() || TI == IdxOf.end()) continue;
    Inputs_[TI->second].push_back({FI->second, E.ToPort});
  }

  // Cache transfer-function coefficients.
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    if (B.Kind != "signal_transfer_fcn") continue;
    auto *NumS = paramS(B, "num");
    auto *DenS = paramS(B, "den");
    TFCache_[I].Num = parsePoly(NumS ? *NumS : "1");
    TFCache_[I].Den = parsePoly(DenS ? *DenS : "1");
    TFCache_[I].Valid = !TFCache_[I].Den.empty();
  }

  StepSize_ = 0.01;
  if (M_.Solver.MaxStep != "auto") {
    try {
      double H = std::stod(M_.Solver.MaxStep);
      if (H > 0.0) StepSize_ = H;
    } catch (...) {
    }
  }

  if (M_.Snapshot.Enabled && M_.Snapshot.Depth > 0)
    SnapshotCap_ = static_cast<size_t>(M_.Snapshot.Depth);
  else
    SnapshotCap_ = 0;

  // Log columns: anything with `data.log_signal: true`, plus every
  // scope / to_workspace / display (they implicitly log).
  for (size_t I = 0; I < N; ++I) {
    const auto &B = M_.Blocks[I];
    bool Implicit = B.Kind == "signal_scope" ||
                    B.Kind == "signal_display" ||
                    B.Kind == "signal_to_workspace";
    if (!B.LogSignal && !Implicit) continue;
    LogBlocks_.push_back(I);
    // `signal_to_workspace` records under `params.variableName` so the
    // CSV column matches the IDE's intent; everything else uses the
    // block id.
    std::string Name = B.Id;
    if (B.Kind == "signal_to_workspace") {
      if (auto *V = paramS(B, "variableName")) Name = *V;
    }
    LogNames_.push_back(std::move(Name));
  }
  LogColumns_.assign(LogNames_.size(), {});
}

//===----------------------------------------------------------------------===//
// reset — pull initial conditions out of `params` into Y_, drop logs.
//===----------------------------------------------------------------------===//

void MflowLinkSim::reset() {
  T_ = M_.Solver.StartTime;
  MajorSteps_ = 0;
  std::fill(Y_.begin(), Y_.end(), 0.0);
  std::fill(Out_.begin(), Out_.end(), 0.0);
  for (auto &C : LogColumns_) C.clear();
  Snapshots_.clear();
  LogsTruncated_ = false;

  for (size_t I = 0; I < M_.Blocks.size(); ++I) {
    const auto &B = M_.Blocks[I];
    if (B.Kind == "signal_integrator") {
      Y_[StateOffset_[I]] = paramD(B, "initialCondition", 0.0);
    } else if (B.Kind == "signal_state_space") {
      // x0 may be a vector literal — Tier C reads scalar only; vector
      // ICs are picked up when state_space gets its full evaluator.
      double X0 = paramD(B, "x0", 0.0);
      for (int J = 0; J < B.ContStateCount; ++J)
        Y_[StateOffset_[I] + J] = X0;
    }
  }

  // Run one evaluation at t=startTime so the very first logged sample
  // reflects t=0 outputs, not the post-construction zeros.
  evalAll(T_, Y_.data(), nullptr);
  logSample();
}

//===----------------------------------------------------------------------===//
// evalAll — one pass over ExecOrder, writes Out_ and (if requested)
//           the dx/dt buffer.
//===----------------------------------------------------------------------===//

void MflowLinkSim::evalAll(double T, const double *State, double *Deriv) {
  if (Deriv) std::memset(Deriv, 0, sizeof(double) * Y_.size());

  auto inputOf = [&](size_t I, const char *PortId) -> double {
    for (auto &P : Inputs_[I])
      if (P.DstPort == PortId) return Out_[P.SrcBlock];
    return 0.0;
  };
  auto sumInput = [&](size_t I, const std::string &Port) -> double {
    // Multiple edges may land on the same input port — sum them
    // (matches Simulink's implicit summing convention for vector
    // joins, and harmless for the single-edge common case).
    double V = 0.0;
    for (auto &P : Inputs_[I])
      if (P.DstPort == Port) V += Out_[P.SrcBlock];
    return V;
  };

  for (size_t Pos = 0; Pos < M_.ExecOrder.size(); ++Pos) {
    size_t I = M_.ExecOrder[Pos];
    const auto &B = M_.Blocks[I];
    const std::string &K = B.Kind;

    if (K == "signal_constant") {
      Out_[I] = paramD(B, "value", 0.0);
    } else if (K == "signal_step") {
      double ST = paramD(B, "stepTime", 1.0);
      double IV = paramD(B, "initialValue", 0.0);
      double FV = paramD(B, "finalValue", 1.0);
      Out_[I] = (T >= ST) ? FV : IV;
    } else if (K == "signal_sine") {
      double A  = paramD(B, "amplitude", 1.0);
      double Bs = paramD(B, "bias", 0.0);
      double F  = paramD(B, "frequency", 1.0);
      double P  = paramD(B, "phase", 0.0);
      Out_[I] = A * std::sin(F * T + P) + Bs;
    } else if (K == "signal_ramp") {
      double Slope = paramD(B, "slope", 1.0);
      double Start = paramD(B, "startTime", 0.0);
      double Init  = paramD(B, "initialOutput", 0.0);
      Out_[I] = T >= Start ? Init + Slope * (T - Start) : Init;
    } else if (K == "signal_pulse") {
      double A  = paramD(B, "amplitude", 1.0);
      double Pp = paramD(B, "period", 1.0);
      double W  = paramD(B, "pulseWidth", 50.0);   // percent
      double D  = paramD(B, "phaseDelay", 0.0);
      double Tt = std::fmod(T - D, Pp);
      if (Tt < 0.0) Tt += Pp;
      Out_[I] = (Tt < Pp * W * 0.01) ? A : 0.0;
    } else if (K == "signal_gain") {
      Out_[I] = paramD(B, "gain", 1.0) * inputOf(I, "in");
    } else if (K == "signal_abs") {
      Out_[I] = std::fabs(inputOf(I, "in"));
    } else if (K == "signal_saturation") {
      double Lo = paramD(B, "lowerLimit", -1.0);
      double Hi = paramD(B, "upperLimit",  1.0);
      double U  = inputOf(I, "in");
      Out_[I] = U < Lo ? Lo : (U > Hi ? Hi : U);
    } else if (K == "signal_sum") {
      double Sum = 0.0;
      auto *Signs = paramS(B, "signs");
      if (Signs && !Signs->empty()) {
        for (size_t Ki = 0; Ki < Signs->size(); ++Ki) {
          char Sg = (*Signs)[Ki];
          double V = sumInput(I, "in" + std::to_string(Ki + 1));
          Sum += (Sg == '-') ? -V : V;
        }
      } else {
        // No `signs` declared — sum every connected input port.
        for (auto &P : Inputs_[I]) Sum += Out_[P.SrcBlock];
      }
      Out_[I] = Sum;
    } else if (K == "signal_product") {
      int NIn = static_cast<int>(paramD(B, "numInputs", 2.0));
      if (NIn < 1) NIn = 1;
      double Prod = 1.0;
      for (int Ki = 1; Ki <= NIn; ++Ki)
        Prod *= sumInput(I, "in" + std::to_string(Ki));
      Out_[I] = Prod;
    } else if (K == "signal_integrator") {
      size_t Off = StateOffset_[I];
      Out_[I] = State[Off];
      if (Deriv) Deriv[Off] = inputOf(I, "in");
    } else if (K == "signal_transfer_fcn") {
      const auto &TF = TFCache_[I];
      int N = static_cast<int>(TF.Den.size()) - 1;
      double U = inputOf(I, "in");
      if (N <= 0 || !TF.Valid) {
        // Static gain: y = (num[0]/den[0]) * u.
        double NumLead = TF.Num.empty() ? 0.0 : TF.Num.front();
        double DenLead = TF.Den.empty() ? 1.0 : TF.Den.front();
        Out_[I] = (NumLead / DenLead) * U;
      } else {
        // Controllable canonical form, den = a_n s^n + ... + a_0,
        // num = b_m s^m + ... + b_0 (right-padded to length n+1 with
        // leading zeros). State x_0..x_{n-1}; x'_k = x_{k+1} for k<n-1
        // and x'_{n-1} = (u - Σ a_k/a_n · x_k) / 1 (already normalised).
        size_t Off = StateOffset_[I];
        double Lead = TF.Den.front();
        if (Deriv) {
          for (int Ki = 0; Ki < N - 1; ++Ki)
            Deriv[Off + Ki] = State[Off + Ki + 1];
          double Last = U / Lead;
          // a_k coefficient → Den[N-k] / Lead.
          for (int Ki = 0; Ki < N; ++Ki)
            Last -= (TF.Den[N - Ki] / Lead) * State[Off + Ki];
          Deriv[Off + N - 1] = Last;
        }
        // Output y = Σ b_k · x_k for strictly proper TF. For
        // not-strictly-proper, the b_n direct-feedthrough term would
        // add (b_n / a_n) · u — but we marked those as non-loop-
        // breakers in lowering, and the topo sort already handles the
        // direct-feedthrough wiring. For Tier C we restrict to the
        // strictly-proper case (degNum < degDen); the loader / lower
        // already classified that as the loop-breaker path.
        std::vector<double> NPad(N + 1, 0.0);
        int Mdeg = static_cast<int>(TF.Num.size()) - 1;
        for (int Ki = 0; Ki <= Mdeg; ++Ki)
          NPad[N - Mdeg + Ki] = TF.Num[Ki];
        double Y = 0.0;
        for (int Ki = 0; Ki < N; ++Ki)
          Y += (NPad[N - Ki] / Lead) * State[Off + Ki];
        Out_[I] = Y;
      }
    } else if (K == "signal_state_space") {
      // Tier-C: only the scalar SISO case (n = ContStateCount, single
      // input/output) with D = 0 (the lowering already marked D ≠ 0
      // as a non-loop-breaker, which we don't support yet).
      auto parseMatrix = [](const std::string &S, std::vector<double> &Vals,
                            int &Rows, int &Cols) {
        Vals.clear();
        Rows = 0;
        Cols = 0;
        std::string T = S;
        auto F = T.find('[');
        if (F != std::string::npos) T.erase(0, F + 1);
        auto L = T.rfind(']');
        if (L != std::string::npos) T.erase(L);
        std::stringstream RS(T);
        std::string Row;
        while (std::getline(RS, Row, ';')) {
          int C = 0;
          std::stringstream CS(Row);
          std::string Tok;
          while (CS >> Tok) {
            try {
              Vals.push_back(std::stod(Tok));
            } catch (...) {
            }
            ++C;
          }
          if (C > 0) {
            ++Rows;
            Cols = C;
          }
        }
      };
      const std::string *AS = paramS(B, "A");
      const std::string *BS = paramS(B, "B");
      const std::string *CS = paramS(B, "C");
      std::vector<double> A, Bm, Cm;
      int Ar = 0, Ac = 0, Br = 0, Bc = 0, Cr = 0, Cc = 0;
      if (AS) parseMatrix(*AS, A, Ar, Ac);
      if (BS) parseMatrix(*BS, Bm, Br, Bc);
      if (CS) parseMatrix(*CS, Cm, Cr, Cc);
      int n = B.ContStateCount;
      size_t Off = StateOffset_[I];
      double U = inputOf(I, "in");
      if (Deriv && static_cast<int>(A.size()) == n * n &&
          static_cast<int>(Bm.size()) >= n) {
        for (int Ri = 0; Ri < n; ++Ri) {
          double D = Bm[Ri] * U;
          for (int Ci = 0; Ci < n; ++Ci)
            D += A[Ri * n + Ci] * State[Off + Ci];
          Deriv[Off + Ri] = D;
        }
      }
      double Y = 0.0;
      if (static_cast<int>(Cm.size()) >= n)
        for (int Ki = 0; Ki < n; ++Ki) Y += Cm[Ki] * State[Off + Ki];
      Out_[I] = Y;
    } else if (K == "signal_scope" || K == "signal_display" ||
               K == "signal_to_workspace" || K == "signal_terminator") {
      Out_[I] = inputOf(I, "in");
    } else if (K == "signal_mux") {
      // Tier-C: scalar signals, so a mux is just the first input.
      Out_[I] = Inputs_[I].empty() ? 0.0 : Out_[Inputs_[I].front().SrcBlock];
    } else if (K == "signal_demux" || K == "signal_switch") {
      // Algebra-only Tier-C stub: passthrough first input. Tier-E adds
      // the proper switch / demux semantics + zero-crossing.
      Out_[I] = Inputs_[I].empty() ? 0.0 : Out_[Inputs_[I].front().SrcBlock];
    } else if (K == "signal_unit_delay" || K == "signal_zoh") {
      // Discrete blocks ride along as a fixed value (their current
      // output) until Tier E adds the sample-time scheduler. The
      // lowering already marked them as loop-breakers so the topo
      // order doesn't try to read them mid-step.
      size_t Off = StateOffset_[I];
      (void)Off;
      // Default to their initialValue param — picked up at reset()
      // once we wire discrete state through Y_; until then, 0.
      Out_[I] = paramD(B, "initialValue", 0.0);
    } else {
      // Loader-level reserved kinds are rejected at lowering, so
      // anything reaching here is an evaluator gap — treat as
      // passthrough so simulation doesn't crash and the user sees
      // the wrong-but-finite result instead of a segfault.
      Out_[I] = Inputs_[I].empty() ? 0.0 : Out_[Inputs_[I].front().SrcBlock];
    }
  }
}

void MflowLinkSim::derivative(double T, const double *State, double *Deriv) {
  evalAll(T, State, Deriv);
}

//===----------------------------------------------------------------------===//
// Classic RK4 fixed-step integrator.
//
// Picked over Dormand-Prince for Tier C because it's the smallest
// thing that gives the demos sensible answers; the existing matlab_*
// ode45 builtins can be plumbed in once `-emit-mflowlink-cpp` (Tier G)
// lands. The user-facing `solver.algorithm` field is passed through
// the IR untouched so the choice surfaces in `--dry-run` output.
//===----------------------------------------------------------------------===//

double MflowLinkSim::stepMajor() {
  const size_t Nx = Y_.size();
  double H = StepSize_;
  // Clamp the final step so we land exactly on stopTime; if we're
  // already there (within an absolute tick), refuse to step. This
  // keeps `runToCompletion`'s loop from logging a duplicate sample
  // at t = stopTime when float drift leaves T_ a few ULP below it.
  if (T_ + H > M_.Solver.StopTime) H = M_.Solver.StopTime - T_;
  if (H <= 1e-12) return 0.0;

  // Snapshot BEFORE we step so step-back can restore exactly here.
  // After a step-back the user is rewriting an alternate future:
  // truncate the log columns down to the snapshot's row count so
  // they don't grow back interleaved.
  if (LogsTruncated_) {
    size_t Rows = LogColumns_.empty() ? 0 : LogColumns_.front().size();
    (void)Rows;
    LogsTruncated_ = false;
  }
  pushSnapshot();

  std::vector<double> K1(Nx), K2(Nx), K3(Nx), K4(Nx), Yt(Nx);
  if (Nx > 0) {
    derivative(T_, Y_.data(), K1.data());
    for (size_t I = 0; I < Nx; ++I) Yt[I] = Y_[I] + 0.5 * H * K1[I];
    derivative(T_ + 0.5 * H, Yt.data(), K2.data());
    for (size_t I = 0; I < Nx; ++I) Yt[I] = Y_[I] + 0.5 * H * K2[I];
    derivative(T_ + 0.5 * H, Yt.data(), K3.data());
    for (size_t I = 0; I < Nx; ++I) Yt[I] = Y_[I] + H * K3[I];
    derivative(T_ + H, Yt.data(), K4.data());
    for (size_t I = 0; I < Nx; ++I)
      Y_[I] += (H / 6.0) * (K1[I] + 2.0 * K2[I] + 2.0 * K3[I] + K4[I]);
  }
  T_ += H;
  ++MajorSteps_;

  // End-of-step output refresh + logging at the new time.
  evalAll(T_, Y_.data(), nullptr);
  logSample();
  return H;
}

void MflowLinkSim::runToCompletion() {
  reset();
  while (T_ < M_.Solver.StopTime - 1e-15) {
    double H = stepMajor();
    if (H <= 0.0) break;
  }
}

//===----------------------------------------------------------------------===//
// Log + CSV
//===----------------------------------------------------------------------===//

void MflowLinkSim::logSample() {
  for (size_t Ci = 0; Ci < LogBlocks_.size(); ++Ci)
    LogColumns_[Ci].push_back({T_, Out_[LogBlocks_[Ci]]});
}

void MflowLinkSim::writeCsv(std::ostream &OS) const {
  OS << "t";
  for (auto &N : LogNames_) OS << "," << N;
  OS << "\n";
  size_t Rows = LogColumns_.empty() ? 0 : LogColumns_.front().size();
  OS.setf(std::ios::scientific);
  OS.precision(9);
  for (size_t R = 0; R < Rows; ++R) {
    OS << LogColumns_.front()[R].T;
    for (auto &C : LogColumns_) OS << "," << C[R].Value;
    OS << "\n";
  }
}

void MflowLinkSim::pushSnapshot() {
  if (SnapshotCap_ == 0) return;
  Snapshot S;
  S.T = T_;
  S.MajorSteps = MajorSteps_;
  S.Y = Y_;
  S.Out = Out_;
  S.LogRows = LogColumns_.empty() ? 0 : LogColumns_.front().size();
  if (Snapshots_.size() >= SnapshotCap_)
    Snapshots_.erase(Snapshots_.begin());
  Snapshots_.push_back(std::move(S));
}

bool MflowLinkSim::stepBackMajor() {
  if (Snapshots_.empty()) return false;
  Snapshot S = std::move(Snapshots_.back());
  Snapshots_.pop_back();
  T_ = S.T;
  MajorSteps_ = S.MajorSteps;
  Y_ = std::move(S.Y);
  Out_ = std::move(S.Out);
  // Truncate any log rows newer than this snapshot — they belong to
  // the now-rewound timeline.
  for (auto &C : LogColumns_)
    if (C.size() > S.LogRows) C.resize(S.LogRows);
  LogsTruncated_ = true;
  return true;
}

std::vector<std::pair<std::string, double>>
MflowLinkSim::currentLoggedOutputs() const {
  std::vector<std::pair<std::string, double>> Out;
  Out.reserve(LogBlocks_.size());
  for (size_t Ci = 0; Ci < LogBlocks_.size(); ++Ci)
    Out.emplace_back(LogNames_[Ci], Out_[LogBlocks_[Ci]]);
  return Out;
}

} // namespace matlab::flowchart
