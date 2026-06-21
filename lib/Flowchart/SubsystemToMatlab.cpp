//===----------------------------------------------------------------------===//
// SubsystemToMatlab — mflowLink Embedded Coder, Tier 1.
//
// Walks the named Flow's subgraph in topological order and emits one
// MATLAB statement per block into a synthesised `matlab::Function`
// AST. The output feeds straight into the existing matlab_llvm
// `-emit-*` lanes. See `docs/embedded_coder_roadmap.md` §3 for the
// architecture rationale and §6 for the per-block lowering table.
//===----------------------------------------------------------------------===//

#include "matlab/Flowchart/SubsystemToMatlab.h"

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace matlab::flowchart {

namespace {

//===-----------------------------------------------------------------===//
// String / identifier helpers.
//===-----------------------------------------------------------------===//

// Replace any character that's not [A-Za-z0-9_] with `_`; prepend `b_`
// if the result would start with a digit. Idempotent on well-formed
// identifiers.
std::string sanitizeIdent(const std::string &S) {
  std::string Out;
  Out.reserve(S.size());
  for (char C : S) {
    if (std::isalnum((unsigned char)C) || C == '_') Out.push_back(C);
    else Out.push_back('_');
  }
  if (Out.empty() || std::isdigit((unsigned char)Out.front()))
    Out = "b_" + Out;
  return Out;
}

// Parse a port id of the shape `u3` / `y2` / `in1` / `out2` → integer.
// Returns -1 when the id doesn't follow the convention; the caller
// then falls back to the textual sort.
int parsePortIndex(const std::string &Id) {
  size_t I = 0;
  while (I < Id.size() && !std::isdigit((unsigned char)Id[I])) ++I;
  if (I >= Id.size()) return -1;
  try { return std::stoi(Id.substr(I)); } catch (...) { return -1; }
}

//===-----------------------------------------------------------------===//
// Tier-1 supported block kinds.
//===-----------------------------------------------------------------===//
const std::set<std::string> &tier1Kinds() {
  static const std::set<std::string> S = {
      "signal_constant", "signal_gain",         "signal_sum",
      "signal_product",  "signal_abs",          "signal_saturation",
      "signal_math_fcn", "signal_trig_fcn",     "signal_relop",
      "signal_logical",  "signal_compare_to_zero",
      "signal_compare_to_constant",
      "signal_mux",      "signal_demux",        "signal_reshape",
      "signal_switch",   "signal_multiport_switch",
      "signal_merge",
      // Tier 3 — stateful discrete blocks.  Each contributes one
      // scalar state slot to the emitted function's signature
      // (one extra arg + one extra return).  See the `isStatefulKind`
      // helper below for the actual list and `stateSlotName` for the
      // naming convention.
      "signal_unit_delay", "signal_zoh",
      "signal_discrete_integrator",
      // Tier 4 — continuous-time integrator, auto-discretised at
      // the subsystem's `Ts` (CLI --target-rate / block sample_time
      // / settings.solver.maxStep). Rejected with a sourced error
      // when SubsystemEmitOptions.RejectContinuous = true (HDL lane).
      "signal_integrator",
      // Tier-5g — continuous Transfer Function (1st-order
      // strictly-proper MVP only).  num at most order 0, den
      // order 1.  Auto-discretised via Forward Euler at the
      // subsystem's chosen Ts.  Higher-order TFs and Zero-Pole /
      // State-Space follow in Tier-5h.
      "signal_transfer_fcn",
      // Tier-5h — continuous Zero-Pole-Gain form. Real roots only
      // (zeros / poles as comma-separated reals + scalar gain).
      // Expanded to (num, den) polynomials via `resolveTFCoeffs`
      // and then routed through the same Forward Euler controllable
      // canonical state-space discretisation as signal_transfer_fcn.
      "signal_zero_pole",
      // Tier-5h — continuous transport delay (`delay` seconds).
      // Discretised as a length-N shift register where
      // N = round(delay / Ts). Each tap is one state slot;
      // output = oldest tap, new tap = current input.
      "signal_transport_delay",
      // Tier-5h — continuous state-space realisation. (A, B, C) as
      // matlab-matrix-literal strings, D = 0 (strictly proper).
      // Discretised via Forward Euler: x[k+1] = (I + Ts*A)*x[k] +
      // Ts*B*u[k]; y = C*x. Contributes N state slots (N = A's row
      // count).
      "signal_state_space",
      // #343 HDL — D flip-flop. One persistent register; the `clk` input
      // maps to the module clock (single-clock design), so it emits as
      // `always_ff @(posedge clk) Q <= D`. (signal_tff / signal_counter
      // simulate but are not yet SV-lowered.)
      "signal_dff",
      // Tier 5 — inline user MATLAB. The block's `params.function_body`
      // becomes a sibling local function in the same TU; the call
      // site emits as `<out> = <fn_name>(<inputs...>)`. SV emit
      // delegates synthesisability to the existing -check-synthesizable
      // pass over the user body.
      "signal_matlab_fcn",
      // MPC Toolbox Tier-3 §4.5/4.6 — MpcMove block.  Embedded
      // Coder emits the static-gain MPC approximation as a single
      // multiply-and-subtract expression `gain * (r - ym)`.  The
      // full QP-solving MPC requires linking runtime_mpc.cpp into
      // the generated artifact (Tier-3b carve-down).
      "signal_mpc_move",
      // Routing-only — emit nothing, but pass through the variable
      // name from the source.
      "signal_inport",   "signal_outport",
      // Sinks: drop entirely (scope/display/to_workspace/terminator
      // aren't part of the subsystem API).
      "signal_scope",    "signal_display",
      "signal_to_workspace", "signal_terminator",
  };
  return S;
}

bool isSinkKind(const std::string &K) {
  return K == "signal_scope" || K == "signal_display" ||
         K == "signal_to_workspace" || K == "signal_terminator";
}

// Tier 3 — stateful (loop-breaker) blocks. Each carries one state
// slot in the emitted function's signature: an extra scalar arg
// `s_<id>` (current state) and an extra scalar return
// `s_<id>_next` (state for the next tick). The state read happens
// at the start of the block's evaluation; the state-update lands
// in the next-state output.
//
// Tier 4 promotes `signal_integrator` (continuous-time `dx/dt = u`)
// into the stateful set: it gets auto-discretised to Forward Euler
// (or Backward Euler / Trapezoidal — see SubsystemEmitOptions) at
// the subsystem's chosen `Ts`. The lowering reports a sourced error
// if no rate can be resolved.
bool isStatefulKind(const std::string &K) {
  return K == "signal_unit_delay" ||
         K == "signal_zoh" ||
         K == "signal_discrete_integrator" ||
         K == "signal_integrator" ||
         // Tier-5g/h: continuous Transfer Function / Zero-Pole-Gain,
         // any order strictly proper. Discretised via Forward Euler
         // on the controllable canonical state-space realisation —
         // contributes `den_degree` state slots to the function
         // signature.
         K == "signal_transfer_fcn" ||
         K == "signal_zero_pole" ||
         // Tier-5h: transport delay — discretised as a length-N
         // shift register with N = round(delay/Ts).
         K == "signal_transport_delay" ||
         // Tier-5h: continuous state-space (A, B, C) with D=0.
         // Contributes N state slots (N = A's row count).
         K == "signal_state_space" ||
         // #343 HDL: clocked registers. The block's `clk` input maps to the
         // module clock (single-clock design), so a D flip-flop emits as a
         // one-element persistent register updated every clock — exactly the
         // unit_delay shape (`always_ff @(posedge clk) Q <= D`).
         K == "signal_dff";
}

// Tier-5i — does this block's output equation overwrite the
// state-read local? For these kinds the state-read hoist must
// write to a SEPARATE local (e.g. `x1_<id>`) so the next-state
// update — emitted at end-of-body, AFTER the output assignment —
// still sees the un-overwritten state value.
//
// Gated on `Method == "tustin"` because:
//
//   - Forward Euler `signal_transfer_fcn` / `signal_zero_pole` /
//     `signal_state_space` output equations have NO direct
//     feedthrough (y = b_0*x or C*x, depends on state only).
//     Their state-update uses `state[k]` not `y[k]`, so even
//     when the output overwrites OutVar, the state-update reads
//     `localFor` which under FE Order=1 == OutVar — yielding
//     `b_0*state` instead of `state`. **This is latent: FE TF
//     with b_0=1 (the existing tf_lowpass) is unaffected; b_0!=1
//     would be wrong.** Fixing that requires emitting the output
//     equation EARLY (right after state-read hoist) so consumers
//     reading the OutVar in a feedback loop see the correct
//     value. Deferred — no Tier ≤ 5h demo exercises b_0!=1
//     Order=1.
//
//   - Tustin always has direct feedthrough through the output:
//     y[k] = NumZ[0]*u[k] + v_1[k] (TF/ZP/SS) or
//     y[k] = (Ts/2)*u[k] + state[k] (integrator). The output
//     overwrites OutVar with a value that depends on u[k]. The
//     state-update needs the ORIGINAL state value, which is only
//     preserved by a separate LocalVar.
//
//     Tustin blocks placed in algebraic feedback loops (output
//     fed back to input without an intervening unit-delay) form a
//     combinational loop that this lowering can't break. Users
//     wanting a Tustin-discretised filter inside a feedback loop
//     should replace the manual Integrator+Sum subgraph with a
//     standalone `signal_transfer_fcn` and let Tustin
//     discretisation handle the math globally. Detection of such
//     loops is a follow-up — current code emits an uninitialised
//     OutVar read at runtime.
bool needsSeparateLocal(const std::string &Kind,
                        const std::string &Method) {
  if (Method != "tustin") return false;
  return Kind == "signal_transfer_fcn" ||
         Kind == "signal_zero_pole"    ||
         Kind == "signal_state_space"  ||
         Kind == "signal_integrator"   ||
         Kind == "signal_discrete_integrator";
}

// Tier-5g — parse a comma-separated coefficient list, highest order
// first.  Mirrors `parsePoly` in lib/Flowchart/MflowLinkSim.cpp.
std::vector<double> parsePoly(const std::string &S) {
  std::vector<double> Out;
  std::stringstream SS(S);
  std::string Tok;
  while (std::getline(SS, Tok, ',')) {
    size_t A = Tok.find_first_not_of(" \t");
    if (A == std::string::npos) continue;
    size_t Bp = Tok.find_last_not_of(" \t");
    try { Out.push_back(std::stod(Tok.substr(A, Bp - A + 1))); }
    catch (...) {}
  }
  return Out;
}

// Tier-5h — expand a list of real roots `r_k` into the coefficients
// of `Π (s - r_k)`, highest power first. Empty roots ⇒ {1} (the
// constant 1 polynomial). Mirrors `expandPoly` in MflowLinkSim.cpp;
// shared here so `signal_zero_pole` can reuse the TF discretisation
// path by first lifting (zeros, poles, gain) into (num, den).
std::vector<double> expandPoly(const std::vector<double> &Roots) {
  std::vector<double> P{1.0};
  for (double R : Roots) {
    std::vector<double> Q(P.size() + 1, 0.0);
    for (size_t I = 0; I < P.size(); ++I) {
      Q[I]     += P[I];
      Q[I + 1] += -R * P[I];
    }
    P = std::move(Q);
  }
  return P;
}

// Tier-5h — parse a comma/space/semicolon-separated matrix string
// like `"1 2; 3 4"` or `"[1, 2; 3, 4]"` into a row-major flat
// vector + dimensions. Mirrors the simulator's `parseMatrix`
// helper in `lib/Flowchart/MflowLinkSim.cpp`.
void parseMatrixStr(const std::string &S, std::vector<double> &Vals,
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
    // Treat commas as whitespace.
    for (auto &ch : Row) if (ch == ',') ch = ' ';
    std::stringstream CS(Row);
    std::string Tok;
    while (CS >> Tok) {
      try { Vals.push_back(std::stod(Tok)); ++C; }
      catch (...) {}
    }
    if (C > 0) {
      ++Rows;
      Cols = C;
    }
  }
}

// Tier-5i — generic polynomial helpers used by the Tustin (bilinear)
// substitution path. Coefficient convention matches `parsePoly` /
// `expandPoly`: highest power first, length = degree + 1.
std::vector<double> polyMul(const std::vector<double> &A,
                            const std::vector<double> &B) {
  if (A.empty() || B.empty()) return {};
  std::vector<double> C(A.size() + B.size() - 1, 0.0);
  for (size_t I = 0; I < A.size(); ++I)
    for (size_t J = 0; J < B.size(); ++J)
      C[I + J] += A[I] * B[J];
  return C;
}

std::vector<double> polyPow(const std::vector<double> &P, int K) {
  std::vector<double> R{1.0};
  for (int I = 0; I < K; ++I) R = polyMul(R, P);
  return R;
}

void polyAddInto(std::vector<double> &Acc,
                 const std::vector<double> &P) {
  if (Acc.size() < P.size())
    Acc.insert(Acc.begin(), P.size() - Acc.size(), 0.0);
  size_t Off = Acc.size() - P.size();
  for (size_t I = 0; I < P.size(); ++I) Acc[Off + I] += P[I];
}

// Tier-5i — Apply Tustin (bilinear) substitution s = (2/Ts)·(z-1)/(z+1)
// to a polynomial in s of degree m ≤ DegN, multiplying every term by
// (z+1)^(DegN - PowS) so the result is a single polynomial in z of
// degree DegN (highest power first, length DegN+1). Used for both
// numerator and denominator: pass the SAME DegN for both so the two
// share a common denominator clearing factor (z+1)^DegN.
std::vector<double> tustinSubst(const std::vector<double> &Ps, int DegN,
                                double Ts) {
  std::vector<double> R(DegN + 1, 0.0);
  std::vector<double> Z1{1.0, -1.0};  // z - 1
  std::vector<double> Z0{1.0, 1.0};   // z + 1
  int M = (int)Ps.size() - 1;
  double TwoOverTs = 2.0 / Ts;
  for (int K = 0; K <= M; ++K) {
    double Coef = Ps[K];
    if (Coef == 0.0) continue;
    int PowS = M - K;  // power of s for this term
    double Scale = Coef;
    for (int J = 0; J < PowS; ++J) Scale *= TwoOverTs;
    auto Term = polyMul(polyPow(Z1, PowS), polyPow(Z0, DegN - PowS));
    for (auto &X : Term) X *= Scale;
    polyAddInto(R, Term);
  }
  return R;
}

// Tier-5i — Tustin discretisation of a continuous SISO TF Num/Den.
// Returns the discrete numerator and denominator in z, both length
// (deg(Den) + 1), normalised so the leading denominator coefficient
// is 1. The discrete numerator is proper (degree = deg(Den)) so the
// realisation has a direct-feedthrough term — that's the defining
// feature of Tustin vs. Forward Euler. Caller must drive the result
// through a realisation that handles direct feedthrough (e.g. Direct
// Form II Transposed).
void tustinTF(const std::vector<double> &Num,
              const std::vector<double> &Den, double Ts,
              std::vector<double> &NumZ, std::vector<double> &DenZ) {
  int N = (int)Den.size() - 1;  // continuous denominator degree
  NumZ = tustinSubst(Num, N, Ts);
  DenZ = tustinSubst(Den, N, Ts);
  double Lead = DenZ.front();
  if (Lead == 0.0) Lead = 1.0;
  for (auto &X : NumZ) X /= Lead;
  for (auto &X : DenZ) X /= Lead;
}

// Tier-5l — Small dense matrix helpers shared by MIMO Tustin.
// Vectors are row-major flat std::vector<double>; dimensions are
// passed explicitly. Sized for the small N typical in control
// systems (≤ ~8); cubic Gauss-Jordan is fine at that scale.
void matEye(int N, std::vector<double> &Out) {
  Out.assign((size_t)N * N, 0.0);
  for (int I = 0; I < N; ++I) Out[(size_t)I * N + I] = 1.0;
}

void matAdd(const std::vector<double> &A, const std::vector<double> &B,
            int Rows, int Cols, std::vector<double> &Out) {
  Out.assign((size_t)Rows * Cols, 0.0);
  for (size_t I = 0; I < (size_t)Rows * Cols; ++I) Out[I] = A[I] + B[I];
}

void matSub(const std::vector<double> &A, const std::vector<double> &B,
            int Rows, int Cols, std::vector<double> &Out) {
  Out.assign((size_t)Rows * Cols, 0.0);
  for (size_t I = 0; I < (size_t)Rows * Cols; ++I) Out[I] = A[I] - B[I];
}

void matScaleInPlace(std::vector<double> &M, double S) {
  for (auto &X : M) X *= S;
}

// Row-major (Ar × Ac) · (Ac × Bc) → (Ar × Bc).
void matMulOuter(const std::vector<double> &A, int Ar, int Ac,
                 const std::vector<double> &B, int Bc,
                 std::vector<double> &Out) {
  Out.assign((size_t)Ar * Bc, 0.0);
  for (int I = 0; I < Ar; ++I)
    for (int J = 0; J < Bc; ++J) {
      double S = 0.0;
      for (int K = 0; K < Ac; ++K)
        S += A[(size_t)I * Ac + K] * B[(size_t)K * Bc + J];
      Out[(size_t)I * Bc + J] = S;
    }
}

// Gauss-Jordan inverse for small square matrices. Partial pivoting
// only. Returns false on singular input (any pivot below 1e-12 in
// absolute value).
bool matInverse(const std::vector<double> &In, int N,
                std::vector<double> &Out) {
  std::vector<double> A = In;
  matEye(N, Out);
  for (int K = 0; K < N; ++K) {
    int P = K;
    double Best = std::abs(A[(size_t)K * N + K]);
    for (int R = K + 1; R < N; ++R) {
      double V = std::abs(A[(size_t)R * N + K]);
      if (V > Best) { Best = V; P = R; }
    }
    if (Best < 1e-12) return false;
    if (P != K) {
      for (int J = 0; J < N; ++J) {
        std::swap(A[(size_t)K * N + J], A[(size_t)P * N + J]);
        std::swap(Out[(size_t)K * N + J], Out[(size_t)P * N + J]);
      }
    }
    double Pv = A[(size_t)K * N + K];
    for (int J = 0; J < N; ++J) {
      A[(size_t)K * N + J]   /= Pv;
      Out[(size_t)K * N + J] /= Pv;
    }
    for (int R = 0; R < N; ++R) {
      if (R == K) continue;
      double F = A[(size_t)R * N + K];
      if (F == 0.0) continue;
      for (int J = 0; J < N; ++J) {
        A[(size_t)R * N + J]   -= F * A[(size_t)K * N + J];
        Out[(size_t)R * N + J] -= F * Out[(size_t)K * N + J];
      }
    }
  }
  return true;
}

// Tier-5l — Tustin (bilinear) discretisation of continuous MIMO
// state-space `dx/dt = A x + B u, y = C x + D u` (input D omitted —
// caller knows D=0 at the source level).  Derivation:
//
//   (I - αA) x[k+1] = (I + αA) x[k] + α B (u[k+1] + u[k])
//
// with α = Ts/2.  Eliminating the u[k+1] dependence via the state
// transform z[k] = x[k] - α M B u[k] (where M = (I - αA)^-1) gives
// the canonical Tustin form:
//
//   Ad = M · (I + αA)
//   Bd = α · (I + Ad) · M · B          ← direct-feedthrough comes
//   Cd = C                              ← from the state transform
//   Dd = α · C · M · B                  ← y = Cd·z + Dd·u
//
// Returns false when (I - αA) is singular (numerically, |det| <
// 1e-12 after partial pivoting). Output matrix shapes: Ad N×N,
// Bd N×P, Cd Q×N, Dd Q×P.
bool tustinSS(const std::vector<double> &A, int N,
              const std::vector<double> &B, int P,
              const std::vector<double> &C, int Q, double Ts,
              std::vector<double> &Ad, std::vector<double> &Bd,
              std::vector<double> &Cd, std::vector<double> &Dd) {
  double Alpha = Ts / 2.0;
  std::vector<double> EyeN;
  matEye(N, EyeN);
  std::vector<double> AlphaA = A;
  matScaleInPlace(AlphaA, Alpha);
  std::vector<double> IminusAA;
  matSub(EyeN, AlphaA, N, N, IminusAA);
  std::vector<double> M;
  if (!matInverse(IminusAA, N, M)) return false;
  std::vector<double> IplusAA;
  matAdd(EyeN, AlphaA, N, N, IplusAA);
  // Ad = M (I+αA).
  matMulOuter(M, N, N, IplusAA, N, Ad);
  // Bd = α (I + Ad) M B.
  std::vector<double> IplusAd;
  matAdd(EyeN, Ad, N, N, IplusAd);
  std::vector<double> Tmp;
  matMulOuter(IplusAd, N, N, M, N, Tmp);          // (I+Ad)·M
  matMulOuter(Tmp, N, N, B, P, Bd);               // ·B
  matScaleInPlace(Bd, Alpha);
  // Cd = C  (unchanged in the transformed basis).
  Cd = C;
  // Dd = α C M B.
  std::vector<double> CM;
  matMulOuter(C, Q, N, M, N, CM);
  matMulOuter(CM, Q, N, B, P, Dd);
  matScaleInPlace(Dd, Alpha);
  return true;
}

// Tier-5i — Faddeev-LeVerrier conversion of a SISO continuous state-
// space (A, B, C) into a transfer function (Num, Den). A is N×N
// row-major, B is N×1, C is 1×N. Returns Num of length N (degree
// N-1, lowest leading coef is the s^{N-1} term) and Den of length
// N+1 (monic in s^N). Both highest-power first.
//
// Algorithm (with monic numerator/denominator convention):
//   M_0 = I,  p_0 = 1
//   M_1 = A * M_0 + p_1 * I,  p_1 = -tr(A * M_0) / 1
//   M_k = A * M_{k-1} + p_k * I,  p_k = -tr(A * M_{k-1}) / k
//   det(sI - A) = s^N + p_1 s^{N-1} + ... + p_N
//   adj(sI - A) = s^{N-1} M_0 + s^{N-2} M_1 + ... + M_{N-1}
// SISO numerator coefficient of s^{N-1-k} is (C * M_k * B).
void ssToTFSiso(const std::vector<double> &A,
                const std::vector<double> &B,
                const std::vector<double> &C, int N,
                std::vector<double> &Num,
                std::vector<double> &Den) {
  auto matMul = [&](const std::vector<double> &X,
                     const std::vector<double> &Y, int Xr, int Xc,
                     int Yc) {
    std::vector<double> R(Xr * Yc, 0.0);
    for (int I = 0; I < Xr; ++I)
      for (int J = 0; J < Yc; ++J) {
        double S = 0.0;
        for (int K = 0; K < Xc; ++K) S += X[I * Xc + K] * Y[K * Yc + J];
        R[I * Yc + J] = S;
      }
    return R;
  };
  std::vector<double> Mk(N * N, 0.0);
  for (int I = 0; I < N; ++I) Mk[I * N + I] = 1.0;  // M_0 = I
  Den.assign(N + 1, 0.0);
  Den[0] = 1.0;  // s^N coefficient (monic)
  Num.assign(N, 0.0);
  // s^{N-1} coefficient of Num = C * M_0 * B.
  {
    auto CB = matMul(C, matMul(Mk, B, N, N, 1), 1, N, 1);
    Num[0] = CB[0];
  }
  for (int K = 1; K <= N; ++K) {
    auto AMk = matMul(A, Mk, N, N, N);
    double Trace = 0.0;
    for (int I = 0; I < N; ++I) Trace += AMk[I * N + I];
    double Pk = -Trace / (double)K;
    Den[K] = Pk;
    // M_k = A * M_{k-1} + p_k * I
    Mk = AMk;
    for (int I = 0; I < N; ++I) Mk[I * N + I] += Pk;
    if (K < N) {
      auto CMB = matMul(C, matMul(Mk, B, N, N, 1), 1, N, 1);
      Num[K] = CMB[0];
    }
  }
}

// Tier-5h — resolve a signal_transfer_fcn or signal_zero_pole block
// to a (num, den) polynomial pair. For zero-pole, expands the
// real-root lists via `expandPoly` and scales num by `gain`. For
// transfer functions, returns the raw num/den parsed straight from
// params. Returns false on any malformed input.
bool resolveTFCoeffs(const Node &N,
                     std::vector<double> &Num,
                     std::vector<double> &Den) {
  if (N.Kind == "signal_transfer_fcn") {
    auto It = N.Params.find("num");
    Num = parsePoly(It == N.Params.end() ? "1" : It->second);
    It  = N.Params.find("den");
    Den = parsePoly(It == N.Params.end() ? "1" : It->second);
    return !Den.empty();
  }
  if (N.Kind == "signal_zero_pole") {
    auto getStr = [&](const char *Key) -> std::string {
      auto It = N.Params.find(Key);
      return It == N.Params.end() ? std::string{} : It->second;
    };
    auto Zeros = parsePoly(getStr("zeros"));
    auto Poles = parsePoly(getStr("poles"));
    double Gain = 1.0;
    auto It = N.Params.find("gain");
    if (It != N.Params.end()) {
      try { Gain = std::stod(It->second); } catch (...) {}
    }
    Num = expandPoly(Zeros);
    for (auto &C : Num) C *= Gain;
    Den = expandPoly(Poles);
    return !Den.empty();
  }
  return false;
}

// Tier 4 — initial-condition param lookup per stateful kind.
// signal_integrator / signal_discrete_integrator carry an
// `initialCondition` (alt-spelling `initial_condition`). Unit Delay
// uses `initialCondition` too; ZOH uses `initialOutput`. Default 0.0.
double initialStateOf(const Node &N) {
  auto get = [&](const char *Key) -> const std::string * {
    auto It = N.Params.find(Key);
    return It == N.Params.end() ? nullptr : &It->second;
  };
  const std::string *S = get("initialCondition");
  if (!S) S = get("initial_condition");
  if (!S) S = get("initialOutput");
  if (!S) S = get("initial_value");
  if (!S) return 0.0;
  try { return std::stod(*S); } catch (...) { return 0.0; }
}

//===-----------------------------------------------------------------===//
// Per-edge lookup: for a given (Block, PortName), find the
// upstream Node id + its output variable name.
//===-----------------------------------------------------------------===//
struct EdgeIndex {
  // map: (toNodeId, toPortId) -> (fromNodeId, fromPortId)
  std::map<std::pair<std::string, std::string>,
           std::pair<std::string, std::string>> Map;
};
EdgeIndex buildEdgeIndex(const Flow &F) {
  EdgeIndex EI;
  for (const auto &E : F.Edges) {
    EI.Map[{E.To.Node, E.To.Port}] = {E.From.Node, E.From.Port};
  }
  return EI;
}

//===-----------------------------------------------------------------===//
// Topological sort over the Flow's edges. Skips inport / outport / sink
// kinds — they're handled separately. Returns the topo order of the
// internal blocks (those that emit a MATLAB statement).
//===-----------------------------------------------------------------===//
std::vector<const Node *>
toposortInternals(const Flow &F, DiagnosticEngine &Diag,
                  const std::string &SubName) {
  std::unordered_map<std::string, const Node *> NodeById;
  for (const auto &N : F.Nodes) NodeById[N.Id] = &N;

  // Internal nodes = neither inport nor outport nor sink.
  std::vector<const Node *> Internal;
  for (const auto &N : F.Nodes) {
    if (N.Kind == "signal_inport" || N.Kind == "signal_outport" ||
        isSinkKind(N.Kind))
      continue;
    Internal.push_back(&N);
  }

  // Build adjacency: in-edges per node, ignoring edges from sinks
  // (which won't appear in practice) and edges to outports / sinks
  // (those are terminal consumers).
  std::unordered_map<std::string, std::vector<std::string>> InEdges;
  std::unordered_map<std::string, int> InDegree;
  std::unordered_set<std::string> InternalSet;
  for (auto *N : Internal) InternalSet.insert(N->Id);

  // Loop-breaker rule: stateful blocks (Tier 3 + Tier 4 stateful set)
  // emit their CURRENT state as the output — that doesn't depend on
  // the in-this-tick input. Drop their outgoing edges from the topo
  // graph so feedback paths through `signal_integrator` /
  // `signal_unit_delay` / `signal_zoh` / `signal_discrete_integrator`
  // resolve cleanly. (The next-state update is computed *after* every
  // dependent block has consumed the current-state read, so the
  // back-edge is legitimately broken.)
  auto isLoopBreaker = [&](const std::string &NodeId) {
    auto It = std::find_if(F.Nodes.begin(), F.Nodes.end(),
                            [&](const Node &N) { return N.Id == NodeId; });
    if (It == F.Nodes.end()) return false;
    return isStatefulKind(It->Kind);
  };
  for (const auto &E : F.Edges) {
    if (!InternalSet.count(E.To.Node)) continue;
    if (!InternalSet.count(E.From.Node)) continue; // inport / sink-edge
    if (isLoopBreaker(E.From.Node)) continue;       // §6.3 loop-breaker
    InEdges[E.To.Node].push_back(E.From.Node);
    InDegree[E.To.Node]++;
  }

  // Kahn's algorithm. Initialise the queue with every internal node
  // that has zero in-edges from another internal node.
  std::vector<std::string> Ready;
  for (auto *N : Internal) {
    if (InDegree.find(N->Id) == InDegree.end()) Ready.push_back(N->Id);
  }
  // Stable order by node id so the emit output is deterministic.
  std::sort(Ready.begin(), Ready.end());

  std::vector<const Node *> Out;
  while (!Ready.empty()) {
    auto Id = Ready.front();
    Ready.erase(Ready.begin());
    Out.push_back(NodeById[Id]);
    // Find every internal node whose in-edges include Id and decrement.
    for (auto *N : Internal) {
      auto It = InEdges.find(N->Id);
      if (It == InEdges.end()) continue;
      auto &Sources = It->second;
      auto SIt = std::find(Sources.begin(), Sources.end(), Id);
      if (SIt == Sources.end()) continue;
      Sources.erase(SIt);
      if (--InDegree[N->Id] == 0) {
        Ready.push_back(N->Id);
        std::sort(Ready.begin(), Ready.end());
      }
    }
  }
  if (Out.size() != Internal.size()) {
    Diag.error(F.Loc, "subsystem \"" + SubName +
                          "\": cycle in subgraph — embedded-coder Tier-1 "
                          "does not solve algebraic loops");
    return {};
  }
  return Out;
}

//===-----------------------------------------------------------------===//
// AST builders — small helpers that wrap ASTContext::make.
//===-----------------------------------------------------------------===//
struct ASTBuilder {
  ASTContext &Ctx;
  // Tier 5 — when WrapFi is true, `lit(V)` wraps numeric literals in
  // `fi(V, signed, W, F)` so the SV pipeline sees concrete fixed-
  // point constants instead of f64.  Software-target emit leaves
  // the literal as a bare FPLiteral.
  bool WrapFi = false;
  bool FiSigned = true;
  int FiWidth = 32;
  int FiFrac = 16;

  NameExpr *name(const std::string &S) {
    auto *N = Ctx.make<NameExpr>();
    N->Name = Ctx.intern(S);
    return N;
  }
  FPLiteral *number(double V) {
    auto *F = Ctx.make<FPLiteral>();
    std::ostringstream OS;
    OS.precision(17);
    OS << V;
    F->Text = Ctx.intern(OS.str());
    return F;
  }
  IntegerLiteral *integer(int V) {
    auto *I = Ctx.make<IntegerLiteral>();
    I->Text = Ctx.intern(std::to_string(V));
    return I;
  }
  // Numeric literal honouring WrapFi — use this for ALL constant
  // operands inside lowerBlock's per-kind dispatch so a single flip
  // of WrapFi switches the entire emitter between software (raw
  // f64) and HDL (fi-wrapped) modes.
  Expr *lit(double V) {
    if (!WrapFi) return number(V);
    auto *Call = Ctx.make<CallOrIndex>();
    Call->Callee = name("fi");
    Call->Args.push_back(number(V));
    Call->Args.push_back(integer(FiSigned ? 1 : 0));
    Call->Args.push_back(integer(FiWidth));
    Call->Args.push_back(integer(FiFrac));
    return Call;
  }
  BinaryOpExpr *bin(BinOp Op, Expr *L, Expr *R) {
    auto *B = Ctx.make<BinaryOpExpr>();
    B->Op = Op; B->LHS = L; B->RHS = R;
    return B;
  }
  // Tier-5k — Fi-aware multiplication. The fi-multiplication of
  // two Q<W>.<F> values yields a Q*.{2F} natural product. Without
  // an explicit cast, Sema widens the inferred FL through the
  // expression tree and downstream slots inherit the wider spec,
  // so the SV pipeline lowers `fi(K) .* x` to `(x << log2(K_raw))`
  // without the `>>> F` normalising shift. Wrapping the result in
  // `fi(prod, S, W, F)` lets Sema infer the outer expression as
  // Q<W>.<F>; the AST → MIR lowering then emits a clamp-style
  // `matlab.fi.cast` that LowerFixedPoint translates into the
  // right shift. Software targets emit plain `.*` on f64 (the
  // Sema-side fi type doesn't change the IEEE arithmetic the
  // emit-* lanes produce).
  Expr *fiMul(Expr *L, Expr *R) {
    Expr *Prod = bin(BinOp::ElemMul, L, R);
    if (!WrapFi) return Prod;
    auto *Call = Ctx.make<CallOrIndex>();
    Call->Callee = name("fi");
    Call->Args.push_back(Prod);
    Call->Args.push_back(integer(FiSigned ? 1 : 0));
    Call->Args.push_back(integer(FiWidth));
    Call->Args.push_back(integer(FiFrac));
    return Call;
  }
  UnaryOpExpr *unary(UnOp Op, Expr *V) {
    auto *U = Ctx.make<UnaryOpExpr>();
    U->Op = Op; U->Operand = V;
    return U;
  }
  CallOrIndex *call(const std::string &FnName, std::vector<Expr *> Args) {
    auto *C = Ctx.make<CallOrIndex>();
    C->Callee = name(FnName);
    C->Args = std::move(Args);
    return C;
  }
  AssignStmt *assign(const std::string &Var, Expr *RHS) {
    auto *A = Ctx.make<AssignStmt>();
    A->LHS.push_back(name(Var));
    A->RHS = RHS;
    A->Suppressed = true;
    return A;
  }
};

//===-----------------------------------------------------------------===//
// Per-block lowering — the heart of Tier 1.
//
// Returns the AssignStmt that defines `<varName> = <expr>` for this
// block. The caller knows `varName` (the canonical output variable
// for the block) and `inputExprs` (one Expr per input port, sorted
// by port id). Returns nullptr if the block kind isn't supported.
//===-----------------------------------------------------------------===//
const std::string *paramS(const Node &N, const std::string &K) {
  return N.getParam(K);
}
double paramD(const Node &N, const std::string &K, double Def) {
  auto *S = paramS(N, K);
  if (!S) return Def;
  try { return std::stod(*S); } catch (...) { return Def; }
}
std::string paramSTR(const Node &N, const std::string &K,
                     const std::string &Def) {
  auto *S = paramS(N, K);
  return S ? *S : Def;
}

Stmt *lowerBlock(const Node &N, const std::string &OutVar,
                 const std::vector<Expr *> &Ins,
                 const std::vector<std::string> &InPortIds,
                 ASTBuilder &B, DiagnosticEngine &Diag) {
  const auto &K = N.Kind;

  auto get = [&](size_t I) -> Expr * {
    return I < Ins.size() ? Ins[I] : static_cast<Expr *>(B.lit(0.0));
  };

  if (K == "signal_constant") {
    // c = <value>;   (value can be scalar or matrix literal)
    auto V = paramSTR(N, "value", "0");
    // Vector / matrix literal: pass through as a MatrixLiteral. For
    // Tier 1 we parse only the scalar case; matrix literals route to
    // `signal_constant` via signal_reshape downstream.
    if (V.find('[') == std::string::npos) {
      double D = 0.0;
      try { D = std::stod(V); } catch (...) {}
      return B.assign(OutVar, B.lit(D));
    }
    // Strip the brackets, parse row-major numbers, build a MatrixLiteral.
    auto L = V.find('['), R = V.rfind(']');
    std::string Inner = V.substr(L + 1, R - L - 1);
    auto *ML = B.Ctx.make<MatrixLiteral>();
    std::string Tok;
    std::vector<Expr *> Row;
    auto endTok = [&]() {
      if (Tok.empty()) return;
      try { Row.push_back(B.lit(std::stod(Tok))); }
      catch (...) { Row.push_back(B.lit(0.0)); }
      Tok.clear();
    };
    auto endRow = [&]() {
      if (!Row.empty()) ML->Rows.push_back(std::move(Row));
      Row.clear();
    };
    for (size_t I = 0; I <= Inner.size(); ++I) {
      char C = I < Inner.size() ? Inner[I] : ';';
      if (C == ';') { endTok(); endRow(); }
      else if (C == ',' || C == ' ' || C == '\t') { endTok(); }
      else Tok.push_back(C);
    }
    return B.assign(OutVar, ML);
  }
  if (K == "signal_gain") {
    double Gain = paramD(N, "gain", 1.0);
    // y = K .* u — fiMul wraps the result in an explicit fi cast
    // so the SV pipeline applies the Q<W>.<F> normalising shift.
    auto *G = B.lit(Gain);
    auto *U = get(0);
    return B.assign(OutVar, B.fiMul(G, U));
  }
  if (K == "signal_mpc_move") {
    // MPC Toolbox Tier-3 §4.5/4.6 — emit the static-gain
    // approximation `u = gain * (r - ym)`.  Input ports are (ym, r)
    // in declaration order; if `r` is unconnected, fall back to
    // the `r_default` parameter.  Same simulator-side semantics.
    double Gain = paramD(N, "gain", 1.0);
    auto *G = B.lit(Gain);
    auto *Ym = get(0);
    Expr *Rr = (Ins.size() >= 2) ? get(1)
                                  : B.lit(paramD(N, "r_default", 0.0));
    auto *Diff = B.bin(BinOp::Sub, Rr, Ym);
    return B.assign(OutVar, B.fiMul(G, Diff));
  }
  if (K == "signal_sum") {
    // signs string like "+-+"; default = all '+' to match the input
    // count.
    auto Signs = paramSTR(N, "signs", std::string(Ins.size(), '+'));
    while (Signs.size() < Ins.size()) Signs.push_back('+');
    Expr *Acc = nullptr;
    for (size_t I = 0; I < Ins.size(); ++I) {
      Expr *T = Ins[I];
      bool Neg = (I < Signs.size() && Signs[I] == '-');
      if (!Acc) {
        Acc = Neg ? static_cast<Expr *>(B.unary(UnOp::Minus, T)) : T;
      } else {
        Acc = B.bin(Neg ? BinOp::Sub : BinOp::Add, Acc, T);
      }
    }
    if (!Acc) Acc = B.lit(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_product") {
    Expr *Acc = nullptr;
    for (Expr *T : Ins) {
      Acc = Acc ? B.fiMul(Acc, T) : T;
    }
    if (!Acc) Acc = B.lit(1.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_abs") {
    return B.assign(OutVar, B.call("abs", {get(0)}));
  }
  if (K == "signal_saturation") {
    double Lo = paramD(N, "lowerLimit", -1.0);
    double Hi = paramD(N, "upperLimit",  1.0);
    auto *U = get(0);
    // Tier-5d — HDL mode emits the if/elseif/else form. The pure-
    // arith form below (used for software targets) routes through
    // `bool * fi` multiplications that the SV synthcheck rejects;
    // an explicit if/elseif/else compiles to a clean 3-way mux at
    // synth time.
    if (B.WrapFi) {
      // Simple if/elseif/else form. The rails come pre-wrapped as
      // `fi(Hi, S, W, F)` / `fi(Lo, S, W, F)` via B.lit; wrap the
      // else-branch passthrough in a matching `fi(U, S, W, F)`
      // cast so Sema joins all three stores at a single Q<W>.<F>
      // spec instead of `any` (which trips the AST → MIR codegen
      // into a malformed `fi(none, ...)` constructor cast at the
      // outport binding). Tier-5f's `UnifyMixedWidthStores` MLIR
      // pass takes care of the integer-width mismatch at LowerFi
      // time; this AST-level wrap takes care of the Sema-level
      // type-join mismatch.
      auto *IfStmt = B.Ctx.make<class IfStmt>();
      IfStmt->Cond = B.bin(BinOp::Gt, U, B.lit(Hi));
      IfStmt->Then = B.Ctx.make<Block>();
      IfStmt->Then->Stmts.push_back(B.assign(OutVar, B.lit(Hi)));
      ElseIf EI;
      EI.Cond = B.bin(BinOp::Lt, U, B.lit(Lo));
      EI.Body = B.Ctx.make<Block>();
      EI.Body->Stmts.push_back(B.assign(OutVar, B.lit(Lo)));
      IfStmt->Elseifs.push_back(EI);
      IfStmt->Else = B.Ctx.make<Block>();
      // Wrap U so the passthrough has the same Q<W>.<F> spec as
      // the rails (Sema joins all three stores cleanly).
      auto *FiU = B.call("fi",
                          {U,
                           B.integer(B.FiSigned ? 1 : 0),
                           B.integer(B.FiWidth),
                           B.integer(B.FiFrac)});
      IfStmt->Else->Stmts.push_back(B.assign(OutVar, FiU));
      return IfStmt;
    }
    // Pure-arith form for software targets:
    //   y = u + (Hi - u) * (u > Hi) + (Lo - u) * (u < Lo)
    // Two correction terms; exactly one fires outside the rails.
    auto *DHi  = B.bin(BinOp::Sub, B.lit(Hi), U);
    auto *GtHi = B.bin(BinOp::Gt,  U, B.lit(Hi));
    auto *CHi  = B.bin(BinOp::ElemMul, DHi, GtHi);
    auto *DLo  = B.bin(BinOp::Sub, B.lit(Lo), U);
    auto *LtLo = B.bin(BinOp::Lt,  U, B.lit(Lo));
    auto *CLo  = B.bin(BinOp::ElemMul, DLo, LtLo);
    auto *S1   = B.bin(BinOp::Add, U, CHi);
    auto *Sat  = B.bin(BinOp::Add, S1, CLo);
    return B.assign(OutVar, Sat);
  }
  if (K == "signal_math_fcn" || K == "signal_trig_fcn") {
    // `function` param picks the builtin: e.g. "sin", "cos", "exp",
    // "log", "sqrt", "pow"…
    auto Fn = paramSTR(N, "function", K == "signal_trig_fcn" ? "sin" : "exp");
    std::vector<Expr *> Args;
    for (Expr *T : Ins) Args.push_back(T);
    if (Args.empty()) Args.push_back(B.lit(0.0));
    return B.assign(OutVar, B.call(Fn, std::move(Args)));
  }
  if (K == "signal_relop") {
    auto Op = paramSTR(N, "op", "==");
    BinOp BO = BinOp::Eq;
    if      (Op == "==") BO = BinOp::Eq;
    else if (Op == "~=" || Op == "!=") BO = BinOp::Ne;
    else if (Op == "<")  BO = BinOp::Lt;
    else if (Op == "<=") BO = BinOp::Le;
    else if (Op == ">")  BO = BinOp::Gt;
    else if (Op == ">=") BO = BinOp::Ge;
    return B.assign(OutVar, B.bin(BO, get(0), get(1)));
  }
  if (K == "signal_logical") {
    auto Op = paramSTR(N, "op", "and");
    Expr *Acc = nullptr;
    for (Expr *T : Ins) Acc = Acc ? B.bin(Op == "or"
                                              ? BinOp::Or
                                              : BinOp::And, Acc, T)
                                  : T;
    if (!Acc) Acc = B.lit(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_compare_to_zero") {
    auto Op = paramSTR(N, "op", ">");
    BinOp BO = BinOp::Gt;
    if      (Op == "==") BO = BinOp::Eq;
    else if (Op == "~=" || Op == "!=") BO = BinOp::Ne;
    else if (Op == "<")  BO = BinOp::Lt;
    else if (Op == "<=") BO = BinOp::Le;
    else if (Op == ">")  BO = BinOp::Gt;
    else if (Op == ">=") BO = BinOp::Ge;
    return B.assign(OutVar, B.bin(BO, get(0), B.lit(0.0)));
  }
  if (K == "signal_compare_to_constant") {
    auto Op = paramSTR(N, "op", ">");
    double C = paramD(N, "constant", 0.0);
    BinOp BO = BinOp::Gt;
    if      (Op == "==") BO = BinOp::Eq;
    else if (Op == "~=" || Op == "!=") BO = BinOp::Ne;
    else if (Op == "<")  BO = BinOp::Lt;
    else if (Op == "<=") BO = BinOp::Le;
    else if (Op == ">")  BO = BinOp::Gt;
    else if (Op == ">=") BO = BinOp::Ge;
    return B.assign(OutVar, B.bin(BO, get(0), B.lit(C)));
  }
  if (K == "signal_mux") {
    // y = [in1, in2, ...]
    auto *ML = B.Ctx.make<MatrixLiteral>();
    std::vector<Expr *> Row;
    for (Expr *T : Ins) Row.push_back(T);
    ML->Rows.push_back(std::move(Row));
    return B.assign(OutVar, ML);
  }
  if (K == "signal_reshape") {
    int R = (int)paramD(N, "rows", 0.0);
    int C = (int)paramD(N, "cols", 0.0);
    if (R <= 0 || C <= 0) {
      // No shape declared — fall back to passthrough.
      return B.assign(OutVar, get(0));
    }
    return B.assign(OutVar,
                    B.call("reshape", {get(0), B.integer(R), B.integer(C)}));
  }
  if (K == "signal_demux") {
    // Tier-C semantics: passthrough first input.
    return B.assign(OutVar, get(0));
  }
  if (K == "signal_switch") {
    // Natural form `y = (ctrl > th)*in1 + (ctrl <= th)*in3` makes
    // the static -emit-* pipeline mis-infer the function's return
    // type as `bool` (the comparison's type leaks through the
    // multiplication when neither operand is a concrete f64
    // literal — saturation works because `Hi - u` anchors the
    // expression to `double` first). Anchor with a `0.0 +` so the
    // top-level Add is `double + (bool*double + bool*double)`:
    //
    //   y = 0.0 + (ctrl > threshold)*in1 + (ctrl <= threshold)*in3
    double Th = paramD(N, "threshold", 0.0);
    auto *Gt = B.bin(BinOp::Gt, get(1), B.lit(Th));
    auto *Le = B.bin(BinOp::Le, get(1), B.lit(Th));
    auto *T  = B.bin(BinOp::ElemMul, Gt, get(0));
    auto *F  = B.bin(BinOp::ElemMul, Le, get(2));
    auto *Sum = B.bin(BinOp::Add, T, F);
    auto *Anchored = B.bin(BinOp::Add, B.lit(0.0), Sum);
    return B.assign(OutVar, Anchored);
  }
  if (K == "signal_multiport_switch") {
    // y = data(in1) where in1 is the 1-based selector. For codegen,
    // emit a chained ternary via call to a synthetic `ms_switch` —
    // simpler: emit a multiport_switch as a sequence of if-else, but
    // the AST emit lanes don't lower IfStmt as expressions. For
    // Tier 1, restrict to the 2-input (selector + one data) form and
    // emit `data` (passthrough).
    return B.assign(OutVar, get(1));
  }
  if (K == "signal_merge") {
    // First non-zero input wins. Tier-1 simplification: emit a sum
    // of all inputs (works when only one is active at a time).
    Expr *Acc = nullptr;
    for (Expr *T : Ins) Acc = Acc ? B.bin(BinOp::Add, Acc, T) : T;
    if (!Acc) Acc = B.lit(0.0);
    return B.assign(OutVar, Acc);
  }
  if (K == "signal_matlab_fcn") {
    // Tier 5 — call into the user's inline MATLAB. The
    // `params.function_body` is parsed at the end of the main
    // lowering (so all matlab_fcn helpers are sibling functions in
    // the same TU); here we just emit the call site. Helper name:
    // `<userFnName>_<sanitizedBlockId>` — guarantees uniqueness if
    // two blocks happen to have user functions with the same name.
    auto *FB = N.getParam("function_body");
    if (!FB) {
      Diag.error(N.Loc, "signal_matlab_fcn \"" + N.Id +
                            "\": missing `params.function_body`");
      return nullptr;
    }
    // Extract function name (first identifier before `(`).
    std::string SourceTxt = *FB;
    size_t Pos = SourceTxt.find("function");
    size_t Eq  = SourceTxt.find('=', Pos);
    size_t Lp  = SourceTxt.find('(', Pos);
    std::string FnName;
    if (Pos != std::string::npos && Lp != std::string::npos) {
      size_t Start = (Eq != std::string::npos && Eq < Lp)
                         ? Eq + 1 : Pos + sizeof("function") - 1;
      while (Start < Lp &&
             (SourceTxt[Start] == ' ' || SourceTxt[Start] == '\t'))
        ++Start;
      size_t End = Lp;
      while (End > Start &&
             (SourceTxt[End - 1] == ' ' || SourceTxt[End - 1] == '\t'))
        --End;
      FnName = SourceTxt.substr(Start, End - Start);
    }
    if (FnName.empty()) {
      Diag.error(N.Loc, "signal_matlab_fcn \"" + N.Id +
                            "\": could not extract function name from "
                            "`params.function_body`");
      return nullptr;
    }
    // Renamed helper: <user_name>_<block_id>.
    std::string Helper = FnName + "_" + sanitizeIdent(N.Id);
    std::vector<Expr *> Args;
    for (Expr *T : Ins) Args.push_back(T);
    return B.assign(OutVar, B.call(Helper, std::move(Args)));
  }
  // Tier-1 doesn't cover this kind.
  Diag.error(N.Loc, "embedded coder Tier-1 doesn't yet support "
                    "`" + K + "` — see "
                    "docs/embedded_coder_roadmap.md §6 for the "
                    "block-coverage table");
  return nullptr;
}

//===-----------------------------------------------------------------===//
// Port collection — find inports / outports + sort by port index.
//===-----------------------------------------------------------------===//
struct PortInfo {
  const Node *N;
  int Index;   // 1-based from `u<k>` / `y<k>`, else order of appearance
  std::string Var;
};

std::vector<PortInfo> collectPorts(const Flow &F, const std::string &Kind) {
  std::vector<PortInfo> Out;
  for (const auto &N : F.Nodes) {
    if (N.Kind != Kind) continue;
    int Idx = parsePortIndex(N.Id);
    if (Idx < 0) Idx = (int)Out.size() + 1;
    Out.push_back({&N, Idx, ""});
  }
  std::sort(Out.begin(), Out.end(),
            [](const PortInfo &A, const PortInfo &B) {
              if (A.Index != B.Index) return A.Index < B.Index;
              return A.N->Id < B.N->Id;
            });
  // Default variable name = the node id sanitised; the caller renames
  // inports to `u<k>` / outports to `y<k>` for readable output.
  for (size_t I = 0; I < Out.size(); ++I) {
    if (Kind == "signal_inport")  Out[I].Var = "u" + std::to_string(I + 1);
    else if (Kind == "signal_outport") Out[I].Var = "y" + std::to_string(I + 1);
    else Out[I].Var = sanitizeIdent(Out[I].N->Id);
  }
  return Out;
}

} // namespace

//===----------------------------------------------------------------------===//
// Tier-6 — nested subsystem context.
//
// Carried across recursive `lowerSubsystemImpl` calls so each
// `signal_subsystem` block resolves its `data.flow_id` to the inner
// subsystem's emitted function, with caching so the same flow
// referenced by multiple outer blocks emits ONE function. The
// `Pending` list is flushed into the TU by `buildSubsystemTU` in
// emission order.
//===----------------------------------------------------------------------===//

struct NestedCtx {
  // flow_id → (function, metadata).  The metadata exposes the inner
  // subsystem's input / output ports and state slot count to the
  // outer site for state-plumbing and call-site emission.
  std::map<std::string, std::pair<matlab::Function *, SubsystemMeta>>
      ByFlowId;
  // Functions to add to the final TU, in the order they were first
  // emitted (innermost subsystems precede their enclosers).
  std::vector<matlab::Function *> Pending;
  // Active flow stack used to detect mutually-recursive subsystem
  // graphs (would otherwise cause unbounded recursion).
  std::vector<std::string> Stack;
};

const Flow *findFlowById(const FlowDoc &Doc, const std::string &FlowId) {
  for (auto &F : Doc.Flows)
    if (F.Id == FlowId) return &F;
  return nullptr;
}

// Forward declaration — the actual subsystem lowering, parameterised
// over the nested-emit context. The public `lowerSubsystemToMatlab`
// is a thin wrapper that constructs a context, calls this impl, and
// discards the pending list (for the single-subsystem case where
// the caller doesn't want nested children in their TU).
matlab::Function *lowerSubsystemImpl(
    const FlowDoc &Doc, const std::string &SubsystemName,
    matlab::ASTContext &AST, matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts, NestedCtx &Ctx);

//===----------------------------------------------------------------------===//
// Public entry points.
//===----------------------------------------------------------------------===//

matlab::Function *lowerSubsystemToMatlab(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts) {
  NestedCtx Ctx;
  return lowerSubsystemImpl(Doc, SubsystemName, AST, Diag, Opts, Ctx);
}

matlab::Function *lowerSubsystemImpl(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts,
    NestedCtx &Ctx) {
  const Flow *Sub = Doc.findFlow(SubsystemName);
  if (!Sub) {
    Diag.error(SourceLocation{}, "subsystem \"" + SubsystemName +
                                     "\" not found in `.mflow` file");
    return nullptr;
  }
  // Tier-6 — recursion guard for mutually-referential nested
  // subsystems. The signal-flow `signal_subsystem` model normally
  // requires DAG shape so this fires only when the user wires a
  // cycle by hand.
  for (const auto &S : Ctx.Stack) {
    if (S == Sub->Id) {
      Diag.error(Sub->Loc,
                 "subsystem \"" + SubsystemName +
                     "\" references itself recursively via "
                     "`signal_subsystem` — embedded coder needs a "
                     "DAG of subsystems");
      return nullptr;
    }
  }
  Ctx.Stack.push_back(Sub->Id);
  // Pop on every return.
  struct StackPopper {
    NestedCtx *C;
    ~StackPopper() { if (C) C->Stack.pop_back(); }
  } Popper{&Ctx};
  (void)Popper;
  auto Inports = collectPorts(*Sub, "signal_inport");
  auto Outports = collectPorts(*Sub, "signal_outport");
  if (Inports.empty() && Outports.empty()) {
    Diag.error(Sub->Loc, "subsystem \"" + SubsystemName +
                             "\" has no signal_inport / signal_outport "
                             "boundary — top-level flows without ports "
                             "go through `-emit-mflowlink-cpp` instead");
    return nullptr;
  }

  ASTBuilder B{AST};
  // Tier 5 — flip the ASTBuilder into HDL mode so per-block numeric
  // literals emit as `fi(V, ...)` calls. SV synth-check rejects f64
  // operands; the static SV pipeline's `runLowerFixedPoint` resolves
  // fi-wrapped constants into the right integer width.
  if (Opts.StateAsPersistent) {
    B.WrapFi   = true;
    B.FiSigned = Opts.FiDefault.Signed;
    B.FiWidth  = Opts.FiDefault.Width;
    B.FiFrac   = Opts.FiDefault.Frac;
  }
  EdgeIndex EI = buildEdgeIndex(*Sub);

  // For each non-port block, decide a unique output variable name.
  std::unordered_map<std::string, std::string> VarOfNode;
  // Tier-5i — port-specific output variable map. For single-output
  // blocks this echoes VarOfNode; for MIMO state-space the block
  // exposes one variable per `out<q>` port. `resolveInputExpr`
  // looks up here first (using the edge's source port) and falls
  // back to `VarOfNode` so non-MIMO blocks still work via the
  // legacy single-var path.
  std::map<std::pair<std::string, std::string>, std::string>
      VarOfNodePort;
  for (auto &P : Inports)  VarOfNode[P.N->Id] = P.Var;
  for (auto &P : Outports) VarOfNode[P.N->Id] = P.Var;
  // Avoid name collisions across blocks.
  std::set<std::string> Used;
  for (auto &Pr : VarOfNode) Used.insert(Pr.second);
  auto uniqueVarFor = [&](const std::string &Id) {
    std::string Base = sanitizeIdent(Id);
    if (Base == "u" || Base == "y" || Base == "t") Base = "v_" + Base;
    std::string Name = Base;
    int Suffix = 1;
    while (Used.count(Name)) Name = Base + "_" + std::to_string(++Suffix);
    Used.insert(Name);
    return Name;
  };

  auto Internal = toposortInternals(*Sub, Diag, SubsystemName);
  if (Internal.empty() && !Outports.empty()) {
    // Allow pure passthrough — an outport wired straight to an inport.
  }
  if (Diag.hasErrors()) return nullptr;

  // Tier-6 — pre-resolve every nested `signal_subsystem` block so
  // we know the inner's metadata (state-slot count, input/output
  // port shape) before allocating outer state slots or emitting
  // call sites. Recursively emits each unique flow_id exactly once
  // into Ctx.Pending; subsequent references reuse the cached
  // function. NestedMeta indexes by the outer block's id.
  std::map<std::string, SubsystemMeta> NestedMeta;
  for (auto *N : Internal) {
    if (N->Kind != "signal_subsystem") continue;
    const std::string *FlowId = N->getData("flow_id");
    if (!FlowId || FlowId->empty()) {
      Diag.error(N->Loc,
                 "signal_subsystem \"" + N->Id +
                     "\": missing data.flow_id (embedded coder needs "
                     "an explicit subflow reference)");
      return nullptr;
    }
    const Flow *Inner = findFlowById(Doc, *FlowId);
    if (!Inner) {
      Diag.error(N->Loc,
                 "signal_subsystem \"" + N->Id +
                     "\": data.flow_id \"" + *FlowId +
                     "\" not found in `.mflow` document");
      return nullptr;
    }
    auto Hit = Ctx.ByFlowId.find(*FlowId);
    if (Hit == Ctx.ByFlowId.end()) {
      auto *InnerFn =
          lowerSubsystemImpl(Doc, Inner->Name, AST, Diag, Opts, Ctx);
      if (!InnerFn) return nullptr;
      auto InnerMetaOpt = describeSubsystem(Doc, Inner->Name, Diag);
      if (!InnerMetaOpt) return nullptr;
      InnerMetaOpt->Name = Inner->Name;
      Ctx.ByFlowId[*FlowId] = {InnerFn, *InnerMetaOpt};
      Ctx.Pending.push_back(InnerFn);
    }
    NestedMeta[N->Id] = Ctx.ByFlowId[*FlowId].second;
  }

  // Tier-6c — multirate scheduling. Walk every stateful block,
  // pull its declared sample period (`sample_time` / `sampleTime`
  // / `Ts` / inherited from `settings.solver.maxStep`), find the
  // minimum positive value as the base period, and compute each
  // block's epoch = round(period / base). Blocks with epoch > 1
  // fire only every `epoch` outer-step ticks; the state update is
  // wrapped in `if mod(_tick, epoch) == 0 ... end` at end-of-body.
  // When ANY block has epoch > 1 the outer gets a hidden `_tick`
  // state slot that counts up each call. Software mode only; HDL
  // multirate needs clock-enable wiring that's a Tier-6c carve-out.
  std::map<std::string, int> BlockEpoch;  // block id → epoch (>=1)
  bool IsMultirate = false;
  auto computeBlockPeriod = [&](const Node *N) -> double {
    double Ts = paramD(*N, "sample_time", 0.0);
    if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
    if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
    return Ts;
  };
  {
    double Base = 0.0;
    for (auto *N : Internal) {
      if (!isStatefulKind(N->Kind)) continue;
      double P = computeBlockPeriod(N);
      if (P > 0.0 && (Base <= 0.0 || P < Base)) Base = P;
    }
    if (Base <= 0.0 && Doc.Settings.Solver.has_value()) {
      const auto &SC = *Doc.Settings.Solver;
      if (SC.MaxStep != "auto") {
        try { Base = std::stod(SC.MaxStep); } catch (...) {}
      }
    }
    if (Base > 0.0) {
      for (auto *N : Internal) {
        if (!isStatefulKind(N->Kind)) continue;
        double P = computeBlockPeriod(N);
        if (P <= 0.0) { BlockEpoch[N->Id] = 1; continue; }
        int E = (int)std::round(P / Base);
        if (E < 1) E = 1;
        BlockEpoch[N->Id] = E;
        if (E > 1) IsMultirate = true;
      }
    }
  }
  // Tier-6c — HDL multirate. The SV pipeline already lowers
  // conditional `if-store` patterns around persistent slot writes
  // to clock-enabled register updates, so the same `if mod(_tick,
  // epoch) == 0 ... end` AST shape works in both software and HDL.
  // The hidden `_tick` counter is allocated as another persistent
  // slot in HDL mode (counts up each clock; resets to 0 on
  // power-up). For HDL the `_tick` slot is declared alongside the
  // other state slots in the State setup loop below.

  // Tier 3 — collect stateful blocks in topo order; each contributes
  // one or more scalar state slots to the function signature.  Per
  // slot: an extra arg `s_<id>` (current state) and an extra return
  // `s_<id>_next` (state for the next tick).  The state read is what
  // the block's "output" becomes (for single-slot stateful blocks
  // like Unit Delay / discrete integrator); the next-state update
  // lands in `s_<id>_next` at the end of the body.
  //
  // Tier-5h — higher-order Transfer Function blocks contribute N
  // slots (where N = denominator order). The block's output isn't
  // a single state read but a linear combination `Σ b_i * x_i` of
  // the per-slot state-reads; that's emitted separately in the
  // dispatch loop below.  Slot LocalVars are per-slot (e.g. `x1`,
  // `x2`, …) rather than the block-id-shared `VarOfNode[id]` that
  // single-slot blocks use.
  struct StateSlot {
    const Node *N;
    std::string CurArg;    // function arg name carrying current state
    std::string NextOut;   // function return name carrying next state
    std::string LocalVar;  // local var for the state-read hoist
  };
  std::vector<StateSlot> States;
  auto stateOrderForBlock = [&](const Node *N) -> int {
    // Tier-5h — for higher-order TF (or Zero-Pole expanded into a
    // TF), the order is the denominator polynomial's degree.
    // For transport_delay, the order is round(delay / Ts).
    // Other stateful blocks contribute one slot each.
    if (N->Kind == "signal_transfer_fcn" ||
        N->Kind == "signal_zero_pole") {
      std::vector<double> Num, Den;
      if (!resolveTFCoeffs(*N, Num, Den)) return 1;
      if (Den.size() < 2) return 1;
      return (int)Den.size() - 1;
    }
    if (N->Kind == "signal_transport_delay") {
      double Delay = paramD(*N, "delay", 0.0);
      double Ts = Opts.TargetRate;
      if (Ts <= 0.0) Ts = paramD(*N, "sample_time", 0.0);
      if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
      if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
      if (Ts <= 0.0 && Doc.Settings.Solver.has_value()) {
        const auto &SC = *Doc.Settings.Solver;
        if (SC.MaxStep != "auto") {
          try { Ts = std::stod(SC.MaxStep); } catch (...) {}
        }
      }
      if (Ts <= 0.0 || Delay <= 0.0) return 1;
      int Taps = (int)std::round(Delay / Ts);
      return Taps < 1 ? 1 : Taps;
    }
    if (N->Kind == "signal_state_space") {
      auto It = N->Params.find("A");
      if (It == N->Params.end()) return 1;
      std::vector<double> A;
      int Ar = 0, Ac = 0;
      parseMatrixStr(It->second, A, Ar, Ac);
      return (Ar > 0 && Ar == Ac) ? Ar : 1;
    }
    return 1;
  };
  for (auto *N : Internal) {
    // Tier-6 — nested subsystem state: one outer slot per inner
    // state arg. Software mode threads them through outer's
    // signature; HDL mode skips (the inner function manages its
    // own persistents).
    if (N->Kind == "signal_subsystem") {
      if (Opts.StateAsPersistent) continue;
      auto MIt = NestedMeta.find(N->Id);
      if (MIt == NestedMeta.end()) continue;
      const SubsystemMeta &Meta = MIt->second;
      for (size_t I = 0; I < Meta.StateArgNames.size(); ++I) {
        StateSlot S;
        S.N = N;
        // `<outer_block_id>_<inner_arg_basename>` keeps the slot
        // names unique across multiple nested subsystem instances.
        std::string Inner = Meta.StateArgNames[I];
        std::string Suffix = (Inner.rfind("s_", 0) == 0)
                                 ? Inner.substr(2) : Inner;
        S.CurArg = "s_" + sanitizeIdent(N->Id) + "_" + Suffix;
        S.NextOut = S.CurArg + "_next";
        // Empty LocalVar — nested subsystem's call statement reads
        // state args directly (no hoisted local).
        S.LocalVar = "";
        States.push_back(S);
      }
      continue;
    }
    if (!isStatefulKind(N->Kind)) continue;
    int Order = stateOrderForBlock(N);
    if (Order == 1) {
      // Single-slot stateful block. LocalVar normally coincides
      // with the block's output variable (state read IS the output);
      // direct-feedthrough-style blocks (Tier-5i) override that.
      StateSlot S;
      S.N        = N;
      S.CurArg   = "s_" + sanitizeIdent(N->Id);
      S.NextOut  = "s_" + sanitizeIdent(N->Id) + "_next";
      if (needsSeparateLocal(N->Kind, Opts.DiscretizeMethod)) {
        S.LocalVar = "x1_" + sanitizeIdent(N->Id);
      } else {
        S.LocalVar = "";  // resolved to VarOfNode[N->Id] below
      }
      States.push_back(S);
    } else {
      // Multi-slot (higher-order TF) — N slots named s_<id>_x1, …,
      // s_<id>_xN with per-slot LocalVars x1_<id>, ..., xN_<id>.
      for (int K = 1; K <= Order; ++K) {
        StateSlot S;
        S.N        = N;
        S.CurArg   = "s_" + sanitizeIdent(N->Id) + "_x" + std::to_string(K);
        S.NextOut  = S.CurArg + "_next";
        S.LocalVar = "x" + std::to_string(K) + "_" + sanitizeIdent(N->Id);
        States.push_back(S);
      }
    }
  }
  // Pre-reserve all the state-related identifiers in `Used` so the
  // generic `uniqueVarFor` doesn't accidentally collide with them.
  for (auto &S : States) {
    Used.insert(S.CurArg);
    Used.insert(S.NextOut);
    if (!S.LocalVar.empty()) Used.insert(S.LocalVar);
  }

  for (auto *N : Internal) {
    VarOfNode[N->Id] = uniqueVarFor(N->Id);
  }
  // Refresh LocalVar for single-slot blocks (now that VarOfNode is
  // populated). Multi-slot LocalVars stay at their per-slot names.
  // Nested-subsystem state slots have no associated local — the
  // call statement reads state args directly, so leave LocalVar
  // empty for them and skip the state-read hoist below.
  for (auto &S : States) {
    if (S.LocalVar.empty() && S.N->Kind != "signal_subsystem")
      S.LocalVar = VarOfNode[S.N->Id];
  }
  // Tier-5i — MIMO state-space exposes one variable per `out<q>`
  // port. Allocate the per-port names now so downstream blocks can
  // wire to specific outputs via the edge index. SISO blocks skip
  // this and fall through to the legacy VarOfNode path.
  for (auto *N : Internal) {
    if (N->Kind != "signal_state_space") continue;
    auto It = N->Params.find("C");
    if (It == N->Params.end()) continue;
    std::vector<double> CM;
    int Cr = 0, Cc = 0;
    parseMatrixStr(It->second, CM, Cr, Cc);
    if (Cr <= 1) continue;  // SISO output — legacy path
    std::string Base = sanitizeIdent(N->Id);
    for (int Q = 1; Q <= Cr; ++Q) {
      std::string PortId = "out" + std::to_string(Q);
      std::string Var = Base + "_y" + std::to_string(Q);
      // Bump on collision the same way `uniqueVarFor` does.
      int Suffix = 1;
      while (Used.count(Var))
        Var = Base + "_y" + std::to_string(Q) + "_" +
              std::to_string(++Suffix);
      Used.insert(Var);
      VarOfNodePort[{N->Id, PortId}] = Var;
    }
    // First output also stamped at the default block id so any
    // legacy lookup (`VarOfNode[N->Id]`) lands on `out1`.
    VarOfNode[N->Id] = VarOfNodePort[{N->Id, "out1"}];
  }

  // Build the function body.
  auto *Body = AST.make<Block>();

  // Resolve a Name expression to the variable feeding the named port.
  // Tier-5i — port-aware lookup: if the source block exposes a
  // per-port variable (e.g. MIMO state-space's `out1`/`out2`/...),
  // use that. Otherwise fall back to the block's single OutVar.
  auto resolveInputExpr = [&](const std::string &ToNode,
                              const std::string &ToPort) -> Expr * {
    auto It = EI.Map.find({ToNode, ToPort});
    if (It == EI.Map.end()) return B.number(0.0);
    const auto &From = It->second;
    auto VP = VarOfNodePort.find(From);
    if (VP != VarOfNodePort.end()) return B.name(VP->second);
    auto VarIt = VarOfNode.find(From.first);
    if (VarIt == VarOfNode.end()) return B.number(0.0);
    return B.name(VarIt->second);
  };

  // Find the input ports of a block (sorted by canonical name —
  // `in1`/`in2`/`u1`/`u2`/...).
  auto inputPortsOf = [&](const Node &N) -> std::vector<std::string> {
    std::vector<std::pair<int, std::string>> Pairs;
    for (const auto &P : N.InPorts) {
      int Idx = parsePortIndex(P.Id);
      if (Idx < 0) Idx = (int)Pairs.size() + 1;
      Pairs.push_back({Idx, P.Id});
    }
    std::sort(Pairs.begin(), Pairs.end());
    std::vector<std::string> Out;
    for (auto &PR : Pairs) Out.push_back(PR.second);
    return Out;
  };

  // §17.5-#4-style discretization spec for each stateful block —
  // computed up front so the per-block dispatch below can render
  // the `<next> = ...` update expression cleanly. Keyed by slot's
  // CurArg so higher-order TF blocks (multiple slots) can each
  // carry their own update expression. Single-slot blocks use the
  // sole entry per CurArg as before.
  std::unordered_map<std::string, Expr *> NextStateExpr;

  // Tier 5 — hard-reject continuous blocks for HDL emit. Software
  // targets (Tier 4) auto-discretise via `--target-rate`; SV must
  // be explicit (the user replaces `signal_integrator` with
  // `signal_discrete_integrator` and Unit Delays in the .mflow).
  if (Opts.RejectContinuous) {
    for (auto *N : Internal) {
      // Tier-5g/h: every continuous block kind currently in the
      // supported set auto-discretises in HDL mode via Forward
      // Euler at the user-picked Ts. Nothing in the reject list
      // for now; future continuous shapes (frequency-domain
      // filters, hybrid networks) would land here.
      if (false) {
        Diag.error(N->Loc,
                   "HDL emit: continuous block `" + N->Kind + "` \"" +
                       N->Id + "\" can't be synthesised — replace with "
                       "the equivalent `signal_discrete_*` in the "
                       ".mflow source (the simulator's continuous "
                       "behaviour is software-only).");
        return nullptr;
      }
    }
  }

  // Tier 5 — `persistent <slot>; if isempty(<slot>) || reset; ... end`
  // initialisation block per stateful block, when SubsystemEmitOptions
  // wants HDL-style internal state (registers).  Routes through the
  // existing matlab_llvm SV pipeline's persistent → reg lowering
  // (`lib/MLIR/Passes/LowerPersistentFiArrays.cpp`).
  if (Opts.StateAsPersistent) {
    for (auto &S : States) {
      auto *Decl = AST.make<PersistentDecl>();
      Decl->Names.push_back(AST.intern(S.CurArg));
      Body->Stmts.push_back(Decl);
    }
    // Wrap each init under `if isempty(<slot>) || reset`.
    for (auto &S : States) {
      // Build: `isempty(<slot>) || reset` — short-circuit OR
      // (BinOp::ShortOr), the form the SV pipeline recognises in
      // its `if isempty(...)` init template.  Bitwise `|` would
      // route through arith.ori which requires integer operands.
      auto *IsEmpty = B.call("isempty", {B.name(S.CurArg)});
      auto *Or = B.bin(BinOp::ShortOr, IsEmpty, B.name("reset"));
      // Build the init expression. For HDL, route through `fi(...)`
      // so the persistent register gets the user's chosen format.
      // Lookup per-port spec; default to the global FiDefault.
      FixedPointSpec Spec = Opts.FiDefault;
      auto It = Opts.FiSpecs.find(S.CurArg);
      if (It != Opts.FiSpecs.end()) Spec = It->second;
      double InitVal = 0.0;
      // Pull the actual IC from the block — the StateSlot vector is
      // already populated above in topo order.
      for (const auto &N : Sub->Nodes) {
        if (N.Id == S.N->Id) {
          InitVal = initialStateOf(N);
          break;
        }
      }
      // fi(<value>, <signed>, <width>, <frac>)
      auto *FiCall = B.call("fi",
                            {B.number(InitVal),
                             B.integer(Spec.Signed ? 1 : 0),
                             B.integer(Spec.Width),
                             B.integer(Spec.Frac)});
      auto *Then = AST.make<Block>();
      Then->Stmts.push_back(B.assign(S.CurArg, FiCall));
      auto *If = AST.make<IfStmt>();
      If->Cond = Or;
      If->Then = Then;
      Body->Stmts.push_back(If);
    }
    // Tier-6c — HDL multirate. For each slow block (epoch > 1),
    // declare a per-block phase counter `phase_<block>` as an
    // additional persistent. Each counter wraps at `epoch - 1`
    // (incremented at end-of-body via an if/else); the block's
    // state update gates on `phase == 0`. Per-block counters
    // avoid the non-synthesisable `mod(_tick, epoch)` shape the
    // software emit uses.
    if (IsMultirate) {
      for (const auto &BP : BlockEpoch) {
        if (BP.second <= 1) continue;
        std::string Phase = "phase_" + sanitizeIdent(BP.first);
        auto *Decl = AST.make<PersistentDecl>();
        Decl->Names.push_back(AST.intern(Phase));
        Body->Stmts.push_back(Decl);
        auto *IsEmpty = B.call("isempty", {B.name(Phase)});
        auto *Or = B.bin(BinOp::ShortOr, IsEmpty, B.name("reset"));
        auto *FiCall = B.call("fi",
                              {B.number(0.0),
                               B.integer(Opts.FiDefault.Signed ? 1 : 0),
                               B.integer(Opts.FiDefault.Width),
                               B.integer(Opts.FiDefault.Frac)});
        auto *Then = AST.make<Block>();
        Then->Stmts.push_back(B.assign(Phase, FiCall));
        auto *If = AST.make<IfStmt>();
        If->Cond = Or;
        If->Then = Then;
        Body->Stmts.push_back(If);
      }
    }
  }

  // Tier-5k — In HDL mode the public input args don't carry an AST
  // type annotation (the `hdl.ports` MLIR attribute is stamped after
  // codegen). Sema's "Phase 5.6 Stage A.1" mechanism in
  // `lib/Sema/TypeInference.cpp` recognises `fi(param, signed, WL,
  // FL)` re-cast and pins the param's binding to that fi spec. Emit
  // one re-cast per public input at the start of the body so Sema
  // propagates Q<W>.<F> through every downstream use; without it
  // the per-port `u_k` stays `any`, breaks fi-multiplication's
  // outer cast, and the SV emitter ends up with a malformed
  // `fi(none, ...)` constructor cast.
  if (Opts.StateAsPersistent) {
    for (auto &P : Inports) {
      FixedPointSpec Spec = Opts.FiDefault;
      auto It = Opts.FiSpecs.find(P.Var);
      if (It != Opts.FiSpecs.end()) Spec = It->second;
      auto *FiCall = B.call("fi",
                            {B.name(P.Var),
                             B.integer(Spec.Signed ? 1 : 0),
                             B.integer(Spec.Width),
                             B.integer(Spec.Frac)});
      Body->Stmts.push_back(B.assign(P.Var, FiCall));
    }
  }

  // Tier 3+4 — hoist every stateful block's STATE READ to the top
  // of the body, BEFORE any consumer evaluates. Without this hoist
  // the topo sort (which drops loop-breakers' outgoing edges) is
  // free to place the consumer ahead of the state-read assignment,
  // and the consumer would read the variable's pre-init zero value
  // — silently producing wrong outputs.  Matches the simulator's
  // "load Z_ first, then evalAll" tick shape (lib/Flowchart/
  // MflowLinkSim.cpp).
  // Iterate per StateSlot — single-slot blocks contribute one entry
  // (LocalVar = block's OutVar); higher-order TF blocks contribute
  // N entries (one local per state, distinct from the block's
  // OutVar). The state-read for each slot becomes `<LocalVar> =
  // <CurArg>`; the block's OutVar gets computed separately below
  // (for higher-order TF, it's the Σ b_i*x_i linear combination).
  for (auto &S : States) {
    // Tier-6 — nested subsystem state slots have no LocalVar (the
    // call statement reads/writes the state args directly).
    if (S.LocalVar.empty()) continue;
    if (Opts.StateAsPersistent) {
      Body->Stmts.push_back(B.assign(S.LocalVar, B.name(S.CurArg)));
    } else {
      // The `+ 0.0` anchors the assignment to `f64` so the static
      // -emit-* pipeline's slot-type inference picks `double`
      // throughout — without it a pure-passthrough subsystem
      // (e.g. one Unit Delay with no internal arithmetic) gets
      // collapsed away as dead-code and the body is empty. The
      // anchor is a no-op at runtime.
      Body->Stmts.push_back(B.assign(S.LocalVar,
                                      B.bin(BinOp::Add, B.name(S.CurArg),
                                            B.number(0.0))));
    }
  }

  // Tier-6 — pre-allocate per-output-port variables for multi-output
  // nested subsystems so downstream blocks can resolve `out1` /
  // `out2` / ... via VarOfNodePort. Single-output inner subsystems
  // route through the legacy VarOfNode path.
  for (auto *N : Internal) {
    if (N->Kind != "signal_subsystem") continue;
    auto MIt = NestedMeta.find(N->Id);
    if (MIt == NestedMeta.end()) continue;
    const SubsystemMeta &Meta = MIt->second;
    if (Meta.OutputNames.size() <= 1) continue;
    std::string Base = sanitizeIdent(N->Id);
    for (size_t I = 0; I < Meta.OutputNames.size(); ++I) {
      // Match the simulator's port convention: `out1`/`out2`/...
      std::string PortId = "out" + std::to_string(I + 1);
      std::string Var = Base + "_y" + std::to_string(I + 1);
      int Suffix = 1;
      while (Used.count(Var))
        Var = Base + "_y" + std::to_string(I + 1) + "_" +
              std::to_string(++Suffix);
      Used.insert(Var);
      VarOfNodePort[{N->Id, PortId}] = Var;
    }
    VarOfNode[N->Id] = VarOfNodePort[{N->Id, "out1"}];
  }

  // Emit one statement per internal block.
  for (auto *N : Internal) {
    auto Ports = inputPortsOf(*N);
    std::vector<Expr *> Ins;
    for (auto &P : Ports) Ins.push_back(resolveInputExpr(N->Id, P));

    // Tier-6 — nested subsystem block. Emit a call to the inner
    // function, threading state args (software mode) or `reset`
    // (HDL mode). Multi-output captures route to per-port vars.
    if (N->Kind == "signal_subsystem") {
      auto MIt = NestedMeta.find(N->Id);
      if (MIt == NestedMeta.end()) {
        Diag.error(N->Loc,
                   "internal: nested subsystem \"" + N->Id +
                       "\" wasn't pre-resolved");
        return nullptr;
      }
      const SubsystemMeta &Meta = MIt->second;
      // Build argument list: P data inputs, then state args (software
      // mode) or `reset` (HDL mode).
      std::vector<Expr *> Args;
      for (size_t I = 0; I < Meta.InputNames.size(); ++I) {
        Expr *A = (I < Ins.size()) ? Ins[I] : B.number(0.0);
        Args.push_back(A);
      }
      if (Opts.StateAsPersistent) {
        if (!Meta.StateArgNames.empty())
          Args.push_back(B.name("reset"));
      } else {
        // Find the outer state slots we allocated for this block
        // and pass their CurArg names through.
        for (auto &S : States) {
          if (S.N != N) continue;
          Args.push_back(B.name(S.CurArg));
        }
      }
      auto *Call = B.call(Meta.Name, std::move(Args));
      // Capture outputs + (software-only) next-state values. In HDL
      // mode we trust the inner's hdl.ports-stamped return type to
      // propagate through Sema's function-call return-type
      // inference; no outer wrap needed.
      size_t NumYs = Meta.OutputNames.size();
      bool HDLMode = Opts.StateAsPersistent;
      size_t NumSnext = HDLMode ? 0 : Meta.StateReturnNames.size();
      if (NumYs == 1 && NumSnext == 0) {
        // Single output, no state — plain assignment.
        const std::string &Dst = VarOfNode[N->Id];
        Body->Stmts.push_back(B.assign(Dst, Call));
      } else {
        // Multi-LHS assign: [y1, y2, ..., s1_next, ...] = inner(...)
        auto *Assign = AST.make<AssignStmt>();
        for (size_t I = 0; I < NumYs; ++I) {
          std::string PortId = "out" + std::to_string(I + 1);
          auto VP = VarOfNodePort.find({N->Id, PortId});
          std::string DstVar = (VP != VarOfNodePort.end())
                                   ? VP->second
                                   : VarOfNode[N->Id];
          Assign->LHS.push_back(B.name(DstVar));
        }
        if (!HDLMode) {
          for (auto &S : States) {
            if (S.N != N) continue;
            Assign->LHS.push_back(B.name(S.NextOut));
          }
        }
        Assign->RHS = Call;
        Assign->Suppressed = true;
        Body->Stmts.push_back(Assign);
      }
      continue;
    }

    if (isStatefulKind(N->Kind)) {
      // State read was hoisted above; we still need to compute the
      // next-state expression here (which DOES depend on the
      // consumer order because the integrator's input is `u[n]`,
      // i.e. the current-tick value upstream — which only exists
      // after every block feeding it has run).
      const std::string CurS  = "s_" + sanitizeIdent(N->Id);
      // Compute next state. Stash in `NextStateExpr` for the
      // final state-update block at end of body. For single-slot
      // blocks the key is the slot's CurArg (== "s_<id>"). For
      // higher-order TF the per-slot entries get filled in the
      // TF dispatch.
      Expr *NextExpr = nullptr;
      if (N->Kind == "signal_unit_delay" || N->Kind == "signal_zoh") {
        // Next state = current input. Software targets anchor with
        // `+ 0.0` so the static -emit-* pipeline's slot inference
        // picks `double` (otherwise pure-passthrough subsystems
        // get DCE'd to `pass`). HDL mode passes the input through
        // directly — adding an f64 literal would taint the
        // persistent slot's fi typing and trip the HWLegalize
        // synthcheck.
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        NextExpr = Opts.StateAsPersistent
                       ? U
                       : B.bin(BinOp::Add, U, B.number(0.0));
      } else if (N->Kind == "signal_dff") {
        // #343 D flip-flop: Q_next = D. Pick the data port explicitly (the
        // `clk` input is the module clock, not data); fall back to the first
        // input if no named data port. Same +0.0 software-anchor as
        // unit_delay so a pure-passthrough register isn't DCE'd.
        Expr *D = nullptr;
        for (size_t pi = 0; pi < Ports.size() && pi < Ins.size(); ++pi)
          if (Ports[pi] == "d" || Ports[pi] == "in" || Ports[pi] == "in1") {
            D = Ins[pi];
            break;
          }
        if (!D) D = Ins.empty() ? B.number(0.0) : Ins.front();
        NextExpr = Opts.StateAsPersistent
                       ? D
                       : B.bin(BinOp::Add, D, B.number(0.0));
      } else if (N->Kind == "signal_discrete_integrator" ||
                 N->Kind == "signal_integrator") {
        // signal_integrator (continuous) gets auto-discretised here
        // — the math is identical to `signal_discrete_integrator`
        // once a sample rate is picked.  Forward-Euler-with-current-
        // u: s_next = s + Ts * u (matches the Tier-3 lowering).
        //
        // Ts resolution order:
        //   1. SubsystemEmitOptions.TargetRate (CLI --target-rate)
        //   2. block's `sample_time` / `sampleTime` / `Ts` param
        //   3. settings.solver.maxStep from the flow doc
        //   4. hard error
        double Ts = Opts.TargetRate;
        if (Ts <= 0.0) Ts = paramD(*N, "sample_time", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
        if (Ts <= 0.0) {
          // settings.solver.maxStep — only meaningful when it's a
          // numeric literal ("auto" means continuous).
          if (Doc.Settings.Solver.has_value()) {
            const auto &SC = *Doc.Settings.Solver;
            if (SC.MaxStep != "auto") {
              try { Ts = std::stod(SC.MaxStep); } catch (...) {}
            }
          }
        }
        if (Ts <= 0.0) {
          Diag.error(N->Loc,
                     "embedded coder: continuous `" + N->Kind +
                         "` block \"" + N->Id +
                         "\" needs a sample period — pass `--target-rate "
                         "<Ts>`, or set `data.sample_time` on the block, "
                         "or declare `settings.solver.maxStep` on the "
                         "flow");
          return nullptr;
        }
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        // Wrap the Ts coefficient in `fi(...)` when in HDL mode so
        // the multiplication stays in integer arithmetic. Software
        // targets emit the bare double (the IEEE arithmetic is what
        // matches the simulator's continuous integrator).
        Expr *TsConst = B.WrapFi ? B.lit(Ts) : B.number(Ts);
        // Tier-5e / 5i — in HDL mode, reference the LOCAL state-read
        // variable (set by the hoisted state-read at top of body)
        // instead of the persistent slot. The local was already
        // converted from f64 → fi/i32 by fptosi; re-fetching the
        // persistent would yield f64 and trigger
        // `matlab.add(f64, i32) → none` downstream. Software
        // targets keep referencing the persistent slot directly
        // (the slot is a plain f64 var, no conversion needed).
        // For Tustin (direct-feedthrough) the state-read local is
        // forced separate from OutVar — see needsSeparateLocal —
        // so the next-state update still sees state[k], not y[k].
        bool SepLocal =
            needsSeparateLocal(N->Kind, Opts.DiscretizeMethod);
        std::string LocalRead =
            SepLocal           ? ("x1_" + sanitizeIdent(N->Id))
            : B.WrapFi         ? VarOfNode[N->Id]
                               : CurS;
        // Tier-5i — Tustin (trapezoidal) integrator: in DF2T form
        //   y[k] = (Ts/2)*u[k] + v[k]         (direct feedthrough)
        //   v_next = v[k] + Ts*u[k]           (same accumulator)
        // The single state slot stores `y[k-1] + (Ts/2)*u[k-1]`, so
        // initial-condition semantics (state init = y at t=0 before
        // the first sample) line up with the Forward-Euler form.
        if (Opts.DiscretizeMethod == "tustin") {
          Expr *HalfTs = B.WrapFi ? B.lit(Ts / 2.0)
                                  : B.number(Ts / 2.0);
          Expr *HalfTsU = B.fiMul(HalfTs, U);
          // Output overwrite with direct feedthrough.
          Body->Stmts.push_back(B.assign(
              VarOfNode[N->Id],
              B.bin(BinOp::Add, B.name(LocalRead), HalfTsU)));
          // State update mirrors Forward Euler — pre-feedthrough.
          Expr *TsU = B.fiMul(TsConst, U);
          NextExpr = B.bin(BinOp::Add, B.name(LocalRead), TsU);
        } else {
          Expr *TsU = B.fiMul(TsConst, U);
          NextExpr = B.bin(BinOp::Add, B.name(LocalRead), TsU);
        }
      } else if (N->Kind == "signal_transfer_fcn" ||
                 N->Kind == "signal_zero_pole") {
        // Tier-5h/5i — continuous N-th order strictly-proper Transfer
        // Function (or Zero-Pole expanded to one).
        //
        //   H(s) = (b_{n-1}*s^{n-1} + ... + b_0) / (s^n + a_{n-1}*s^{n-1} + ... + a_0)
        //
        // Two discretisation paths gated on Opts.DiscretizeMethod:
        // - "forward_euler" (default): controllable canonical state-
        //   space + Forward Euler. Strict-proper continuous TF stays
        //   strict-proper in discrete form.
        // - "tustin" (Tier-5i): polynomial substitution s = (2/Ts)·
        //   (z-1)/(z+1), then Direct Form II Transposed. Discrete TF
        //   gains direct feedthrough n_n*u[k]; same N state slots.
        std::vector<double> NumIn, DenIn;
        if (!resolveTFCoeffs(*N, NumIn, DenIn)) {
          Diag.error(N->Loc,
                     "could not extract (num, den) for `" + N->Kind +
                         "` block \"" + N->Id + "\"");
          return nullptr;
        }
        if (DenIn.size() < 2 || DenIn.front() == 0.0) {
          Diag.error(N->Loc,
                     "signal_transfer_fcn \"" + N->Id +
                         "\": denominator must be degree ≥ 1 with "
                         "non-zero leading coefficient");
          return nullptr;
        }
        if (NumIn.size() >= DenIn.size()) {
          Diag.error(N->Loc,
                     "signal_transfer_fcn \"" + N->Id +
                         "\": only strictly-proper TFs are supported "
                         "(numerator degree must be strictly less than "
                         "denominator degree).");
          return nullptr;
        }
        int Order = (int)DenIn.size() - 1;
        // Same Ts-resolution ladder as the integrator.
        double Ts = Opts.TargetRate;
        if (Ts <= 0.0) Ts = paramD(*N, "sample_time", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
        if (Ts <= 0.0) {
          if (Doc.Settings.Solver.has_value()) {
            const auto &SC = *Doc.Settings.Solver;
            if (SC.MaxStep != "auto") {
              try { Ts = std::stod(SC.MaxStep); } catch (...) {}
            }
          }
        }
        if (Ts <= 0.0) {
          Diag.error(N->Loc,
                     "embedded coder: signal_transfer_fcn \"" + N->Id +
                         "\" needs a sample period — pass "
                         "`--target-rate <Ts>` or set "
                         "`data.sample_time` / `settings.solver.maxStep`");
          return nullptr;
        }
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        std::string SlotPrefix = "s_" + sanitizeIdent(N->Id) + "_x";
        std::string LocalPrefix = "x";
        std::string LocalSuffix = "_" + sanitizeIdent(N->Id);
        // Tier-5i — Tustin direct-feedthrough Order=1 TF/ZP uses a
        // separate state-read local (`x1_<id>`); FE Order=1 keeps
        // the legacy LocalVar = OutVar convention so subsequent
        // consumers can read the block's OutVar without waiting
        // for the per-block dispatch to emit the output equation.
        bool SepLocal =
            needsSeparateLocal(N->Kind, Opts.DiscretizeMethod);
        auto localFor = [&](int K) -> std::string {
          if (Order == 1 && !SepLocal) return VarOfNode[N->Id];
          return LocalPrefix + std::to_string(K) + LocalSuffix;
        };
        auto slotFor = [&](int K) -> std::string {
          if (Order == 1) return "s_" + sanitizeIdent(N->Id);
          return SlotPrefix + std::to_string(K);
        };
        auto fiC = [&](double V) -> Expr * {
          return B.WrapFi ? B.lit(V) : B.number(V);
        };
        const std::string &OutV = VarOfNode[N->Id];
        if (Opts.DiscretizeMethod == "tustin") {
          // Tier-5i — Tustin bilinear: H(z) = NumZ/DenZ, both length
          // Order+1. After normalisation DenZ[0]=1; NumZ[0] is the
          // direct-feedthrough term n_n. Direct Form II Transposed
          // (DF2T) realises the discrete IIR with Order state slots:
          //   y    = n_n * u + v_1
          //   v_i_next = n_{n-i} * u - d_{n-i} * y + v_{i+1}   (1..n-1)
          //   v_n_next = n_0 * u - d_0 * y
          std::vector<double> NumZ, DenZ;
          tustinTF(NumIn, DenIn, Ts, NumZ, DenZ);
          // Emit the output expression first (the next-state updates
          // reference the local OutVar that carries y[k]).
          Expr *YExpr = B.fiMul(fiC(NumZ[0]), U);
          YExpr = B.bin(BinOp::Add, YExpr, B.name(localFor(1)));
          Body->Stmts.push_back(B.assign(OutV, YExpr));
          // Now build the next-state expressions per slot.
          for (int K = 1; K <= Order; ++K) {
            // n_{n-K} = NumZ[K], d_{n-K} = DenZ[K]
            Expr *Term = B.fiMul(fiC(NumZ[K]), U);
            Expr *Neg  = B.fiMul(fiC(-DenZ[K]), B.name(OutV));
            Expr *Acc  = B.bin(BinOp::Add, Term, Neg);
            if (K < Order) {
              Acc = B.bin(BinOp::Add, Acc, B.name(localFor(K + 1)));
            }
            NextStateExpr[slotFor(K)] = Acc;
          }
          NextExpr = nullptr;
          continue;
        }
        // Forward Euler — controllable canonical state-space.
        //   x_1' = x_2;  x_2' = x_3;  ...;  x_{n-1}' = x_n
        //   x_n' = -a_0*x_1 - a_1*x_2 - ... - a_{n-1}*x_n + u
        //   y    =  b_0*x_1 +  b_1*x_2 + ... +  b_{n-1}*x_n
        // Normalise: divide every coefficient by den's leading
        // coefficient so the canonical-form denominator is
        // s^n + a_{n-1}*s^{n-1} + ... + a_0.
        double Lead = DenIn.front();
        // A[i] = coefficient of s^i in the normalised denominator
        // (lowest power first), with A[Order] = 1 dropped (implicit).
        std::vector<double> A(Order, 0.0);
        for (int i = 0; i < Order; ++i)
          A[i] = DenIn[DenIn.size() - 1 - i] / Lead;
        // BV[i] = coefficient of s^i in the numerator (lowest power
        // first), 0.0 for missing powers. Length Order (strictly
        // proper ⇒ no s^Order term).
        std::vector<double> BV(Order, 0.0);
        for (size_t i = 0; i < NumIn.size(); ++i) {
          size_t Power = NumIn.size() - 1 - i;
          if ((int)Power < Order) BV[Power] = NumIn[i] / Lead;
        }
        // Build the next-state expressions per slot. The first
        // Order-1 slots are simple advances; the last slot rolls
        // up all the -A[i]*x_{i+1} terms plus the input.
        for (int K = 1; K <= Order - 1; ++K) {
          // x_K_next = x_K + Ts * x_{K+1}
          Expr *XK   = B.name(localFor(K));
          Expr *XK1  = B.name(localFor(K + 1));
          Expr *Term = B.fiMul(fiC(Ts), XK1);
          NextStateExpr[slotFor(K)] = B.bin(BinOp::Add, XK, Term);
        }
        // Last slot: x_n_next = x_n + Ts*(-A[0]*x_1 - ... - A[n-1]*x_n + u)
        Expr *Acc = U;
        for (int K = 1; K <= Order; ++K) {
          Expr *NegA = fiC(-A[K - 1]);
          Expr *Term = B.fiMul(NegA, B.name(localFor(K)));
          Acc = B.bin(BinOp::Add, Acc, Term);
        }
        // x_n + Ts * (Acc)
        Expr *XN     = B.name(localFor(Order));
        Expr *TsAcc  = B.fiMul(fiC(Ts), Acc);
        Expr *XNNext = B.bin(BinOp::Add, XN, TsAcc);
        NextStateExpr[slotFor(Order)] = XNNext;
        // Emit the block's output `OutVar = Σ b_i * x_i`. For Order=1
        // the existing state-read hoist already set OutVar = state;
        // overwrite with the b_0-weighted form so the output matches
        // the controllable-canonical y equation.
        Expr *YAcc = nullptr;
        for (int K = 1; K <= Order; ++K) {
          Expr *Term = B.fiMul(fiC(BV[K - 1]), B.name(localFor(K)));
          YAcc = YAcc ? B.bin(BinOp::Add, YAcc, Term) : Term;
        }
        if (!YAcc) YAcc = B.number(0.0);
        Body->Stmts.push_back(B.assign(OutV, YAcc));
        NextExpr = nullptr;  // already filled per slot above
        continue;            // skip the single-key store below
      } else if (N->Kind == "signal_state_space") {
        // Tier-5h/5i — continuous state-space (A, B, C; D = 0
        // strict-proper). Supports SISO (B is N×1, C is 1×N) and
        // MIMO (B is N×P, C is Q×N — Tier-5i). Two discretisation
        // paths gated on Opts.DiscretizeMethod:
        // - "forward_euler" (default):
        //     x[k+1] = (I + Ts*A)*x[k] + Ts*B*u[k];   y[k] = C*x[k]
        //   N state slots in the original (A, B, C) basis. MIMO
        //   reads P inputs (`in1`..`inP`) and emits Q output
        //   assignments (one per `out1`..`outQ` port variable).
        // - "tustin": SISO only (Faddeev-LeVerrier → DF2T). MIMO
        //   Tustin is a future follow-up (requires matrix Tustin
        //   transformation with non-trivial state basis change).
        auto getStr = [&](const char *Key) -> std::string {
          auto It = N->Params.find(Key);
          return It == N->Params.end() ? std::string{} : It->second;
        };
        std::vector<double> AM, BM, CM;
        int Ar = 0, Ac = 0, Br = 0, Bc = 0, Cr = 0, Cc = 0;
        parseMatrixStr(getStr("A"), AM, Ar, Ac);
        parseMatrixStr(getStr("B"), BM, Br, Bc);
        parseMatrixStr(getStr("C"), CM, Cr, Cc);
        if (Ar == 0 || Ar != Ac) {
          Diag.error(N->Loc,
                     "signal_state_space \"" + N->Id +
                         "\": A matrix must be square and non-empty");
          return nullptr;
        }
        int Order = Ar;
        if (Br != Order || Bc < 1) {
          Diag.error(N->Loc,
                     "signal_state_space \"" + N->Id +
                         "\": B must be " + std::to_string(Order) +
                         "×P (P ≥ 1 input columns)");
          return nullptr;
        }
        if (Cc != Order || Cr < 1) {
          Diag.error(N->Loc,
                     "signal_state_space \"" + N->Id +
                         "\": C must be Q×" + std::to_string(Order) +
                         " (Q ≥ 1 output rows)");
          return nullptr;
        }
        int P = Bc;  // number of inputs
        int Q = Cr;  // number of outputs
        bool MIMO = (P > 1 || Q > 1);
        // D, if present, must be the zero matrix.
        auto DStr = getStr("D");
        if (!DStr.empty()) {
          std::vector<double> DM; int Dr=0, Dc=0;
          parseMatrixStr(DStr, DM, Dr, Dc);
          for (double D : DM) {
            if (D != 0.0) {
              Diag.error(N->Loc,
                         "signal_state_space \"" + N->Id +
                             "\": D must be zero (strict-proper only)");
              return nullptr;
            }
          }
        }
        double Ts = Opts.TargetRate;
        if (Ts <= 0.0) Ts = paramD(*N, "sample_time", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
        if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
        if (Ts <= 0.0 && Doc.Settings.Solver.has_value()) {
          const auto &SC = *Doc.Settings.Solver;
          if (SC.MaxStep != "auto") {
            try { Ts = std::stod(SC.MaxStep); } catch (...) {}
          }
        }
        if (Ts <= 0.0) {
          Diag.error(N->Loc,
                     "embedded coder: signal_state_space \"" + N->Id +
                         "\" needs a sample period");
          return nullptr;
        }
        std::string LocalPrefix = "x";
        std::string LocalSuffix = "_" + sanitizeIdent(N->Id);
        std::string SlotPrefix = "s_" + sanitizeIdent(N->Id) + "_x";
        // Tier-5i — Tustin direct-feedthrough Order=1 SS uses a
        // separate state-read local (`x1_<id>`); FE Order=1 keeps
        // legacy LocalVar = OutVar.
        bool SepLocal =
            needsSeparateLocal(N->Kind, Opts.DiscretizeMethod);
        // MIMO output is per-port, so SISO-style OutVar reuse
        // doesn't apply — each output port gets its own variable.
        auto localFor = [&](int K) -> std::string {
          if (Order == 1 && !SepLocal && !MIMO)
            return VarOfNode[N->Id];
          return LocalPrefix + std::to_string(K) + LocalSuffix;
        };
        auto slotFor = [&](int K) -> std::string {
          if (Order == 1) return "s_" + sanitizeIdent(N->Id);
          return SlotPrefix + std::to_string(K);
        };
        auto fiC = [&](double V) -> Expr * {
          return B.WrapFi ? B.lit(V) : B.number(V);
        };
        // Resolve the K-th input expression. Ins is already sorted
        // by canonical port name (`in`, `in1`, `in2`, ...). For SISO
        // we just take Ins.front(); for MIMO we index into Ins with
        // bounds-check fallback to zero.
        auto inputExpr = [&](int K /*1-based*/) -> Expr * {
          if (K < 1 || K > (int)Ins.size()) return B.number(0.0);
          return Ins[K - 1];
        };
        const std::string &OutV = VarOfNode[N->Id];
        if (Opts.DiscretizeMethod == "tustin") {
          // Tier-5l — matrix bilinear Tustin (supports MIMO + SISO).
          //
          //   Ad = M(I + αA),  Bd = α(I + Ad)MB
          //   Cd = C,           Dd = α·C·M·B
          //
          // (α = Ts/2, M = (I - αA)^-1.) The direct-feedthrough term
          // `Dd·u` lives in the output equation; state-update reads
          // x[k] (NOT y[k]) so the SeparateLocal hoist (Tier-5i) is
          // already correct.
          std::vector<double> Ad, Bd, Cd, Dd;
          if (!tustinSS(AM, Order, BM, P, CM, Q, Ts, Ad, Bd, Cd, Dd)) {
            Diag.error(N->Loc,
                       "signal_state_space \"" + N->Id +
                           "\": Tustin requires (I - (Ts/2)·A) to be "
                           "invertible at the chosen sample period");
            return nullptr;
          }
          // Output equations first (each y_q depends on current x
          // AND current u via Dd). State update happens at end-of-
          // body and reads localFor() = the un-overwritten state.
          for (int Qi = 1; Qi <= Q; ++Qi) {
            Expr *YAcc = nullptr;
            for (int K = 1; K <= Order; ++K) {
              double Cqk = Cd[(size_t)(Qi - 1) * Order + (K - 1)];
              if (Cqk == 0.0) continue;
              Expr *T = B.fiMul(fiC(Cqk), B.name(localFor(K)));
              YAcc = YAcc ? B.bin(BinOp::Add, YAcc, T) : T;
            }
            for (int K = 1; K <= P; ++K) {
              double Dqk = Dd[(size_t)(Qi - 1) * P + (K - 1)];
              if (Dqk == 0.0) continue;
              Expr *T = B.fiMul(fiC(Dqk), inputExpr(K));
              YAcc = YAcc ? B.bin(BinOp::Add, YAcc, T) : T;
            }
            if (!YAcc) YAcc = B.number(0.0);
            std::string Dst;
            if (MIMO) {
              std::string PortId = "out" + std::to_string(Qi);
              auto It = VarOfNodePort.find({N->Id, PortId});
              Dst = (It != VarOfNodePort.end()) ? It->second : OutV;
            } else {
              Dst = OutV;
            }
            Body->Stmts.push_back(B.assign(Dst, YAcc));
          }
          // State update: z_i[k+1] = Σ_j Ad[i,j]·z_j + Σ_k Bd[i,k]·u_k.
          for (int I = 1; I <= Order; ++I) {
            Expr *Acc = nullptr;
            for (int J = 1; J <= Order; ++J) {
              double Aij = Ad[(size_t)(I - 1) * Order + (J - 1)];
              if (Aij == 0.0) continue;
              Expr *T = B.fiMul(fiC(Aij), B.name(localFor(J)));
              Acc = Acc ? B.bin(BinOp::Add, Acc, T) : T;
            }
            for (int K = 1; K <= P; ++K) {
              double Bik = Bd[(size_t)(I - 1) * P + (K - 1)];
              if (Bik == 0.0) continue;
              Expr *T = B.fiMul(fiC(Bik), inputExpr(K));
              Acc = Acc ? B.bin(BinOp::Add, Acc, T) : T;
            }
            if (!Acc) Acc = B.number(0.0);
            NextStateExpr[slotFor(I)] = Acc;
          }
          NextExpr = nullptr;
          continue;
        }
        // Forward Euler — row i:
        //   x_i_next = x_i + Ts*(Σ_j A[i,j]*x_j + Σ_k B[i,k]*u_k)
        for (int I = 1; I <= Order; ++I) {
          // Σ_k B[i,k]*u_k
          Expr *Acc = nullptr;
          for (int K = 1; K <= P; ++K) {
            double Bik = BM[(I - 1) * P + (K - 1)];
            if (Bik == 0.0) continue;
            Expr *T = B.fiMul(fiC(Bik), inputExpr(K));
            Acc = Acc ? B.bin(BinOp::Add, Acc, T) : T;
          }
          // Σ_j A[i,j]*x_j
          for (int J = 1; J <= Order; ++J) {
            double Aij = AM[(I - 1) * Order + (J - 1)];
            if (Aij == 0.0) continue;
            Expr *T = B.fiMul(fiC(Aij), B.name(localFor(J)));
            Acc = Acc ? B.bin(BinOp::Add, Acc, T) : T;
          }
          if (!Acc) Acc = B.number(0.0);
          Expr *TsAcc = B.fiMul(fiC(Ts), Acc);
          Expr *XI = B.name(localFor(I));
          NextStateExpr[slotFor(I)] = B.bin(BinOp::Add, XI, TsAcc);
        }
        // Outputs: y_q = Σ_n C[q,n]*x_n. For MIMO emit one stmt per
        // output port (the `out<q>` var was pre-allocated above).
        // For SISO emit a single OutVar assignment as before.
        for (int Qi = 1; Qi <= Q; ++Qi) {
          Expr *YAcc = nullptr;
          for (int K = 1; K <= Order; ++K) {
            double Cqk = CM[(Qi - 1) * Order + (K - 1)];
            if (Cqk == 0.0) continue;
            Expr *Term = B.fiMul(fiC(Cqk), B.name(localFor(K)));
            YAcc = YAcc ? B.bin(BinOp::Add, YAcc, Term) : Term;
          }
          if (!YAcc) YAcc = B.number(0.0);
          std::string Dst;
          if (MIMO) {
            std::string PortId = "out" + std::to_string(Qi);
            auto It = VarOfNodePort.find({N->Id, PortId});
            Dst = (It != VarOfNodePort.end()) ? It->second : OutV;
          } else {
            Dst = OutV;
          }
          Body->Stmts.push_back(B.assign(Dst, YAcc));
        }
        NextExpr = nullptr;
        continue;
      } else if (N->Kind == "signal_transport_delay") {
        // Tier-5h — chain of N unit delays. State slots
        // `s_<id>_x1` … `s_<id>_xN` form a shift register; output
        // is the oldest tap (x_1), new value enters at x_N.
        //   x_1_next = x_2; x_2_next = x_3; ...; x_{N-1}_next = x_N;
        //   x_N_next = u
        //   y = x_1
        int Order = (int)0;
        // Recompute order locally so the dispatch matches the
        // state-slot allocation (avoid re-running the lambda since
        // it captures by reference).
        {
          double Delay = paramD(*N, "delay", 0.0);
          double Ts = Opts.TargetRate;
          if (Ts <= 0.0) Ts = paramD(*N, "sample_time", 0.0);
          if (Ts <= 0.0) Ts = paramD(*N, "sampleTime", 0.0);
          if (Ts <= 0.0) Ts = paramD(*N, "Ts", 0.0);
          if (Ts <= 0.0 && Doc.Settings.Solver.has_value()) {
            const auto &SC = *Doc.Settings.Solver;
            if (SC.MaxStep != "auto") {
              try { Ts = std::stod(SC.MaxStep); } catch (...) {}
            }
          }
          if (Ts <= 0.0 || Delay <= 0.0) {
            Diag.error(N->Loc,
                       "signal_transport_delay \"" + N->Id +
                           "\" needs both `delay` and a sample period "
                           "(`--target-rate <Ts>` or block "
                           "`data.sample_time`)");
            return nullptr;
          }
          Order = (int)std::round(Delay / Ts);
          if (Order < 1) Order = 1;
        }
        std::string SlotPrefix = "s_" + sanitizeIdent(N->Id) + "_x";
        std::string LocalPrefix = "x";
        std::string LocalSuffix = "_" + sanitizeIdent(N->Id);
        auto localFor = [&](int K) -> std::string {
          if (Order == 1) return VarOfNode[N->Id];
          return LocalPrefix + std::to_string(K) + LocalSuffix;
        };
        auto slotFor = [&](int K) -> std::string {
          if (Order == 1) return "s_" + sanitizeIdent(N->Id);
          return SlotPrefix + std::to_string(K);
        };
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        // Shift register state-update chain.
        for (int K = 1; K <= Order - 1; ++K) {
          NextStateExpr[slotFor(K)] = B.name(localFor(K + 1));
        }
        // Newest tap takes the input.
        NextStateExpr[slotFor(Order)] =
            Opts.StateAsPersistent
                ? U
                : B.bin(BinOp::Add, U, B.number(0.0));
        // Block output = oldest tap (x_1) — re-assert it here so the
        // state-read hoist's `LocalVar = current state` writes to
        // the block's output variable for multi-slot delays.
        if (Order > 1) {
          Body->Stmts.push_back(
              B.assign(VarOfNode[N->Id], B.name(localFor(1))));
        }
        NextExpr = nullptr;
        continue;
      } else {
        NextExpr = B.number(0.0);
      }
      // Single-slot block — key by CurArg ("s_<id>") to match the
      // higher-order-TF per-slot keying.
      NextStateExpr["s_" + sanitizeIdent(N->Id)] = NextExpr;
      continue;
    }

    auto *Stmt = lowerBlock(*N, VarOfNode[N->Id], Ins, Ports, B, Diag);
    if (!Stmt) return nullptr;
    Body->Stmts.push_back(Stmt);
  }

  // Tier-5k — In HDL mode, wrap each outport assignment in
  // `fi(<rhs>, S, W, F)` so Sema narrows the output type back to
  // the user-declared Q<W>.<F>. Without it, sum-chains widen the
  // inferred FL to (sum of operand FLs) and the SV port emits
  // with a wider-than-spec width. Skip the wrap when the upstream
  // block is a boolean producer (comparators / logical ops) since
  // wrapping a 1-bit value in a Q<W>.<F> cast routes through the
  // f64-quantize runtime, which is not synthesisable.
  auto isBooleanProducer = [&](const std::string &Kind) {
    return Kind == "signal_relop" || Kind == "signal_logical" ||
           Kind == "signal_compare_to_zero" ||
           Kind == "signal_compare_to_constant";
  };
  // Tier-6 — a nested signal_subsystem's return value is already
  // narrowed by the inner's own outport wrap. Re-wrapping at the
  // outer would route the call result through `matlab_fi_quantize_s`
  // (Sema can't propagate the call's fi return type cross-function),
  // which doesn't synthesise.
  auto isNarrowedProducer = [&](const std::string &Kind) {
    return Kind == "signal_subsystem";
  };
  // For each outport, append `<port_var> = <feeding_var>;`.
  for (auto &P : Outports) {
    // An outport has exactly one input (`in`). Resolve to its source.
    Expr *Rhs = nullptr;
    std::string UpstreamKind;
    for (const auto &Port : P.N->InPorts) {
      Rhs = resolveInputExpr(P.N->Id, Port.Id);
      // Look up the upstream block's kind via the edge index so the
      // outport wrap can skip booleans.
      auto It = EI.Map.find({P.N->Id, Port.Id});
      if (It != EI.Map.end()) {
        for (const auto &NN : Sub->Nodes) {
          if (NN.Id == It->second.first) {
            UpstreamKind = NN.Kind;
            break;
          }
        }
      }
      break;
    }
    if (!Rhs) Rhs = B.number(0.0);
    if (Opts.StateAsPersistent && !isBooleanProducer(UpstreamKind) &&
        !isNarrowedProducer(UpstreamKind)) {
      FixedPointSpec Spec = Opts.FiDefault;
      auto It = Opts.FiSpecs.find(P.Var);
      if (It != Opts.FiSpecs.end()) Spec = It->second;
      Rhs = B.call("fi",
                    {Rhs,
                     B.integer(Spec.Signed ? 1 : 0),
                     B.integer(Spec.Width),
                     B.integer(Spec.Frac)});
    }
    Body->Stmts.push_back(B.assign(P.Var, Rhs));
  }

  // Tier 3 — emit `<NextOut> = <NextExpr>;` for every stateful block
  // so the multi-return picks up the next-state values. In Tier-5
  // persistent mode the next state lands directly in the persistent
  // slot (no separate `_next` return — the persistent itself is the
  // mutable storage), which the SV pipeline lowers to a register.
  // Tier-6 — nested subsystem state slots are written directly by
  // the call statement above (multi-LHS assign captures both Y and
  // S_next), so skip them here.
  // Tier-6c — multirate gating. When a stateful block's epoch > 1,
  // the state update only fires every `epoch` ticks.
  // Software mode: gate on `mod(_tick, epoch) == 0` (single global
  //   `_tick` counter increments each tick; non-firing ticks hold
  //   the previous state).
  // HDL mode: `mod` doesn't synthesise, so for each multirate
  //   block we emit a separate counter that wraps at `epoch-1`
  //   (`phase_<block>`), and gate on `phase == 0`. The phase
  //   counter increments by 1 and resets to 0 at epoch-1 via an
  //   if/else; both branches synthesise to a clean 2-way mux.
  for (auto &S : States) {
    if (S.N->Kind == "signal_subsystem") continue;
    Expr *E = NextStateExpr[S.CurArg];
    if (!E) E = B.number(0.0);
    int Epoch = 1;
    auto EIt = BlockEpoch.find(S.N->Id);
    if (EIt != BlockEpoch.end()) Epoch = EIt->second;
    Stmt *Assign;
    if (Opts.StateAsPersistent) {
      Assign = B.assign(S.CurArg, E);
    } else {
      Assign = B.assign(S.NextOut, E);
    }
    if (Epoch <= 1) {
      Body->Stmts.push_back(Assign);
      continue;
    }
    // Multirate slow slot — gate on epoch boundary.
    Expr *Cond;
    if (Opts.StateAsPersistent) {
      // HDL: per-block phase counter (declared above with the
      // other persistents). gate = (phase_<block> == 0).
      std::string Phase = "phase_" + sanitizeIdent(S.N->Id);
      Cond = B.bin(BinOp::Eq, B.name(Phase),
                    B.WrapFi ? B.lit(0.0) : B.number(0.0));
    } else {
      // Software: global `_tick` + mod.
      auto *Mod = B.call("mod", {B.name("_tick"), B.integer(Epoch)});
      Cond = B.bin(BinOp::Eq, Mod, B.integer(0));
    }
    auto *If = AST.make<IfStmt>();
    If->Cond = Cond;
    If->Then = AST.make<Block>();
    If->Then->Stmts.push_back(Assign);
    // Hold path — assign next = current state.
    Stmt *Hold;
    if (Opts.StateAsPersistent) {
      Hold = B.assign(S.CurArg, B.name(S.CurArg));
    } else {
      Hold = B.assign(S.NextOut, B.name(S.CurArg));
    }
    If->Else = AST.make<Block>();
    If->Else->Stmts.push_back(Hold);
    Body->Stmts.push_back(If);
  }
  // Tier-6c — tick / phase counter updates.
  // Software emit writes to `_tick_next` (returned alongside the
  // other state-next values).
  // HDL emit increments per-block `phase_<block>` counters that
  // wrap at `epoch-1`. Each counter starts at 0 and rolls over
  // synchronously with its block's epoch.
  if (IsMultirate) {
    if (Opts.StateAsPersistent) {
      for (const auto &BP : BlockEpoch) {
        if (BP.second <= 1) continue;
        std::string Phase = "phase_" + sanitizeIdent(BP.first);
        // phase = (phase == epoch-1) ? 0 : phase + 1
        auto *Cond = B.bin(BinOp::Eq, B.name(Phase),
                            B.WrapFi ? B.lit((double)(BP.second - 1))
                                     : B.number(BP.second - 1));
        auto *If = AST.make<IfStmt>();
        If->Cond = Cond;
        If->Then = AST.make<Block>();
        If->Then->Stmts.push_back(B.assign(
            Phase, B.WrapFi ? B.lit(0.0) : B.number(0.0)));
        If->Else = AST.make<Block>();
        If->Else->Stmts.push_back(B.assign(
            Phase, B.bin(BinOp::Add, B.name(Phase), B.integer(1))));
        Body->Stmts.push_back(If);
      }
    } else {
      Body->Stmts.push_back(
          B.assign("_tick_next",
                   B.bin(BinOp::Add, B.name("_tick"), B.integer(1))));
    }
  }

  // Build the function node.
  auto *Fn = AST.make<Function>();
  Fn->Name = AST.intern(sanitizeIdent(SubsystemName));
  for (auto &P : Inports)  Fn->Inputs.push_back(AST.intern(P.Var));
  // Tier-6 — a stateful nested subsystem also requires the outer
  // to forward `reset` even when the outer itself has no own state
  // slots. Detect by looking at the nested metadata.
  bool NestedNeedsReset = false;
  for (const auto &MP : NestedMeta) {
    if (!MP.second.StateArgNames.empty()) {
      NestedNeedsReset = true;
      break;
    }
  }
  if (Opts.StateAsPersistent && (!States.empty() || NestedNeedsReset)) {
    // Tier 5 — HDL needs an explicit `reset` boundary to clear the
    // persistent regs on power-up. Add it as the final arg so the
    // synthesised SV module exposes `reset` alongside the data
    // inputs. Stateless subsystems have no regs and skip the reset.
    Fn->Inputs.push_back(AST.intern("reset"));
  } else if (!Opts.StateAsPersistent) {
    // Tier 3 — append state args after the inports so the public
    // signature reads `step(u1, ..., uN, s_<a>, s_<b>, ...)`.
    for (auto &S : States) Fn->Inputs.push_back(AST.intern(S.CurArg));
  }
  // Tier-6c — multirate counter as the last input arg (after all
  // state args). Software mode only; HDL multirate is gated above.
  if (IsMultirate && !Opts.StateAsPersistent) {
    Fn->Inputs.push_back(AST.intern("_tick"));
  }
  for (auto &P : Outports) Fn->Outputs.push_back(AST.intern(P.Var));
  if (!Opts.StateAsPersistent) {
    // And next-state returns after the regular outports:
    //   `[y1, ..., yM, s_<a>_next, s_<b>_next, ...]`.
    for (auto &S : States) Fn->Outputs.push_back(AST.intern(S.NextOut));
  }
  // Tier-6c — multirate counter as the last return.
  if (IsMultirate && !Opts.StateAsPersistent) {
    Fn->Outputs.push_back(AST.intern("_tick_next"));
  }
  Fn->Body = Body;
  return Fn;
}

matlab::TranslationUnit *buildSubsystemTU(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts) {
  // Tier-6 — pass a NestedCtx through the lowering so any
  // `signal_subsystem` block's referenced flow gets emitted as a
  // sibling helper in the same TU. The outer subsystem ends up
  // last in TU.Functions; inner helpers precede it in emission
  // order (innermost first).
  NestedCtx Ctx;
  auto *Fn =
      lowerSubsystemImpl(Doc, SubsystemName, AST, Diag, Opts, Ctx);
  if (!Fn) return nullptr;

  auto *TU = AST.make<TranslationUnit>();
  // Tier-6 — inner helpers are pushed FIRST so Sema's
  // TypeInference visits them before the outer subsystem.  That
  // lets the outer's `visitCallOrIndex` pull a typed return value
  // from the inner function's already-typed `OutputRefs` instead
  // of returning `Any` (which would route the call result through
  // the non-synthesisable f64 quantize cast in HDL mode).
  for (auto *Helper : Ctx.Pending) {
    TU->Functions.push_back(Helper);
  }

  // Tier 5 — collect every `signal_matlab_fcn` block in the subsystem
  // and add its `params.function_body` as a sibling local function in
  // the same TU. The block-level dispatch already emits a call site
  // named `<userFnName>_<sanitizedBlockId>` (renamed for uniqueness);
  // here we parse the user-supplied body, rename the entry, and
  // append to the TU's Functions list so Sema + lowering pick it up
  // alongside the main subsystem function.
  // Tier-6b — user functions must be pushed BEFORE the outer Fn so
  // Sema's TypeInference visits them first (same ordering as nested
  // subsystems). And in HDL mode, the user function's args are
  // re-cast to fi at the start of its body so Sema's Phase 5.6
  // Stage A.1 `ParamFiSpec` mechanism pins their types — without
  // this, bare-int arithmetic inside the body (`u1 * 3 + u2 * 5`)
  // leaves the args as `any`, and the outer's outport wrap routes
  // the call result through the non-synthesisable
  // `matlab_fi_quantize_s` constructor.
  const Flow *Sub = Doc.findFlow(SubsystemName);
  if (Sub) {
    for (const auto &N : Sub->Nodes) {
      if (N.Kind != "signal_matlab_fcn") continue;
      auto It = N.Params.find("function_body");
      if (It == N.Params.end()) continue;
      // Parse the user body in its own SourceManager so its
      // SourceLocations don't collide with the main TU's. The
      // outer Diag still sees any errors.
      auto LocalSM = std::make_unique<matlab::SourceManager>();
      auto LocalDiag = std::make_unique<matlab::DiagnosticEngine>(*LocalSM);
      matlab::FileID F = LocalSM->addBuffer(
          "<signal_matlab_fcn:" + N.Id + ">", It->second);
      matlab::Lexer L(*LocalSM, F, *LocalDiag);
      auto Toks = L.tokenize();
      if (LocalDiag->hasErrors()) {
        Diag.error(N.Loc,
                   "signal_matlab_fcn \"" + N.Id +
                       "\": lex error in `params.function_body`");
        return nullptr;
      }
      matlab::Parser P(std::move(Toks), AST, *LocalDiag);
      auto *FnTU = P.parseFile();
      if (!FnTU || LocalDiag->hasErrors() || FnTU->Functions.empty()) {
        Diag.error(N.Loc,
                   "signal_matlab_fcn \"" + N.Id +
                       "\": parse error in `params.function_body`");
        return nullptr;
      }
      // Rename the entry function to `<orig>_<blockId>` so callers
      // who reference the unique helper name resolve cleanly.
      auto *UserFn = FnTU->Functions.front();
      std::string Helper =
          std::string(UserFn->Name) + "_" + sanitizeIdent(N.Id);
      UserFn->Name = AST.intern(Helper);
      // Tier-6b — inject `<arg> = fi(<arg>, S, W, F)` at the start of
      // the body in HDL mode so Sema pins the args' fi specs.
      if (Opts.StateAsPersistent && UserFn->Body) {
        ASTBuilder UB{AST};
        UB.WrapFi   = true;
        UB.FiSigned = Opts.FiDefault.Signed;
        UB.FiWidth  = Opts.FiDefault.Width;
        UB.FiFrac   = Opts.FiDefault.Frac;
        std::vector<Stmt *> Casts;
        for (auto Arg : UserFn->Inputs) {
          std::string Name(Arg);
          auto *FiCall = UB.call("fi",
                                  {UB.name(Name),
                                   UB.integer(UB.FiSigned ? 1 : 0),
                                   UB.integer(UB.FiWidth),
                                   UB.integer(UB.FiFrac)});
          Casts.push_back(UB.assign(Name, FiCall));
        }
        UserFn->Body->Stmts.insert(UserFn->Body->Stmts.begin(),
                                    Casts.begin(), Casts.end());
      }
      TU->Functions.push_back(UserFn);
      // The LocalSM/LocalDiag go out of scope here; the AST nodes
      // they back stay alive in `AST` (the bump allocator on the
      // outer ASTContext). The string_views into the SM's buffer
      // would dangle — re-intern them.
      // (Best-effort: most string_view fields are short-lived
      // identifiers that the resolver re-interns; for fields like
      // FPLiteral.Text we walk the body and re-intern below.)
      // Simpler approach: intern the whole source text into `AST`
      // so the LocalSM's buffer isn't the backing store.  Since
      // we already have the text in `It->second`, we can ensure
      // it's interned in the outer AST too — but the parser
      // already created string_views pointing at the LocalSM
      // buffer. Drop both LocalSM and LocalDiag at TU end-of-life
      // by appending to a TU-scoped reservoir.  For now we leak
      // them deliberately by stashing in a static — they're tiny
      // per matlab_fcn block and the matlabc process exits at end
      // of compile anyway.
      static std::vector<std::unique_ptr<matlab::SourceManager>> KeepSM;
      static std::vector<std::unique_ptr<matlab::DiagnosticEngine>> KeepDiag;
      KeepSM.push_back(std::move(LocalSM));
      KeepDiag.push_back(std::move(LocalDiag));
    }
  }
  // Outer subsystem function comes LAST so Sema visits inner helpers
  // (nested subsystems + user matlab_fcn bodies) first — see the
  // Tier-6 ordering note above.
  TU->Functions.push_back(Fn);

  // Synthesise a driver script that calls the function with concrete
  // f64 args. That call site forces the static `-emit-*` pipeline to
  // refine the function's slots to `double` instead of `none` / `void*`
  // (the function-only-file pitfall documented during the §17.5 #8
  // work). The driver is a single AssignStmt placed in the TU's
  // Script node — matlab_llvm allows a script body before function
  // definitions in the same file.
  ASTBuilder B{AST};
  // Tier 5 — skip the priming driver for HDL emit. HDL gets its
  // arg/result types from the `hdl.ports` MLIR attribute the
  // caller stamps post-lowering (no need to type-refine via a
  // call site), and the call's f64 args confuse the SV pipeline
  // (e.g. `arith.shli` on an f64 operand of the constant-mul
  // optimisation).  Software targets keep the driver — they
  // need the concrete-typed call site to refine the function's
  // slots to `double`.
  if (!Fn->Inputs.empty() && !Opts.StateAsPersistent) {
    std::vector<Expr *> Args;
    for (size_t I = 0; I < Fn->Inputs.size(); ++I)
      Args.push_back(B.number(0.0));
    auto *Call = B.call(std::string(Fn->Name), std::move(Args));
    auto *Driver = AST.make<AssignStmt>();
    // Tier 3 — multi-return functions need one LHS per output for
    // the static -emit-* pipeline to refine ALL output types to
    // f64 (a single-LHS call keeps the trailing returns at `none`).
    // Use `[_p1, _p2, ..., _pK]` for stateful subsystems with K
    // returns; a single `__mflowlink_priming` keeps the stateless
    // case readable.
    if (Fn->Outputs.size() > 1) {
      for (size_t I = 0; I < Fn->Outputs.size(); ++I) {
        Driver->LHS.push_back(B.name("__mflowlink_priming" +
                                       std::to_string(I + 1)));
      }
    } else {
      Driver->LHS.push_back(B.name("__mflowlink_priming"));
    }
    Driver->RHS = Call;
    Driver->Suppressed = true;
    auto *S = AST.make<Script>();
    S->Body = AST.make<Block>();
    S->Body->Stmts.push_back(Driver);
    TU->ScriptNode = S;
  }
  return TU;
}

//===----------------------------------------------------------------------===//
// Tier 2 — subsystem metadata + per-target class-wrapper rendering.
//
// `describeSubsystem` recomputes the public surface (inputs, outputs,
// state slots) without touching the AST.  `emitSubsystemClassWrapper`
// renders a small target-specific class/struct that bundles the
// functional `step(...)` into the more ergonomic "class with mutating
// step(u) → y" idiom — matches the user's pick during planning (see
// docs/embedded_coder_roadmap.md §5).
//===----------------------------------------------------------------------===//

std::optional<SubsystemMeta> describeSubsystem(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::DiagnosticEngine &Diag) {
  const Flow *Sub = Doc.findFlow(SubsystemName);
  if (!Sub) {
    Diag.error(SourceLocation{}, "subsystem \"" + SubsystemName +
                                     "\" not found in `.mflow` file");
    return std::nullopt;
  }
  SubsystemMeta M;
  M.Name = sanitizeIdent(SubsystemName);

  // Inputs.
  auto Inports = collectPorts(*Sub, "signal_inport");
  for (auto &P : Inports) M.InputNames.push_back(P.Var);

  // Outputs.
  auto Outports = collectPorts(*Sub, "signal_outport");
  for (auto &P : Outports) M.OutputNames.push_back(P.Var);

  // Stateful blocks → one state slot each. Capture the per-block
  // initial condition so the class wrapper can default-init each
  // member field to the right value (matches the simulator's
  // t = 0 snapshot).
  for (const auto &N : Sub->Nodes) {
    // Tier-6 — nested subsystem state: enumerate the inner flow's
    // state slots and bubble them up under an `s_<outer_id>_<inner>`
    // prefix. Matches the lowerSubsystemImpl allocation so the
    // class wrapper instantiates member fields aligned with the
    // function signature.
    if (N.Kind == "signal_subsystem") {
      const std::string *FlowId = N.getData("flow_id");
      if (!FlowId || FlowId->empty()) continue;
      const Flow *Inner = findFlowById(Doc, *FlowId);
      if (!Inner) continue;
      auto InnerMeta = describeSubsystem(Doc, Inner->Name, Diag);
      if (!InnerMeta) continue;
      for (size_t I = 0; I < InnerMeta->StateArgNames.size(); ++I) {
        std::string Inner0 = InnerMeta->StateArgNames[I];
        std::string Suf = (Inner0.rfind("s_", 0) == 0)
                              ? Inner0.substr(2) : Inner0;
        std::string Base = "s_" + sanitizeIdent(N.Id) + "_" + Suf;
        M.StateArgNames.push_back(Base);
        M.StateReturnNames.push_back(Base + "_next");
        double Init = (I < InnerMeta->StateInitVals.size())
                          ? InnerMeta->StateInitVals[I] : 0.0;
        M.StateInitVals.push_back(Init);
      }
      continue;
    }
    if (!isStatefulKind(N.Kind)) continue;
    // Tier-5h — higher-order TF / ZP blocks contribute N slots,
    // named `s_<id>_x1` … `s_<id>_xN`. Single-slot blocks keep the
    // legacy `s_<id>` name.
    int Order = 1;
    if (N.Kind == "signal_transfer_fcn" || N.Kind == "signal_zero_pole") {
      std::vector<double> Num, Den;
      if (resolveTFCoeffs(N, Num, Den) && Den.size() >= 2)
        Order = (int)Den.size() - 1;
    } else if (N.Kind == "signal_transport_delay") {
      // Match the stateOrderForBlock logic in lowerSubsystemToMatlab.
      // describeSubsystem doesn't have access to SubsystemEmitOptions,
      // so use the per-block params + the flow's solver maxStep as
      // the fallback ladder. The CLI --target-rate override isn't
      // captured here, but the class wrapper still emits a state
      // field per slot — the count just may differ from the
      // actual emitted function signature when --target-rate
      // overrides the per-block sample period.  Acceptable for
      // the metadata view; the source-of-truth is the function
      // itself.
      auto getD = [&](const char *K) -> double {
        auto It = N.Params.find(K);
        if (It == N.Params.end()) return 0.0;
        try { return std::stod(It->second); } catch (...) { return 0.0; }
      };
      double Delay = getD("delay");
      double Ts = getD("sample_time");
      if (Ts <= 0.0) Ts = getD("sampleTime");
      if (Ts <= 0.0) Ts = getD("Ts");
      if (Ts <= 0.0 && Doc.Settings.Solver.has_value()) {
        const auto &SC = *Doc.Settings.Solver;
        if (SC.MaxStep != "auto") {
          try { Ts = std::stod(SC.MaxStep); } catch (...) {}
        }
      }
      if (Delay > 0.0 && Ts > 0.0) {
        int Taps = (int)std::round(Delay / Ts);
        if (Taps >= 1) Order = Taps;
      }
    } else if (N.Kind == "signal_state_space") {
      auto It = N.Params.find("A");
      if (It != N.Params.end()) {
        std::vector<double> A;
        int Ar = 0, Ac = 0;
        parseMatrixStr(It->second, A, Ar, Ac);
        if (Ar > 0 && Ar == Ac) Order = Ar;
      }
    }
    if (Order == 1) {
      M.StateArgNames.push_back("s_" + sanitizeIdent(N.Id));
      M.StateReturnNames.push_back("s_" + sanitizeIdent(N.Id) + "_next");
      M.StateInitVals.push_back(initialStateOf(N));
    } else {
      for (int K = 1; K <= Order; ++K) {
        std::string Base = "s_" + sanitizeIdent(N.Id) + "_x" +
                           std::to_string(K);
        M.StateArgNames.push_back(Base);
        M.StateReturnNames.push_back(Base + "_next");
        // No per-slot initial condition for higher-order TF — all
        // states zero by convention (matches the simulator's
        // controllable-canonical realisation at t=0).
        M.StateInitVals.push_back(0.0);
      }
    }
  }
  // Tier-6c — multirate counter. Mirror the lowerSubsystemImpl
  // detection: if any stateful block has a sample_time > base
  // period, the class wrapper carries a hidden `_tick` member
  // that initialises to 0 and threads through the multi-return.
  {
    auto getD = [&](const Node &N, const char *Key) -> double {
      auto It = N.Params.find(Key);
      if (It == N.Params.end()) return 0.0;
      try { return std::stod(It->second); } catch (...) { return 0.0; }
    };
    double Base = 0.0;
    for (const auto &N : Sub->Nodes) {
      if (!isStatefulKind(N.Kind)) continue;
      double Ts = getD(N, "sample_time");
      if (Ts <= 0.0) Ts = getD(N, "sampleTime");
      if (Ts <= 0.0) Ts = getD(N, "Ts");
      if (Ts > 0.0 && (Base <= 0.0 || Ts < Base)) Base = Ts;
    }
    if (Base <= 0.0 && Doc.Settings.Solver.has_value()) {
      const auto &SC = *Doc.Settings.Solver;
      if (SC.MaxStep != "auto") {
        try { Base = std::stod(SC.MaxStep); } catch (...) {}
      }
    }
    bool IsMultirate = false;
    if (Base > 0.0) {
      for (const auto &N : Sub->Nodes) {
        if (!isStatefulKind(N.Kind)) continue;
        double Ts = getD(N, "sample_time");
        if (Ts <= 0.0) Ts = getD(N, "sampleTime");
        if (Ts <= 0.0) Ts = getD(N, "Ts");
        if (Ts <= 0.0) continue;
        int E = (int)std::round(Ts / Base);
        if (E > 1) { IsMultirate = true; break; }
      }
    }
    if (IsMultirate) {
      M.StateArgNames.push_back("_tick");
      M.StateReturnNames.push_back("_tick_next");
      M.StateInitVals.push_back(0.0);
    }
  }
  return M;
}

namespace {

// Convert "snake_case" → "SnakeCase" so we get an idiomatic class
// name (Python: `DiscretePid`, C++: `DiscretePid`, …).
std::string toCamelCase(const std::string &S) {
  std::string Out;
  bool Up = true;
  for (char C : S) {
    if (C == '_' || C == '-') { Up = true; continue; }
    Out.push_back(Up ? std::toupper((unsigned char)C) : C);
    Up = false;
  }
  return Out;
}

std::string joinComma(const std::vector<std::string> &V) {
  std::string Out;
  for (size_t I = 0; I < V.size(); ++I) {
    if (I) Out += ", ";
    Out += V[I];
  }
  return Out;
}

} // namespace

std::string emitSubsystemClassWrapper(const SubsystemMeta &M,
                                       const std::string &Target) {
  std::string ClassName = toCamelCase(M.Name);
  std::ostringstream OS;
  bool Stateful = !M.StateArgNames.empty();
  size_t N = M.InputNames.size();   // public arg count
  size_t Outs = M.OutputNames.size(); // public return count

  // Tier 4 — render a state slot's default-init expression. Falls
  // back to 0.0 when the .mflow didn't capture an initial value
  // (early lowering paths may produce a SubsystemMeta without IC
  // info — older code paths).
  auto initFor = [&](size_t I) -> std::string {
    if (I >= M.StateInitVals.size()) return "0.0";
    std::ostringstream Tmp;
    Tmp.precision(17);
    Tmp << M.StateInitVals[I];
    return Tmp.str();
  };

  if (Target == "python") {
    OS << "\n\n";
    OS << "# Tier-2 class wrapper: object-style step() that carries\n";
    OS << "# the per-block state across calls.  Mirrors the functional\n";
    OS << "# `" << M.Name << "(...)` above; idiomatic for service / ML\n";
    OS << "# code that owns one controller instance per signal.\n";
    OS << "class " << ClassName << ":\n";
    OS << "    def __init__(self):\n";
    if (!Stateful) {
      OS << "        pass\n";
    } else {
      for (size_t I = 0; I < M.StateArgNames.size(); ++I)
        OS << "        self." << M.StateArgNames[I] << " = "
           << initFor(I) << "\n";
    }
    OS << "    def step(self";
    for (auto &In : M.InputNames) OS << ", " << In;
    OS << "):\n";
    OS << "        ";
    // LHS — y outputs + state-next.
    std::vector<std::string> Lhs;
    for (auto &Y : M.OutputNames) Lhs.push_back(Y);
    for (auto &S : M.StateReturnNames) Lhs.push_back(S);
    if (Lhs.size() == 1) OS << Lhs[0];
    else OS << joinComma(Lhs);
    OS << " = " << M.Name << "(";
    OS << joinComma(M.InputNames);
    for (auto &S : M.StateArgNames) OS << ", self." << S;
    OS << ")\n";
    // Latch state.
    for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
      OS << "        self." << M.StateArgNames[I] << " = "
         << M.StateReturnNames[I] << "\n";
    }
    if (Outs == 1) OS << "        return " << M.OutputNames[0] << "\n";
    else           OS << "        return " << joinComma(M.OutputNames)
                      << "\n";
    return OS.str();
  }

  if (Target == "cpp") {
    OS << "\n\n";
    OS << "// Tier-2 class wrapper: object-style step() that carries\n";
    OS << "// the per-block state across calls.\n";
    OS << "struct " << ClassName << " {\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I)
      OS << "  double " << M.StateArgNames[I] << " = "
         << initFor(I) << ";\n";
    if (Outs == 1) OS << "  double step(";
    else           OS << "  std::tuple<";
    if (Outs > 1) {
      for (size_t I = 0; I < Outs; ++I) {
        if (I) OS << ", ";
        OS << "double";
      }
      OS << "> step(";
    }
    for (size_t I = 0; I < N; ++I) {
      if (I) OS << ", ";
      OS << "double " << M.InputNames[I];
    }
    OS << ") {\n";
    if (Stateful || Outs > 1) {
      OS << "    ";
      OS << "auto _r = " << M.Name << "(";
      OS << joinComma(M.InputNames);
      for (auto &S : M.StateArgNames) OS << ", " << S;
      OS << ");\n";
      // Latch state: tuple elements after the y outputs.
      for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
        OS << "    " << M.StateArgNames[I] << " = std::get<"
           << (Outs + I) << ">(_r);\n";
      }
      if (Outs == 1) {
        OS << "    return std::get<0>(_r);\n";
      } else {
        // Pack the y outputs into a tuple to return.
        OS << "    return std::make_tuple(";
        for (size_t I = 0; I < Outs; ++I) {
          if (I) OS << ", ";
          OS << "std::get<" << I << ">(_r)";
        }
        OS << ");\n";
      }
    } else {
      // Stateless single-output — direct return.
      OS << "    return " << M.Name << "(" << joinComma(M.InputNames)
         << ");\n";
    }
    OS << "  }\n";
    OS << "};\n";
    return OS.str();
  }

  if (Target == "c") {
    // C: struct + free `<ClassName>_step` taking pointer-to-self.
    OS << "\n\n";
    OS << "// Tier-2 class wrapper (C): struct + free step function.\n";
    OS << "typedef struct {\n";
    if (M.StateArgNames.empty()) {
      OS << "  char _unused;  /* zero-state placeholder */\n";
    } else {
      for (auto &S : M.StateArgNames) OS << "  double " << S << ";\n";
    }
    OS << "} " << ClassName << ";\n";
    // Tier 4 — init helper that lays down the initial conditions
    // captured from the .mflow. Callers do `<Class> obj; <Class>_init(&obj);`
    // before the first step.
    if (Stateful) {
      OS << "static void " << ClassName << "_init(" << ClassName
         << " *self) {\n";
      for (size_t I = 0; I < M.StateArgNames.size(); ++I)
        OS << "  self->" << M.StateArgNames[I] << " = " << initFor(I)
           << ";\n";
      OS << "}\n";
    }
    if (Outs == 1) OS << "static double ";
    else           OS << "static void ";
    OS << ClassName << "_step(" << ClassName << " *self";
    for (auto &In : M.InputNames) OS << ", double " << In;
    if (Outs > 1) {
      for (auto &Y : M.OutputNames) OS << ", double *" << Y << "_out";
    }
    OS << ") {\n";
    // Allocate locals for the function's full set of returns.
    std::vector<std::string> AllOuts;
    for (auto &Y : M.OutputNames) AllOuts.push_back("y_" + Y);
    for (auto &S : M.StateReturnNames) AllOuts.push_back(S);
    for (auto &L : AllOuts) OS << "  double " << L << ";\n";
    OS << "  " << M.Name << "(";
    OS << joinComma(M.InputNames);
    for (auto &S : M.StateArgNames) OS << ", self->" << S;
    for (auto &L : AllOuts) OS << ", &" << L;
    OS << ");\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
      OS << "  self->" << M.StateArgNames[I] << " = "
         << M.StateReturnNames[I] << ";\n";
    }
    if (Outs == 1) {
      OS << "  return y_" << M.OutputNames[0] << ";\n";
    } else {
      for (auto &Y : M.OutputNames) {
        OS << "  *" << Y << "_out = y_" << Y << ";\n";
      }
    }
    OS << "}\n";
    return OS.str();
  }

  if (Target == "typescript") {
    OS << "\n\n";
    OS << "// Tier-2 class wrapper: object-style step() that carries\n";
    OS << "// the per-block state across calls.\n";
    OS << "class " << ClassName << " {\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I)
      OS << "  " << M.StateArgNames[I] << ": number = "
         << initFor(I) << ";\n";
    OS << "  step(";
    for (size_t I = 0; I < N; ++I) {
      if (I) OS << ", ";
      OS << M.InputNames[I] << ": number";
    }
    OS << "): ";
    if (Outs == 1) OS << "number";
    else {
      OS << "[";
      for (size_t I = 0; I < Outs; ++I) {
        if (I) OS << ", ";
        OS << "number";
      }
      OS << "]";
    }
    OS << " {\n";
    // Destructure the call result.
    OS << "    const [";
    std::vector<std::string> AllReturns;
    for (auto &Y : M.OutputNames) AllReturns.push_back(Y);
    for (auto &S : M.StateReturnNames) AllReturns.push_back(S);
    OS << joinComma(AllReturns);
    OS << "] = " << M.Name << "(";
    OS << joinComma(M.InputNames);
    for (auto &S : M.StateArgNames) OS << ", this." << S;
    OS << ");\n";
    for (size_t I = 0; I < M.StateArgNames.size(); ++I) {
      OS << "    this." << M.StateArgNames[I] << " = "
         << M.StateReturnNames[I] << ";\n";
    }
    if (Outs == 1) {
      OS << "    return " << M.OutputNames[0] << ";\n";
    } else {
      OS << "    return [";
      OS << joinComma(M.OutputNames);
      OS << "];\n";
    }
    OS << "  }\n";
    OS << "}\n";
    return OS.str();
  }

  // Unsupported target — return empty so the caller skips the
  // wrapper. SystemVerilog handles state via registers natively
  // (Tier 5) and doesn't go through this wrapper.
  return {};
}

//===----------------------------------------------------------------------===//
// Tier-7 — whole-diagram emit.
//
// Walks the entry flow (kind=program or function), categorises
// each block as source / sink / stateful internal / stateless
// internal, and synthesises a `simulate()` Function with:
//
//     function [<log_1>, <log_2>, ..., <log_M>, t_log] = simulate()
//         <state-var init>
//         <log array preallocation>
//         for k = 1 : N
//             t = (k - 1) * Ts;
//             <source generators consume t>
//             <internal-block dispatch — state read / compute / state update>
//             <sink logging — log_q(k) = upstream>
//         end
//     end
//
// Wraps the result in a TU + driver call so the downstream -emit-*
// lanes type-refine through. SystemVerilog whole-diagram is
// deliberately rejected here — the SV emit lane stays per-
// subsystem; whole-diagram simulation lives on the host.
//===----------------------------------------------------------------------===//

namespace {

bool isSourceKind(const std::string &K) {
  return K == "signal_sine" || K == "signal_step" ||
         K == "signal_ramp" || K == "signal_constant" ||
         K == "signal_pulse" || K == "signal_clock" ||
         K == "signal_chirp" ||
         K == "signal_repeating_sequence" ||
         K == "signal_function_call_generator";
}

bool isDiagramSink(const std::string &K) {
  return K == "signal_scope" || K == "signal_to_workspace" ||
         K == "signal_display" || K == "signal_terminator";
}

// Emit the source block's `<out> = expr(t)` at the per-tick site.
// `T` is the current-time variable name (always "t" in the
// emitted body). Returns an expression that reads as the source's
// instantaneous value at time `t`.
Expr *lowerSourceExpr(const Node &N, const std::string &TVar,
                       ASTBuilder &B) {
  const std::string &K = N.Kind;
  if (K == "signal_constant") {
    return B.number(paramD(N, "value", 0.0));
  }
  if (K == "signal_clock") {
    return B.name(TVar);
  }
  if (K == "signal_step") {
    // out = (t >= stepTime) ? finalValue : initialValue
    double ST = paramD(N, "stepTime", 1.0);
    double IV = paramD(N, "initialValue", 0.0);
    double FV = paramD(N, "finalValue", 1.0);
    auto *Cond = B.bin(BinOp::Ge, B.name(TVar), B.number(ST));
    auto *Diff = B.bin(BinOp::Sub, B.number(FV), B.number(IV));
    auto *Mul  = B.bin(BinOp::ElemMul, Cond, Diff);
    return B.bin(BinOp::Add, B.number(IV), Mul);
  }
  if (K == "signal_sine") {
    // out = amp * sin(freq * t + phase) + bias.  The simulator
    // treats `frequency` as angular ω (rad/s) — match that
    // convention exactly so the cosim diff stays tight.
    double A  = paramD(N, "amplitude", 1.0);
    double F  = paramD(N, "frequency", 1.0);
    double P  = paramD(N, "phase", 0.0);
    double Bs = paramD(N, "bias", 0.0);
    auto *Arg = B.bin(BinOp::ElemMul, B.number(F), B.name(TVar));
    Arg = B.bin(BinOp::Add, Arg, B.number(P));
    auto *Sin = B.call("sin", {Arg});
    auto *Scaled = B.bin(BinOp::ElemMul, B.number(A), Sin);
    return B.bin(BinOp::Add, Scaled, B.number(Bs));
  }
  if (K == "signal_ramp") {
    // out = (t >= start) ? init + slope * (t - start) : init
    double Slope = paramD(N, "slope", 1.0);
    double Start = paramD(N, "startTime", 0.0);
    double Init  = paramD(N, "initialOutput", 0.0);
    auto *Cond = B.bin(BinOp::Ge, B.name(TVar), B.number(Start));
    auto *Dt = B.bin(BinOp::Sub, B.name(TVar), B.number(Start));
    auto *Slope_Dt = B.bin(BinOp::ElemMul, B.number(Slope), Dt);
    auto *Mul = B.bin(BinOp::ElemMul, Cond, Slope_Dt);
    return B.bin(BinOp::Add, B.number(Init), Mul);
  }
  if (K == "signal_pulse") {
    // out = (mod(t - phase, period) < period * width / 100) ? amp : 0
    double A   = paramD(N, "amplitude", 1.0);
    double Pp  = paramD(N, "period", 1.0);
    double W   = paramD(N, "pulseWidth", 50.0);
    double Phd = paramD(N, "phaseDelay", 0.0);
    auto *Shifted = B.bin(BinOp::Sub, B.name(TVar), B.number(Phd));
    auto *Mod = B.call("mod", {Shifted, B.number(Pp)});
    auto *Cond = B.bin(BinOp::Lt, Mod, B.number(Pp * W * 0.01));
    return B.bin(BinOp::ElemMul, Cond, B.number(A));
  }
  if (K == "signal_chirp") {
    // Linear frequency sweep from f0 to f1 over the sweep period T:
    //   out = amp * sin(2π * (f0 + (f1 - f0) * t / (2*T)) * t + phase)
    // The factor (f1 - f0)/(2T) is the instantaneous-frequency slope
    // / 2; integrating ω(t) = 2π*(f0 + (f1-f0)*t/T) gives the
    // argument above. `targetFrequency` + `targetTime` match the
    // Simulink Chirp block's parameter names.
    double A  = paramD(N, "amplitude", 1.0);
    double F0 = paramD(N, "initialFrequency", 0.1);
    double F1 = paramD(N, "targetFrequency", 1.0);
    double Tt = paramD(N, "targetTime", 1.0);
    double P  = paramD(N, "phase", 0.0);
    double Pi2 = 2.0 * 3.141592653589793238;
    if (Tt <= 0.0) Tt = 1.0;
    // Inner = (f0 + (f1-f0)*t/(2*Tt)) * t
    auto *Slope = B.bin(BinOp::ElemMul, B.number((F1 - F0) / (2.0 * Tt)),
                         B.name(TVar));
    auto *Freq  = B.bin(BinOp::Add, B.number(F0), Slope);
    auto *Inner = B.bin(BinOp::ElemMul, Freq, B.name(TVar));
    auto *Arg   = B.bin(BinOp::ElemMul, B.number(Pi2), Inner);
    Arg = B.bin(BinOp::Add, Arg, B.number(P));
    auto *Sin = B.call("sin", {Arg});
    return B.bin(BinOp::ElemMul, B.number(A), Sin);
  }
  if (K == "signal_repeating_sequence") {
    // LUT-driven periodic sequence: piecewise-linear interpolation
    // through (timeValues, outputValues), looping with period
    // (timeValues[end] - timeValues[0]). For the MVP we emit the
    // simplest shape — a sawtooth from outputValues[0] to
    // outputValues[end] over [0, period]:
    //   out = out0 + (out1 - out0) * mod(t, period) / period
    // Multi-segment interpolation is a follow-up (needs piecewise
    // emission). The user can encode arbitrary periodic signals
    // with two points covering the linear segment.
    auto It = N.Params.find("outputValues");
    auto Tt = N.Params.find("timeValues");
    double Out0 = 0.0, Out1 = 1.0, T0 = 0.0, T1 = 1.0;
    if (It != N.Params.end()) {
      std::vector<double> O; int Or = 0, Oc = 0;
      parseMatrixStr(It->second, O, Or, Oc);
      if (!O.empty()) Out0 = O.front();
      if (O.size() >= 2) Out1 = O.back();
    }
    if (Tt != N.Params.end()) {
      std::vector<double> T; int Tr = 0, Tc = 0;
      parseMatrixStr(Tt->second, T, Tr, Tc);
      if (!T.empty()) T0 = T.front();
      if (T.size() >= 2) T1 = T.back();
    }
    double Period = T1 - T0;
    if (Period <= 0.0) Period = 1.0;
    auto *Shifted = B.bin(BinOp::Sub, B.name(TVar), B.number(T0));
    auto *Mod = B.call("mod", {Shifted, B.number(Period)});
    auto *Slope = B.number((Out1 - Out0) / Period);
    auto *Lin = B.bin(BinOp::ElemMul, Slope, Mod);
    return B.bin(BinOp::Add, B.number(Out0), Lin);
  }
  // Default — emit 0.0.  Unknown source kinds (function_call_generator,
  // band-limited noise, etc.) lower as constant zero; the diagnostic
  // path catches this at the dispatch layer below.
  return B.number(0.0);
}

} // namespace

matlab::TranslationUnit *buildDiagramTU(
    const FlowDoc &Doc,
    const std::string &EntryFlowName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts) {
  // HDL whole-diagram is out of scope — sources/sinks are host-
  // side; SV stays per-subsystem.
  if (Opts.StateAsPersistent) {
    Diag.error(SourceLocation{},
               "whole-diagram emit doesn't support SystemVerilog "
               "(Tier-7 carve-out): use `--subsystem <name>` to "
               "emit a single subsystem instead");
    return nullptr;
  }

  const Flow *Entry = Doc.findFlow(EntryFlowName);
  if (!Entry) {
    Diag.error(SourceLocation{},
               "flow \"" + EntryFlowName +
                   "\" not found in `.mflow` document");
    return nullptr;
  }
  // Tick-count + period resolution.  Priority order:
  //   1. SubsystemEmitOptions.TargetRate / explicit --ticks (carved
  //      out of CLI surface — falls back to settings.solver).
  //   2. settings.solver.maxStep + stopTime - startTime.
  //   3. Default Ts = 0.01, N = 100 (so cold-start emits still run).
  double Ts = Opts.TargetRate;
  double Tstart = 0.0, Tstop = 0.0;
  if (Doc.Settings.Solver.has_value()) {
    const auto &SC = *Doc.Settings.Solver;
    if (Ts <= 0.0 && SC.MaxStep != "auto") {
      try { Ts = std::stod(SC.MaxStep); } catch (...) {}
    }
    Tstart = SC.StartTime;
    Tstop  = SC.StopTime;
  }
  if (Ts <= 0.0) Ts = 0.01;
  double Span = Tstop - Tstart;
  int NTicks = Span > 0.0 ? (int)std::round(Span / Ts) : 100;
  if (NTicks < 1) NTicks = 1;
  // Tier-7e: `--ticks <N>` CLI override beats the solver-derived
  // count. Useful for short smoke runs of a long-stopTime model
  // without editing the .mflow.
  if (Opts.TickCount > 0) NTicks = Opts.TickCount;
  // Log decimation — emit one log entry every `Decim` ticks (default
  // 1 = no decimation). The log array's length matches the count
  // of emitted entries (ceil(NTicks / Decim)).
  int Decim = Opts.LogDecimation > 0 ? Opts.LogDecimation : 1;
  int LogLen = (NTicks + Decim - 1) / Decim;

  ASTBuilder B{AST};
  // Sema-friendly: leave WrapFi off for whole-diagram emit; the
  // function operates in f64 throughout (no HDL).
  EdgeIndex EI = buildEdgeIndex(*Entry);

  // Categorise blocks.
  std::vector<const Node *> Sources, Sinks, Internal;
  for (const auto &N : Entry->Nodes) {
    if (N.Kind == "signal_inport" || N.Kind == "signal_outport")
      continue;  // boundary tags ignored at whole-diagram level
    if (isSourceKind(N.Kind)) { Sources.push_back(&N); continue; }
    if (isDiagramSink(N.Kind)) { Sinks.push_back(&N); continue; }
    Internal.push_back(&N);
  }

  // Variable allocation: one local per block output; one log
  // array per scope/to_workspace sink (display/terminator drop
  // their input). State slots for stateful internal blocks
  // declared as local vars updated in place (no return-thread).
  std::set<std::string> Used{"t", "k", "Ts", "N", "simulate"};
  // Pre-reserve every helper function name (nested subsystem flows
  // referenced from the entry flow). Without this, a block id whose
  // sanitised name matches the helper's function name would shadow
  // the call site — emitting `pi_ctrl = pi_ctrl(...)` looks like
  // self-indexing to Sema, not a function call.
  for (const auto &N : Entry->Nodes) {
    if (N.Kind != "signal_subsystem") continue;
    const std::string *FlowId = N.getData("flow_id");
    if (!FlowId || FlowId->empty()) continue;
    const Flow *Inner = findFlowById(Doc, *FlowId);
    if (Inner) Used.insert(sanitizeIdent(Inner->Name));
  }
  auto uniqueName = [&](const std::string &Base) {
    std::string Cand = sanitizeIdent(Base);
    if (Cand.empty()) Cand = "v";
    std::string Name = Cand;
    int Suffix = 1;
    while (Used.count(Name))
      Name = Cand + "_" + std::to_string(++Suffix);
    Used.insert(Name);
    return Name;
  };
  std::unordered_map<std::string, std::string> VarOfNode;
  for (const auto *N : Sources)  VarOfNode[N->Id] = uniqueName(N->Id);
  for (const auto *N : Internal) VarOfNode[N->Id] = uniqueName(N->Id);

  // Sinks logged per-block: log_<id> is the array name.  The
  // function's return list is the sink columns in source order
  // followed by the `t_log` column (always last).
  struct SinkLog {
    const Node *N;
    std::string Var;       // log_<id>
    std::string Source;    // upstream block id
    std::string SourcePort;
  };
  std::vector<SinkLog> Logs;
  for (const auto *N : Sinks) {
    if (N->Kind == "signal_terminator") continue;  // drop
    SinkLog L;
    L.N = N;
    L.Var = uniqueName("log_" + N->Id);
    // Find the upstream feeding the sink's first input port.
    for (const auto &P : N->InPorts) {
      auto It = EI.Map.find({N->Id, P.Id});
      if (It != EI.Map.end()) {
        L.Source = It->second.first;
        L.SourcePort = It->second.second;
        break;
      }
    }
    Logs.push_back(L);
  }
  std::string TLog = uniqueName("t_log");

  // State for stateful internal blocks. Two paths:
  //
  //   * **Inline (single-slot)** — Unit Delay / ZOH / discrete
  //     integrator / 1st-order TF stay inline as a single local var
  //     `s_<id>` updated each tick with the per-kind next-state
  //     expression. Keeps the emitted body compact for the common
  //     case.
  //   * **Helper-bound (Tier-7 multi-slot + nested)** — every
  //     `signal_subsystem` and every higher-order stateful primitive
  //     (TF/ZP order ≥ 2 / state-space / transport-delay) is lowered
  //     via `lowerSubsystemImpl` into a helper Function (the
  //     primitive case wraps the block in a synthesised single-
  //     block subsystem flow first). The diagram body emits a call
  //     site `[<outs>, <sn_next>] = helper(<ins>, <sn>)` and
  //     latches each state slot from its next-state local. This
  //     pulls the per-subsystem emit's full state-space / Tustin /
  //     transport-delay machinery into the whole-diagram lane
  //     without duplicating ~200 lines of state-update emission.
  struct LocalStateSlot {
    const Node *N;
    std::string Var;      // current state local
    double Init;          // initial value (from block's IC or 0)
    int Epoch = 1;        // multirate epoch
    std::string PhaseVar; // optional per-block phase counter
  };
  std::vector<LocalStateSlot> States;
  struct HelperBinding {
    const Node *N;
    Function *Fn = nullptr;
    SubsystemMeta Meta;
    // Per-slot allocated current-state local (size = Meta.StateArgNames.size()).
    std::vector<std::string> StateVars;
    // Per-slot next-state local captured from the call return.
    std::vector<std::string> StateNextVars;
    // Per-port output local (size = Meta.OutputNames.size()).
    std::vector<std::string> OutVars;
  };
  std::map<std::string, HelperBinding> Helpers;

  auto stateOrderForBlock = [&](const Node *N) -> int {
    if (N->Kind == "signal_transfer_fcn" ||
        N->Kind == "signal_zero_pole") {
      std::vector<double> Num, Den;
      if (!resolveTFCoeffs(*N, Num, Den)) return 1;
      if (Den.size() < 2) return 1;
      return (int)Den.size() - 1;
    }
    if (N->Kind == "signal_state_space") {
      auto It = N->Params.find("A");
      if (It == N->Params.end()) return 1;
      std::vector<double> A; int Ar = 0, Ac = 0;
      parseMatrixStr(It->second, A, Ar, Ac);
      return (Ar > 0 && Ar == Ac) ? Ar : 1;
    }
    if (N->Kind == "signal_transport_delay") {
      // Same heuristic the per-subsystem lane uses.
      double Delay = paramD(*N, "delay", 0.0);
      double LocalTs = paramD(*N, "sample_time", 0.0);
      if (LocalTs <= 0.0) LocalTs = paramD(*N, "sampleTime", 0.0);
      if (LocalTs <= 0.0) LocalTs = paramD(*N, "Ts", 0.0);
      if (LocalTs <= 0.0) LocalTs = Ts;
      if (Delay > 0.0 && LocalTs > 0.0) {
        int Taps = (int)std::round(Delay / LocalTs);
        if (Taps >= 1) return Taps;
      }
    }
    return 1;
  };

  // Local FlowDoc copy so we can append synthesised wrapper flows
  // without mutating the caller's Doc. FlowDoc is trivially copyable
  // (just vectors / maps / strings under the hood).
  FlowDoc DocLocal = Doc;
  NestedCtx Ctx;

  // Synthesise a single-block wrapper flow around the given primitive
  // block. The resulting flow has boundary `signal_inport`s named
  // `u1`..`uP`, the original block (params + ports preserved), and
  // boundary `signal_outport`s named `y1`..`yQ`. Edges hook them up
  // by the block's declared port ids.
  auto buildWrapperFlow = [&](const Node *N) -> std::string {
    Flow F;
    F.Name = EntryFlowName + "__" + sanitizeIdent(N->Id) + "_helper";
    F.Id   = "synth_" + F.Name;
    F.Kind = "function";
    auto mkInport = [&](int Idx) {
      Node In;
      In.Id = "u" + std::to_string(Idx);
      In.Kind = "signal_inport";
      Port OutP; OutP.Id = "out";
      In.OutPorts.push_back(OutP);
      return In;
    };
    auto mkOutport = [&](int Idx) {
      Node Out;
      Out.Id = "y" + std::to_string(Idx);
      Out.Kind = "signal_outport";
      Port InP; InP.Id = "in";
      Out.InPorts.push_back(InP);
      return Out;
    };
    for (size_t I = 0; I < N->InPorts.size(); ++I)
      F.Nodes.push_back(mkInport((int)I + 1));
    F.Nodes.push_back(*N);
    for (size_t I = 0; I < N->OutPorts.size(); ++I)
      F.Nodes.push_back(mkOutport((int)I + 1));
    for (size_t I = 0; I < N->InPorts.size(); ++I) {
      Edge E;
      E.Id   = "synth_in_" + std::to_string(I + 1);
      E.Kind = "data";
      E.From.Node = "u" + std::to_string(I + 1);
      E.From.Port = "out";
      E.To.Node = N->Id;
      E.To.Port = N->InPorts[I].Id;
      F.Edges.push_back(E);
    }
    for (size_t I = 0; I < N->OutPorts.size(); ++I) {
      Edge E;
      E.Id   = "synth_out_" + std::to_string(I + 1);
      E.Kind = "data";
      E.From.Node = N->Id;
      E.From.Port = N->OutPorts[I].Id;
      E.To.Node = "y" + std::to_string(I + 1);
      E.To.Port = "in";
      F.Edges.push_back(E);
    }
    DocLocal.Flows.push_back(F);
    return F.Name;
  };

  // Pre-pass: bind helpers for every nested subsystem and every
  // multi-slot stateful primitive.
  for (auto *N : Internal) {
    std::string HelperFlowName;
    if (N->Kind == "signal_subsystem") {
      const std::string *FlowId = N->getData("flow_id");
      if (!FlowId || FlowId->empty()) {
        Diag.error(N->Loc,
                   "signal_subsystem \"" + N->Id +
                       "\": missing data.flow_id (embedded coder "
                       "needs an explicit subflow reference)");
        return nullptr;
      }
      const Flow *Inner = findFlowById(DocLocal, *FlowId);
      if (!Inner) {
        Diag.error(N->Loc,
                   "signal_subsystem \"" + N->Id +
                       "\": data.flow_id \"" + *FlowId +
                       "\" not found in `.mflow` document");
        return nullptr;
      }
      HelperFlowName = Inner->Name;
    } else if (isStatefulKind(N->Kind) && stateOrderForBlock(N) > 1) {
      HelperFlowName = buildWrapperFlow(N);
    } else {
      continue; // single-slot stateful + stateless handled inline
    }
    auto Hit = Ctx.ByFlowId.find(HelperFlowName);
    Function *Fn = nullptr;
    SubsystemMeta Meta;
    if (Hit != Ctx.ByFlowId.end()) {
      // Same flow already lowered (multiple instances of the same
      // nested subsystem). Reuse the cached function + meta.
      Fn = Hit->second.first;
      Meta = Hit->second.second;
    } else {
      Fn = lowerSubsystemImpl(DocLocal, HelperFlowName, AST, Diag,
                               Opts, Ctx);
      if (!Fn) return nullptr;
      auto MetaOpt = describeSubsystem(DocLocal, HelperFlowName, Diag);
      if (!MetaOpt) return nullptr;
      MetaOpt->Name = HelperFlowName;
      Meta = *MetaOpt;
      Ctx.ByFlowId[HelperFlowName] = {Fn, Meta};
      Ctx.Pending.push_back(Fn);
    }

    HelperBinding HB;
    HB.N = N;
    HB.Fn = Fn;
    HB.Meta = Meta;
    for (size_t I = 0; I < HB.Meta.StateArgNames.size(); ++I) {
      HB.StateVars.push_back(
          uniqueName(N->Id + "_" + HB.Meta.StateArgNames[I]));
      HB.StateNextVars.push_back(
          uniqueName(N->Id + "_" + HB.Meta.StateReturnNames[I]));
    }
    if (HB.Meta.OutputNames.size() <= 1) {
      HB.OutVars.push_back(VarOfNode[N->Id]);
    } else {
      for (size_t I = 0; I < HB.Meta.OutputNames.size(); ++I)
        HB.OutVars.push_back(
            uniqueName(N->Id + "_" + HB.Meta.OutputNames[I]));
    }
    Helpers[N->Id] = std::move(HB);
  }

  // Per-port output map for multi-output helpers — downstream
  // consumers resolve their input through this when the upstream
  // is a multi-output helper. Keyed by (blockId, portId).
  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> &P) const {
      return std::hash<std::string>()(P.first) ^
             (std::hash<std::string>()(P.second) << 1);
    }
  };
  std::unordered_map<std::pair<std::string, std::string>, std::string,
                     PairHash> VarOfNodePort;
  for (auto &[Id, HB] : Helpers) {
    if (HB.OutVars.size() <= 1) continue;
    for (size_t I = 0;
         I < HB.N->OutPorts.size() && I < HB.OutVars.size(); ++I) {
      VarOfNodePort[{HB.N->Id, HB.N->OutPorts[I].Id}] = HB.OutVars[I];
    }
  }

  auto Body = AST.make<Block>();

  // Declare Ts and N as constants at function start.
  Body->Stmts.push_back(B.assign("Ts", B.number(Ts)));
  Body->Stmts.push_back(B.assign("N", B.integer(NTicks)));
  // Tier-7e: when --decimation > 1, sink logs hold one entry per
  // `Decim` ticks. `LogN` is the log-array length; `kd` (declared
  // below) is the decimated write index that advances by 1 every
  // `Decim` simulation ticks. Plain Decim == 1 falls through to
  // the historical "log every tick" shape (kd == k).
  bool Decimate = Decim > 1;
  if (Decimate) {
    Body->Stmts.push_back(B.assign("Decim", B.integer(Decim)));
    Body->Stmts.push_back(B.assign("LogN", B.integer(LogLen)));
    Body->Stmts.push_back(B.assign("kd", B.integer(0)));
  }
  // Pre-allocate log arrays as zeros(1, LogN) (or 1, N when no
  // decimation).
  auto preallocZeros = [&](const std::string &Var) {
    auto *Call = B.call(
        "zeros", {B.integer(1), B.name(Decimate ? "LogN" : "N")});
    Body->Stmts.push_back(B.assign(Var, Call));
  };
  for (const auto &L : Logs) preallocZeros(L.Var);
  preallocZeros(TLog);

  // Initialise inline single-slot state vars.
  for (const auto *N : Internal) {
    if (Helpers.count(N->Id)) continue;     // handled via helper
    if (!isStatefulKind(N->Kind)) continue;
    LocalStateSlot S;
    S.N = N;
    S.Var = uniqueName("s_" + N->Id);
    S.Init = initialStateOf(*N);
    // Multirate per-block epoch (reused logic from the per-
    // subsystem lane). Base rate = the loop's Ts.
    double BlockTs = paramD(*N, "sample_time", 0.0);
    if (BlockTs <= 0.0) BlockTs = paramD(*N, "sampleTime", 0.0);
    if (BlockTs <= 0.0) BlockTs = paramD(*N, "Ts", 0.0);
    if (BlockTs > 0.0) {
      int E = (int)std::round(BlockTs / Ts);
      if (E > 1) {
        S.Epoch = E;
        S.PhaseVar = uniqueName("phase_" + N->Id);
      }
    }
    States.push_back(S);
    Body->Stmts.push_back(B.assign(S.Var, B.number(S.Init)));
    if (!S.PhaseVar.empty())
      Body->Stmts.push_back(B.assign(S.PhaseVar, B.integer(0)));
  }
  // Initialise helper-bound state vars.
  for (auto &[Id, HB] : Helpers) {
    for (size_t I = 0; I < HB.StateVars.size(); ++I) {
      double Init = (I < HB.Meta.StateInitVals.size())
                        ? HB.Meta.StateInitVals[I] : 0.0;
      Body->Stmts.push_back(B.assign(HB.StateVars[I], B.number(Init)));
    }
  }

  // Build the for-loop body (per-tick computations).
  auto *Loop = AST.make<Block>();
  // t = (k - 1) * Ts
  Loop->Stmts.push_back(B.assign(
      "t", B.bin(BinOp::ElemMul,
                  B.bin(BinOp::Sub, B.name("k"), B.integer(1)),
                  B.name("Ts"))));

  // Source generators.
  for (const auto *N : Sources) {
    Expr *Val = lowerSourceExpr(*N, "t", B);
    Loop->Stmts.push_back(B.assign(VarOfNode[N->Id], Val));
  }

  // Internal blocks — toposort first so consumers come after
  // producers within the loop body.
  auto Topo = toposortInternals(*Entry, Diag, EntryFlowName);
  if (Diag.hasErrors()) return nullptr;
  std::vector<const Node *> TopoInternal;
  for (auto *N : Topo) {
    if (isSourceKind(N->Kind) || isDiagramSink(N->Kind)) continue;
    TopoInternal.push_back(N);
  }

  // Helper: resolve an input port's upstream value as an Expr.
  // Consults VarOfNodePort first so multi-output helpers' per-port
  // outputs are routed correctly; falls back to the legacy single
  // VarOfNode[id] for blocks with a single output local.
  auto resolveIn = [&](const std::string &ToNode,
                        const std::string &ToPort) -> Expr * {
    auto It = EI.Map.find({ToNode, ToPort});
    if (It == EI.Map.end()) return B.number(0.0);
    const std::string &Up = It->second.first;
    const std::string &UpPort = It->second.second;
    auto VPIt = VarOfNodePort.find({Up, UpPort});
    if (VPIt != VarOfNodePort.end()) return B.name(VPIt->second);
    auto VarIt = VarOfNode.find(Up);
    if (VarIt == VarOfNode.end()) return B.number(0.0);
    return B.name(VarIt->second);
  };
  auto inputPortsOf = [&](const Node &N) -> std::vector<std::string> {
    std::vector<std::pair<int, std::string>> Pairs;
    for (const auto &P : N.InPorts) {
      int Idx = parsePortIndex(P.Id);
      if (Idx < 0) Idx = (int)Pairs.size() + 1;
      Pairs.push_back({Idx, P.Id});
    }
    std::sort(Pairs.begin(), Pairs.end());
    std::vector<std::string> Out;
    for (auto &PR : Pairs) Out.push_back(PR.second);
    return Out;
  };

  // Loop-breaker hoist: every inline single-slot stateful block's
  // OutVar must be assigned from the state slot BEFORE any consumer
  // evaluates. Mirrors `lowerSubsystemImpl`'s state-read hoist —
  // keeps the topo's drop-outgoing-edges-of-stateful-blocks
  // semantics consistent at the per-tick emit site. Helper-bound
  // blocks don't need this — their output materialises from the
  // function call return (which is emitted at the block's topo
  // position).
  for (auto &S : States) {
    Loop->Stmts.push_back(
        B.assign(VarOfNode[S.N->Id], B.name(S.Var)));
  }

  for (const auto *N : TopoInternal) {
    const std::string &OutVar = VarOfNode[N->Id];
    auto Ports = inputPortsOf(*N);
    std::vector<Expr *> Ins;
    for (auto &P : Ports) Ins.push_back(resolveIn(N->Id, P));

    // Helper-bound block: emit a function call to the helper plus
    // a state-latch tail (`s = s_next`).
    auto HIt = Helpers.find(N->Id);
    if (HIt != Helpers.end()) {
      auto &HB = HIt->second;
      // Build call args: public inputs in port order, then state
      // args. Public inputs come from upstream wiring (via Ins);
      // state args are the per-slot StateVars allocated above.
      std::vector<Expr *> Args;
      size_t NumPubIn = HB.Meta.InputNames.size();
      for (size_t I = 0; I < NumPubIn; ++I) {
        Expr *A = (I < Ins.size()) ? Ins[I] : B.number(0.0);
        Args.push_back(A);
      }
      for (auto &SV : HB.StateVars) Args.push_back(B.name(SV));
      auto *Call = B.call(HB.Meta.Name, std::move(Args));
      size_t NumYs = HB.Meta.OutputNames.size();
      size_t NumSnext = HB.Meta.StateReturnNames.size();
      if (NumYs <= 1 && NumSnext == 0) {
        // Plain single-output stateless subsystem helper.
        std::string Dst = HB.OutVars.empty() ? OutVar : HB.OutVars[0];
        Loop->Stmts.push_back(B.assign(Dst, Call));
      } else {
        auto *Assign = AST.make<AssignStmt>();
        for (size_t I = 0; I < HB.OutVars.size(); ++I)
          Assign->LHS.push_back(B.name(HB.OutVars[I]));
        for (size_t I = 0; I < HB.StateNextVars.size(); ++I)
          Assign->LHS.push_back(B.name(HB.StateNextVars[I]));
        Assign->RHS = Call;
        Assign->Suppressed = true;
        Loop->Stmts.push_back(Assign);
      }
      // State latch.
      for (size_t I = 0; I < HB.StateVars.size(); ++I)
        Loop->Stmts.push_back(
            B.assign(HB.StateVars[I], B.name(HB.StateNextVars[I])));
      continue;
    }

    // Find the state slot (if any) for this block.
    LocalStateSlot *Slot = nullptr;
    for (auto &S : States) if (S.N == N) { Slot = &S; break; }

    if (Slot) {
      // Stateful single-slot block — state-read already hoisted
      // above; here we only compute the next state and gate on
      // the epoch counter for multirate.
      Expr *NextExpr = nullptr;
      if (N->Kind == "signal_unit_delay" || N->Kind == "signal_zoh") {
        NextExpr = Ins.empty() ? B.number(0.0) : Ins.front();
      } else if (N->Kind == "signal_dff") {
        // #343 D flip-flop — Q_next = D (data port; `clk` is the module clock).
        Expr *D = nullptr;
        for (size_t pi = 0; pi < Ports.size() && pi < Ins.size(); ++pi)
          if (Ports[pi] == "d" || Ports[pi] == "in" || Ports[pi] == "in1") {
            D = Ins[pi];
            break;
          }
        NextExpr = D ? D : (Ins.empty() ? B.number(0.0) : Ins.front());
      } else if (N->Kind == "signal_discrete_integrator" ||
                 N->Kind == "signal_integrator") {
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        // Forward-Euler integrator: s_next = s + Ts * u.  Honour
        // the per-block sample period when set, else use the
        // outer base rate.
        double LocalTs = paramD(*N, "sample_time", 0.0);
        if (LocalTs <= 0.0) LocalTs = paramD(*N, "sampleTime", Ts);
        Expr *Step = B.bin(BinOp::ElemMul, B.number(LocalTs), U);
        NextExpr = B.bin(BinOp::Add, B.name(Slot->Var), Step);
      } else if (N->Kind == "signal_transfer_fcn" ||
                 N->Kind == "signal_zero_pole") {
        // 1st-order strictly-proper: H(s) = b0 / (a1 s + a0).
        // Forward Euler: x_next = x + Ts*(b0/a1*u - a0/a1*x).
        std::vector<double> Num, Den;
        resolveTFCoeffs(*N, Num, Den);
        double Lead = Den.empty() ? 1.0 : Den.front();
        double A0 = Den.size() >= 2 ? Den[1] / Lead : 0.0;
        double B0 = !Num.empty() ? Num.front() / Lead : 1.0;
        Expr *U = Ins.empty() ? B.number(0.0) : Ins.front();
        Expr *MinusA0 = B.bin(BinOp::ElemMul,
                               B.number(-A0), B.name(Slot->Var));
        Expr *B0u = B.bin(BinOp::ElemMul, B.number(B0), U);
        Expr *Acc = B.bin(BinOp::Add, B0u, MinusA0);
        Expr *Step = B.bin(BinOp::ElemMul, B.number(Ts), Acc);
        NextExpr = B.bin(BinOp::Add, B.name(Slot->Var), Step);
      } else {
        NextExpr = B.name(Slot->Var);  // no-op
      }
      // Multirate gating: only update state when phase == 0.
      if (Slot->Epoch > 1) {
        auto *If = AST.make<IfStmt>();
        If->Cond = B.bin(BinOp::Eq, B.name(Slot->PhaseVar),
                          B.integer(0));
        If->Then = AST.make<Block>();
        If->Then->Stmts.push_back(B.assign(Slot->Var, NextExpr));
        Loop->Stmts.push_back(If);
        // Phase advance: phase = (phase == epoch - 1) ? 0 : phase + 1
        auto *PhaseIf = AST.make<IfStmt>();
        PhaseIf->Cond = B.bin(BinOp::Eq, B.name(Slot->PhaseVar),
                               B.integer(Slot->Epoch - 1));
        PhaseIf->Then = AST.make<Block>();
        PhaseIf->Then->Stmts.push_back(
            B.assign(Slot->PhaseVar, B.integer(0)));
        PhaseIf->Else = AST.make<Block>();
        PhaseIf->Else->Stmts.push_back(B.assign(
            Slot->PhaseVar,
            B.bin(BinOp::Add, B.name(Slot->PhaseVar), B.integer(1))));
        Loop->Stmts.push_back(PhaseIf);
      } else {
        Loop->Stmts.push_back(B.assign(Slot->Var, NextExpr));
      }
      continue;
    }

    // Stateless block — fall back to the per-block lowering
    // shared with the per-subsystem emit.
    auto *Stmt = lowerBlock(*N, OutVar, Ins, Ports, B, Diag);
    if (!Stmt) return nullptr;
    Loop->Stmts.push_back(Stmt);
  }

  // Sink logging — append the upstream's current value to the
  // log column. When decimation is on, gate writes on
  // `mod(k - 1, Decim) == 0` and use a separate decimated write
  // index `kd` that advances by 1 each fire. Consult
  // VarOfNodePort first so multi-output helpers route through
  // their per-port locals; fall back to VarOfNode for the single-
  // output case.
  Block *LogBlock = Loop;
  IfStmt *DecimIf = nullptr;
  if (Decimate) {
    DecimIf = AST.make<IfStmt>();
    // The loop unroller folds k to a constant per iteration but
    // doesn't re-flow that constant into kd, so compute kd as a
    // stateful counter incremented INSIDE the gate. kd is
    // initialised to 0 in the body prologue (see emit just below
    // the LogN setup) and bumped by 1 each fire.
    auto *KMinus1 = B.bin(BinOp::Sub, B.name("k"), B.integer(1));
    DecimIf->Cond = B.bin(BinOp::Eq,
                           B.call("mod", {KMinus1, B.name("Decim")}),
                           B.integer(0));
    DecimIf->Then = AST.make<Block>();
    LogBlock = DecimIf->Then;
    LogBlock->Stmts.push_back(B.assign(
        "kd", B.bin(BinOp::Add, B.name("kd"), B.integer(1))));
  }
  std::string LogIdx = Decimate ? std::string("kd") : std::string("k");
  for (const auto &L : Logs) {
    Expr *Src;
    if (L.Source.empty()) {
      Src = B.number(0.0);
    } else {
      auto VPIt = VarOfNodePort.find({L.Source, L.SourcePort});
      if (VPIt != VarOfNodePort.end()) {
        Src = B.name(VPIt->second);
      } else if (VarOfNode.count(L.Source)) {
        Src = B.name(VarOfNode[L.Source]);
      } else {
        Src = B.number(0.0);
      }
    }
    auto *Idx = B.call(L.Var, {B.name(LogIdx)});
    auto *Assign = AST.make<AssignStmt>();
    Assign->LHS.push_back(Idx);
    Assign->RHS = Src;
    Assign->Suppressed = true;
    LogBlock->Stmts.push_back(Assign);
  }
  // t_log column. Decimated and undecimated emits both use the same
  // index variable so all log columns stay aligned.
  {
    auto *Idx = B.call(TLog, {B.name(LogIdx)});
    auto *Assign = AST.make<AssignStmt>();
    Assign->LHS.push_back(Idx);
    Assign->RHS = B.name("t");
    Assign->Suppressed = true;
    LogBlock->Stmts.push_back(Assign);
  }
  if (Decimate) Loop->Stmts.push_back(DecimIf);

  // Wrap the loop body in `for k = 1 : N`.
  auto *For = AST.make<ForStmt>();
  For->Var = AST.intern("k");
  auto *Range = AST.make<RangeExpr>();
  Range->Start = B.integer(1);
  Range->End   = B.name("N");
  For->Iter = Range;
  For->Body = Loop;
  Body->Stmts.push_back(For);

  // Build the function: returns [log1, log2, ..., t_log].
  auto *Fn = AST.make<Function>();
  Fn->Name = AST.intern("simulate");
  for (const auto &L : Logs) Fn->Outputs.push_back(AST.intern(L.Var));
  Fn->Outputs.push_back(AST.intern(TLog));
  Fn->Body = Body;

  auto *TU = AST.make<TranslationUnit>();
  // Drain helper functions (lowered via lowerSubsystemImpl during
  // the helper-binding pre-pass) before the top-level simulate().
  // Sema's TypeInference visits in order; placing helpers first
  // means simulate()'s call sites see typed return values.
  for (auto *Helper : Ctx.Pending) TU->Functions.push_back(Helper);
  TU->Functions.push_back(Fn);
  return TU;
}

//===----------------------------------------------------------------------===//
// Tier-7d — whole-diagram cocotb SIL emit.
//
// Walk the entry flow and synthesise a Python cocotb testbench that:
//   * runs every source / non-DUT internal / sink host-side (mirrors
//     `buildDiagramTU`'s per-tick body, just rendered as Python text);
//   * drives the DUT's input ports with Q<W>.<F>-packed values;
//   * waits a clock edge (+ user-supplied pipeline latency) and reads
//     back the DUT outputs;
//   * compares the decoded outputs against the same subsystem's
//     reference Python emit (from `-emit-python --subsystem <flow>`);
//   * accumulates per-tick CSV rows and writes them at sign-off.
//
// MVP carve-outs (logged + diagnosed):
//   - one DUT block per diagram;
//   - single-slot stateful internals only (TF order ≥ 2 deferred);
//   - no nested subsystems on the host side;
//   - DUT public ports are scalars in the Q<W>.<F> default format.
//===----------------------------------------------------------------------===//
namespace {

// Render a numeric literal at Python precision. Plain `std::to_string`
// truncates to 6 digits which loses precision for Ts / IC values; the
// `precision(17)` recipe matches the per-subsystem class wrapper.
std::string pyDouble(double V) {
  std::ostringstream Os;
  Os.precision(17);
  Os << V;
  return Os.str();
}

// Render the Python expression for a source block's instantaneous
// value at scalar `t`. Mirrors `lowerSourceExpr` but emits text.
std::string pySourceExpr(const Node &N) {
  const std::string &K = N.Kind;
  if (K == "signal_constant")
    return pyDouble(paramD(N, "value", 0.0));
  if (K == "signal_clock")
    return "t";
  if (K == "signal_step") {
    double ST = paramD(N, "stepTime", 1.0);
    double IV = paramD(N, "initialValue", 0.0);
    double FV = paramD(N, "finalValue", 1.0);
    std::ostringstream Os;
    Os << "(" << pyDouble(FV) << " if t >= " << pyDouble(ST)
       << " else " << pyDouble(IV) << ")";
    return Os.str();
  }
  if (K == "signal_sine") {
    double A  = paramD(N, "amplitude", 1.0);
    double F  = paramD(N, "frequency", 1.0);
    double P  = paramD(N, "phase", 0.0);
    double Bs = paramD(N, "bias", 0.0);
    std::ostringstream Os;
    Os << pyDouble(A) << " * math.sin(" << pyDouble(F) << " * t + "
       << pyDouble(P) << ") + " << pyDouble(Bs);
    return Os.str();
  }
  if (K == "signal_ramp") {
    double Slope = paramD(N, "slope", 1.0);
    double Start = paramD(N, "startTime", 0.0);
    double Init  = paramD(N, "initialOutput", 0.0);
    std::ostringstream Os;
    Os << "(" << pyDouble(Init) << " + " << pyDouble(Slope)
       << " * (t - " << pyDouble(Start) << ")"
       << " if t >= " << pyDouble(Start)
       << " else " << pyDouble(Init) << ")";
    return Os.str();
  }
  if (K == "signal_pulse") {
    double A   = paramD(N, "amplitude", 1.0);
    double Pp  = paramD(N, "period", 1.0);
    double W   = paramD(N, "pulseWidth", 50.0);
    double Phd = paramD(N, "phaseDelay", 0.0);
    std::ostringstream Os;
    Os << "(" << pyDouble(A) << " if ((t - " << pyDouble(Phd)
       << ") % " << pyDouble(Pp) << ") < "
       << pyDouble(Pp * W * 0.01) << " else 0.0)";
    return Os.str();
  }
  if (K == "signal_chirp") {
    double A  = paramD(N, "amplitude", 1.0);
    double F0 = paramD(N, "initialFrequency", 0.1);
    double F1 = paramD(N, "targetFrequency", 1.0);
    double Tt = paramD(N, "targetTime", 1.0);
    double P  = paramD(N, "phase", 0.0);
    if (Tt <= 0.0) Tt = 1.0;
    std::ostringstream Os;
    Os << pyDouble(A) << " * math.sin(2 * math.pi * ("
       << pyDouble(F0) << " + " << pyDouble((F1 - F0) / (2.0 * Tt))
       << " * t) * t + " << pyDouble(P) << ")";
    return Os.str();
  }
  if (K == "signal_repeating_sequence") {
    auto It = N.Params.find("outputValues");
    auto Tt = N.Params.find("timeValues");
    double Out0 = 0.0, Out1 = 1.0, T0 = 0.0, T1 = 1.0;
    if (It != N.Params.end()) {
      std::vector<double> O; int Or = 0, Oc = 0;
      parseMatrixStr(It->second, O, Or, Oc);
      if (!O.empty()) Out0 = O.front();
      if (O.size() >= 2) Out1 = O.back();
    }
    if (Tt != N.Params.end()) {
      std::vector<double> T; int Tr = 0, Tc = 0;
      parseMatrixStr(Tt->second, T, Tr, Tc);
      if (!T.empty()) T0 = T.front();
      if (T.size() >= 2) T1 = T.back();
    }
    double Period = T1 - T0;
    if (Period <= 0.0) Period = 1.0;
    std::ostringstream Os;
    Os << pyDouble(Out0) << " + " << pyDouble((Out1 - Out0) / Period)
       << " * ((t - " << pyDouble(T0) << ") % " << pyDouble(Period)
       << ")";
    return Os.str();
  }
  return "0.0";
}

} // namespace

std::optional<std::string> emitDiagramCocotbHarness(
    const FlowDoc &Doc,
    const std::string &EntryFlowName,
    const DiagramCocotbOptions &Opts,
    matlab::DiagnosticEngine &Diag) {
  const Flow *Entry = Doc.findFlow(EntryFlowName);
  if (!Entry) {
    Diag.error(SourceLocation{},
               "flow \"" + EntryFlowName +
                   "\" not found in `.mflow` document");
    return std::nullopt;
  }

  if (Opts.Duts.empty()) {
    Diag.error(SourceLocation{},
               "cocotb-SIL: DiagramCocotbOptions::Duts is empty — "
               "caller must populate at least one DutSpec.");
    return std::nullopt;
  }
  // Locate every DUT block in the entry flow and snapshot its node
  // pointer. Multi-DUT (`--dut a,b,c`) populates `Opts.Duts` with
  // one entry per DUT; single-DUT runs with a one-element list.
  std::vector<const Node *> DutNodes(Opts.Duts.size(), nullptr);
  for (size_t I = 0; I < Opts.Duts.size(); ++I) {
    const std::string &BId = Opts.Duts[I].BlockId;
    for (const auto &N : Entry->Nodes) {
      if (N.Id == BId) { DutNodes[I] = &N; break; }
    }
    if (!DutNodes[I]) {
      Diag.error(SourceLocation{},
                 "--dut block \"" + BId +
                     "\" not found in entry flow \"" + EntryFlowName + "\"");
      return std::nullopt;
    }
    if (DutNodes[I]->Kind != "signal_subsystem") {
      Diag.error(DutNodes[I]->Loc,
                 "--dut block \"" + BId +
                     "\" must be a signal_subsystem (kind is \"" +
                     DutNodes[I]->Kind + "\")");
      return std::nullopt;
    }
  }
  auto isDutNode = [&](const Node *N) -> int {
    for (size_t I = 0; I < DutNodes.size(); ++I)
      if (DutNodes[I] == N) return (int)I;
    return -1;
  };
  auto dutByBlockId = [&](const std::string &BId) -> int {
    for (size_t I = 0; I < Opts.Duts.size(); ++I)
      if (Opts.Duts[I].BlockId == BId) return (int)I;
    return -1;
  };
  bool MultiDut = Opts.Duts.size() > 1;
  bool AnySequential = false;
  for (const auto &D : Opts.Duts) if (D.Sequential) AnySequential = true;

  // Categorise blocks: sources / sinks / internal-non-DUT / DUT.
  // Nested non-DUT signal_subsystem blocks are accepted when the
  // caller supplies the matching HostHelpers entry (the orchestrator
  // self-invokes `-emit-python --subsystem <flow>` for each one).
  // Without a matching HostHelpers entry the legacy "Tier-7d follow-
  // up" diagnostic still fires.
  auto hostHelperFor = [&](const std::string &BlockId)
      -> const DiagramCocotbOptions::HostHelper * {
    for (const auto &H : Opts.HostHelpers)
      if (H.BlockId == BlockId) return &H;
    return nullptr;
  };
  std::vector<const Node *> Sources, Sinks, Internal;
  for (const auto &N : Entry->Nodes) {
    if (isDutNode(&N) >= 0) continue;
    if (N.Kind == "signal_inport" || N.Kind == "signal_outport") continue;
    if (isSourceKind(N.Kind)) { Sources.push_back(&N); continue; }
    if (isDiagramSink(N.Kind)) { Sinks.push_back(&N); continue; }
    if (N.Kind == "signal_subsystem" && !hostHelperFor(N.Id)) {
      Diag.error(N.Loc,
                 "cocotb-SIL: non-DUT `signal_subsystem` \"" + N.Id +
                 "\" needs a per-subsystem Python reference; the "
                 "orchestrator should populate "
                 "DiagramCocotbOptions::HostHelpers for it before "
                 "calling emitDiagramCocotbHarness.");
      return std::nullopt;
    }
    Internal.push_back(&N);
  }

  // Allocate variable names per block.
  std::set<std::string> Used{
      "t", "k", "Ts", "N", "host", "ref", "dut", "TOL", "math",
      "cocotb", "log_t", "fi_signed", "fi_w", "fi_f", "csv", "row",
      "log_dut", "log_ref", "log_err", "pack_fi", "unpack_fi"};
  auto uniqueName = [&](const std::string &Base) {
    std::string Cand = sanitizeIdent(Base);
    if (Cand.empty()) Cand = "v";
    std::string Name = Cand;
    int Suf = 1;
    while (Used.count(Name)) Name = Cand + "_" + std::to_string(++Suf);
    Used.insert(Name);
    return Name;
  };
  std::unordered_map<std::string, std::string> VarOfNode;
  for (auto *N : Sources)  VarOfNode[N->Id] = uniqueName(N->Id);
  for (auto *N : Internal) VarOfNode[N->Id] = uniqueName(N->Id);
  // Each DUT block contributes one var per output port — those
  // carry the sampled-from-DUT value into the rest of the diagram.
  // Inputs are computed from upstream values like any other block.
  // For multi-DUT, DutOutVars[d][i] is the local for DUT d's i-th
  // output port.
  std::vector<std::vector<std::string>> DutOutVars(Opts.Duts.size());
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    const auto &Spec = Opts.Duts[D];
    for (size_t I = 0; I < Spec.OutputPorts.size(); ++I)
      DutOutVars[D].push_back(uniqueName(Spec.BlockId + "_" +
                                         Spec.OutputPorts[I]));
  }

  // Per-sink log array.
  struct SinkLog { const Node *N; std::string Name; std::string Src; };
  std::vector<SinkLog> Logs;
  EdgeIndex EI = buildEdgeIndex(*Entry);
  for (auto *N : Sinks) {
    if (N->Kind == "signal_terminator") continue;
    SinkLog L; L.N = N;
    L.Name = uniqueName("log_" + N->Id);
    for (const auto &P : N->InPorts) {
      auto It = EI.Map.find({N->Id, P.Id});
      if (It != EI.Map.end()) { L.Src = It->second.first; break; }
    }
    Logs.push_back(L);
  }

  // State slots for non-DUT stateful internals.
  struct LocalState { const Node *N; std::string Var; double Init; };
  std::vector<LocalState> States;
  auto stateOrderForBlock = [&](const Node *N) -> int {
    if (N->Kind == "signal_transfer_fcn" || N->Kind == "signal_zero_pole") {
      std::vector<double> Num, Den;
      if (!resolveTFCoeffs(*N, Num, Den)) return 1;
      if (Den.size() < 2) return 1;
      return (int)Den.size() - 1;
    }
    if (N->Kind == "signal_state_space") {
      auto It = N->Params.find("A");
      if (It == N->Params.end()) return 1;
      std::vector<double> A; int Ar = 0, Ac = 0;
      parseMatrixStr(It->second, A, Ar, Ac);
      return (Ar > 0 && Ar == Ac) ? Ar : 1;
    }
    return 1;
  };
  for (auto *N : Internal) {
    // Helper-bound subsystems carry their own state inside the
    // imported Python class — no host-local state slot needed.
    if (N->Kind == "signal_subsystem" && hostHelperFor(N->Id)) continue;
    if (!isStatefulKind(N->Kind)) continue;
    int Order = stateOrderForBlock(N);
    if (Order > 1) {
      Diag.error(N->Loc,
                 "cocotb-SIL MVP supports single-slot stateful blocks "
                 "only (block \"" + N->Id + "\" needs " +
                 std::to_string(Order) + " state slots) — Tier-7d "
                 "follow-up");
      return std::nullopt;
    }
    LocalState S; S.N = N;
    S.Var = uniqueName("s_" + N->Id);
    S.Init = initialStateOf(*N);
    States.push_back(S);
  }
  // Helper var name (CamelCase class lives in HostHelper, but the
  // HostModel field uses snake_case `helper_<block_id>`).
  std::unordered_map<std::string, std::string> HelperFieldOf;
  for (const auto &H : Opts.HostHelpers) {
    HelperFieldOf[H.BlockId] = uniqueName("helper_" + H.BlockId);
  }

  // Resolve Ts / N from the model's solver settings.
  double Ts = 0.0;
  double Tstart = 0.0, Tstop = 0.0;
  if (Doc.Settings.Solver.has_value()) {
    const auto &SC = *Doc.Settings.Solver;
    if (SC.MaxStep != "auto") {
      try { Ts = std::stod(SC.MaxStep); } catch (...) {}
    }
    Tstart = SC.StartTime;
    Tstop  = SC.StopTime;
  }
  if (Ts <= 0.0) Ts = 0.01;
  double Span = Tstop - Tstart;
  int NTicks = Span > 0.0 ? (int)std::round(Span / Ts) : 100;
  if (NTicks < 1) NTicks = 1;

  // Helpers for input resolution.
  auto upstreamVar = [&](const std::string &ToNode,
                         const std::string &ToPort) -> std::string {
    auto It = EI.Map.find({ToNode, ToPort});
    if (It == EI.Map.end()) return "0.0";
    const std::string &Up = It->second.first;
    const std::string &UpPort = It->second.second;
    if (int D = dutByBlockId(Up); D >= 0) {
      // Reading a DUT's output. Map UpPort to its index in that
      // DUT's OutputPorts; fall back to its first output on
      // mismatch.
      const auto &Spec = Opts.Duts[D];
      for (size_t I = 0; I < Spec.OutputPorts.size(); ++I)
        if (Spec.OutputPorts[I] == UpPort) return DutOutVars[D][I];
      if (!DutOutVars[D].empty()) return DutOutVars[D].front();
      return "0.0";
    }
    auto VIt = VarOfNode.find(Up);
    if (VIt == VarOfNode.end()) return "0.0";
    return VIt->second;
  };
  auto inputPortsOf = [&](const Node &N) -> std::vector<std::string> {
    std::vector<std::pair<int, std::string>> Pairs;
    for (const auto &P : N.InPorts) {
      int Idx = parsePortIndex(P.Id);
      if (Idx < 0) Idx = (int)Pairs.size() + 1;
      Pairs.push_back({Idx, P.Id});
    }
    std::sort(Pairs.begin(), Pairs.end());
    std::vector<std::string> Out;
    for (auto &PR : Pairs) Out.push_back(PR.second);
    return Out;
  };

  // Render a non-DUT internal block's body line. Supports stateless
  // blocks via direct Python expressions and the same single-slot
  // stateful blocks as `buildDiagramTU`. For coverage outside this
  // set, error out (Tier-7d follow-up to share `lowerBlock`).
  auto renderInternal = [&](const Node *N,
                            std::string &Body,
                            std::string &PostBody) -> bool {
    const std::string &OutVar = VarOfNode[N->Id];
    auto Ports = inputPortsOf(*N);
    std::vector<std::string> Ins;
    for (auto &P : Ports) Ins.push_back(upstreamVar(N->Id, P));

    // Host-side nested subsystem: call its helper class's step()
    // and capture the return into OutVar. The helper instance is
    // pinned on the HostModel (so its state carries across ticks).
    if (auto *Helper = hostHelperFor(N->Id)) {
      std::ostringstream Os;
      Os << "        " << OutVar << " = self." << HelperFieldOf[N->Id]
         << ".step(";
      for (size_t I = 0; I < Ins.size(); ++I) {
        if (I) Os << ", ";
        Os << Ins[I];
      }
      Os << ")\n";
      Body += Os.str();
      return true;
    }

    LocalState *Slot = nullptr;
    for (auto &S : States) if (S.N == N) { Slot = &S; break; }
    if (Slot) {
      // State-read hoisted earlier (we just emit OutVar = self.<slot>
      // at the head of the tick body in the generator). Compute the
      // next-state expression here and assign in PostBody so that any
      // consumers downstream of stateful blocks see the *current*
      // state on this tick.
      std::string U = Ins.empty() ? std::string("0.0") : Ins.front();
      std::string Next;
      if (N->Kind == "signal_unit_delay" || N->Kind == "signal_zoh") {
        Next = U;
      } else if (N->Kind == "signal_discrete_integrator" ||
                 N->Kind == "signal_integrator") {
        double LocalTs = paramD(*N, "sample_time", 0.0);
        if (LocalTs <= 0.0) LocalTs = paramD(*N, "sampleTime", Ts);
        std::ostringstream Os;
        Os << "self." << Slot->Var << " + " << pyDouble(LocalTs)
           << " * (" << U << ")";
        Next = Os.str();
      } else if (N->Kind == "signal_transfer_fcn" ||
                 N->Kind == "signal_zero_pole") {
        std::vector<double> Num, Den;
        resolveTFCoeffs(*N, Num, Den);
        double Lead = Den.empty() ? 1.0 : Den.front();
        double A0 = Den.size() >= 2 ? Den[1] / Lead : 0.0;
        double B0 = !Num.empty() ? Num.front() / Lead : 1.0;
        std::ostringstream Os;
        Os << "self." << Slot->Var << " + " << pyDouble(Ts) << " * ("
           << pyDouble(B0) << " * (" << U << ") - " << pyDouble(A0)
           << " * self." << Slot->Var << ")";
        Next = Os.str();
      } else {
        Next = "self." + Slot->Var;
      }
      PostBody += "        self." + Slot->Var + " = " + Next + "\n";
      // Current-state read happens once at the top of each tick via
      // `OutVar = host.<slot>` — generated up in the prologue below.
      return true;
    }

    // Stateless internals — limited set for the MVP. Extending this
    // to full `lowerBlock` parity is the natural next iteration.
    if (N->Kind == "signal_sum") {
      auto It = N->Params.find("signs");
      std::string Signs = It != N->Params.end() ? It->second : "++";
      std::ostringstream Os;
      Os << "        " << OutVar << " = ";
      for (size_t I = 0; I < Ins.size(); ++I) {
        char S = (I < Signs.size()) ? Signs[I] : '+';
        if (I == 0)
          Os << (S == '-' ? "-(" + Ins[I] + ")" : Ins[I]);
        else
          Os << " " << S << " (" << Ins[I] << ")";
      }
      Os << "\n";
      Body += Os.str();
      return true;
    }
    if (N->Kind == "signal_gain") {
      double G = paramD(*N, "gain", 1.0);
      std::ostringstream Os;
      Os << "        " << OutVar << " = " << pyDouble(G) << " * ("
         << (Ins.empty() ? "0.0" : Ins.front()) << ")\n";
      Body += Os.str();
      return true;
    }
    if (N->Kind == "signal_constant") {
      Body += "        " + OutVar + " = " +
              pyDouble(paramD(*N, "value", 0.0)) + "\n";
      return true;
    }
    Diag.error(N->Loc,
               "cocotb-SIL MVP: block kind \"" + N->Kind +
               "\" (block \"" + N->Id + "\") not in the host-side "
               "rendering set yet — Tier-7d follow-up");
    return false;
  };

  // Build the per-tick Python body in two halves:
  //   * Pre-DUT: state-reads, sources, host internals strictly
  //     upstream of the DUT, DUT input resolution.
  //   * Post-DUT: host internals strictly downstream of the DUT,
  //     sink logging, state-next updates.
  std::ostringstream Os;
  Os << "# Generated by matlabc -emit-cocotb. Do not edit.\n";
  Os << "#\n";
  Os << "# Whole-diagram SIL harness. Entry flow: " << EntryFlowName
     << "\n";
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    const auto &Spec = Opts.Duts[D];
    Os << "# DUT[" << D << "] block       : " << Spec.BlockId
       << " (SV module: " << Spec.ModuleName << ")\n";
    Os << "# DUT[" << D << "] reference   : " << Spec.RefModule << "."
       << Spec.RefClass << "\n";
  }
  if (!Opts.WrapperModule.empty())
    Os << "# Wrapper module  : " << Opts.WrapperModule << "\n";
  Os << "# Q-format        : " << (Opts.FiSigned ? "Q" : "UQ")
     << Opts.FiWidth - Opts.FiFrac << "." << Opts.FiFrac << "\n";
  Os << "# Tolerance       : " << pyDouble(Opts.Tolerance) << "\n";
  Os << "import math, os, csv\n";
  Os << "import cocotb\n";
  if (AnySequential) {
    Os << "from cocotb.clock import Clock\n";
    Os << "from cocotb.triggers import RisingEdge, Timer\n";
  } else {
    Os << "from cocotb.triggers import Timer\n";
  }
  Os << "from cocotb_fi import pack_fi, unpack_fi\n";
  for (const auto &Spec : Opts.Duts) {
    Os << "from " << Spec.RefModule << " import " << Spec.RefClass
       << "\n";
  }
  // Host-side helper subsystems. Each one was emitted as a
  // standalone Python module by the orchestrator's self-invocation;
  // we import the class here and instantiate it in HostModel below.
  for (const auto &H : Opts.HostHelpers) {
    Os << "from " << H.ModuleName << " import " << H.ClassName << "\n";
  }
  Os << "\n";

  // Host model — keeps per-tick state for non-DUT stateful blocks.
  Os << "class HostModel:\n";
  Os << "    \"\"\"Host-side reference for everything in the entry "
        "flow except\n";
  Os << "    the DUT. Per-tick state for stateful blocks lives on "
        "the instance;\n";
  Os << "    pre_dut() returns the DUT input tuple, post_dut() "
        "consumes the DUT\n";
  Os << "    output tuple and updates downstream values + state.\n";
  Os << "    \"\"\"\n";
  Os << "    def __init__(self):\n";
  if (States.empty() && Opts.HostHelpers.empty())
    Os << "        pass\n";
  for (auto &S : States)
    Os << "        self." << S.Var << " = " << pyDouble(S.Init) << "\n";
  // Helper-class instances — one per non-DUT signal_subsystem.
  for (const auto &H : Opts.HostHelpers) {
    Os << "        self." << HelperFieldOf[H.BlockId] << " = "
       << H.ClassName << "()\n";
  }

  // Pre-DUT: every node strictly upstream of the DUT. The MVP runs
  // every source unconditionally and every non-DUT internal in
  // toposort order — this is sound because the DUT's inputs may
  // depend on any subset of sources / stateful blocks, and post-DUT
  // sees the same updated values. Splitting at the DUT site is a
  // pure-performance refinement (post can skip pre-evaluated nodes).
  auto Topo = toposortInternals(*Entry, Diag, EntryFlowName);
  if (Diag.hasErrors()) return std::nullopt;
  std::vector<const Node *> TopoNonDut;
  for (auto *N : Topo) {
    if (isDutNode(N) >= 0) continue;
    if (isSourceKind(N->Kind) || isDiagramSink(N->Kind)) continue;
    TopoNonDut.push_back(N);
  }

  Os << "    def pre_dut(self, t):\n";
  // State-reads — load current state into the block's OutVar so any
  // downstream block sees this-tick's value. The return statement at
  // the end of the method always emits a body line so no `pass`
  // sentinel is needed.
  for (auto &S : States)
    Os << "        " << VarOfNode[S.N->Id] << " = self." << S.Var << "\n";
  // Sources.
  for (auto *N : Sources)
    Os << "        " << VarOfNode[N->Id] << " = " << pySourceExpr(*N) << "\n";
  // Non-DUT internals (toposorted).
  std::string PreBody;
  std::string PostBody;  // collected next-state updates; rendered after DUT
  for (auto *N : TopoNonDut) {
    if (!renderInternal(N, PreBody, PostBody)) return std::nullopt;
  }
  Os << PreBody;
  // Stash variables on `self` so post_dut can use them (Python
  // closures over locals don't survive method boundaries). Cheap +
  // explicit.
  for (auto *N : Sources)
    Os << "        self._" << VarOfNode[N->Id] << " = "
       << VarOfNode[N->Id] << "\n";
  for (auto *N : Internal)
    Os << "        self._" << VarOfNode[N->Id] << " = "
       << VarOfNode[N->Id] << "\n";
  // Resolve every DUT's input tuple from the entry flow's wiring,
  // concatenated in `Opts.Duts` order. The harness drives them in
  // the same order — see the `dut.<prefix>_<port> = pack_fi(...)`
  // chain below.
  size_t TotalIns = 0;
  for (const auto &Spec : Opts.Duts) TotalIns += Spec.InputPorts.size();
  Os << "        return (";
  size_t IdxI = 0;
  for (const auto &Spec : Opts.Duts) {
    for (size_t I = 0; I < Spec.InputPorts.size(); ++I) {
      if (IdxI) Os << ", ";
      Os << upstreamVar(Spec.BlockId, Spec.InputPorts[I]);
      ++IdxI;
    }
  }
  if (TotalIns == 1) Os << ",";
  Os << ")\n\n";

  // Post-DUT: replay non-DUT pre-evaluated values onto locals, accept
  // the DUT output tuple, drive sinks, advance state.
  Os << "    def post_dut(self, t, dut_outs):\n";
  // Replay state-reads (the same values pre_dut used — they're still
  // valid for state-next computation).
  for (auto &S : States)
    Os << "        " << VarOfNode[S.N->Id] << " = self." << S.Var << "\n";
  for (auto *N : Sources)
    Os << "        " << VarOfNode[N->Id] << " = self._"
       << VarOfNode[N->Id] << "\n";
  for (auto *N : Internal)
    Os << "        " << VarOfNode[N->Id] << " = self._"
       << VarOfNode[N->Id] << "\n";
  // Unpack DUT outputs into the named OutVars. dut_outs is a flat
  // tuple concatenating every DUT's outputs in `Opts.Duts` order.
  {
    size_t IdxO = 0;
    for (size_t D = 0; D < Opts.Duts.size(); ++D) {
      for (size_t I = 0; I < DutOutVars[D].size(); ++I) {
        Os << "        " << DutOutVars[D][I] << " = dut_outs["
           << IdxO << "]\n";
        ++IdxO;
      }
    }
  }
  // Sink logging: log[k] = upstream value.
  for (auto &L : Logs) {
    std::string Src = "0.0";
    if (!L.Src.empty()) {
      int DUp = dutByBlockId(L.Src);
      if (DUp >= 0 && !DutOutVars[DUp].empty()) {
        // Look up the sink's source port → DUT output index.
        const auto &Spec = Opts.Duts[DUp];
        Src = DutOutVars[DUp].front();
        for (size_t I = 0; I < Spec.OutputPorts.size(); ++I) {
          // SinkLog doesn't keep the source-port name today — we
          // route through VarOfNodePort-style first-output fallback.
          // (Multi-output sink wiring is a future enhancement; the
          // first port is the common case.)
          (void)I;
        }
      } else if (VarOfNode.count(L.Src)) {
        Src = VarOfNode[L.Src];
      }
    }
    Os << "        self." << L.Name << ".append(" << Src << ")\n";
  }
  // State-next updates. The DUT-output unpack always emits a body
  // line so `pass` isn't needed even when there are no logs / state
  // updates.
  if (!PostBody.empty()) Os << PostBody;
  Os << "\n";

  // Per-test driver. Multi-DUT routes through the same shape as
  // single-DUT, but with each DUT's signals prefixed by its block
  // id (e.g. `dut.<block_id>__u1`, `dut.<block_id>__y1`) when the
  // wrapper SV instantiates multiple DUTs.
  //
  // The harness keeps one pending-ref FIFO per DUT so each DUT's
  // pipeline-latency-aligned compare is independent. The CSV
  // header gets columns per DUT × output.
  auto portName = [&](size_t D, const std::string &Port) -> std::string {
    if (!MultiDut) return Port;
    return Opts.Duts[D].BlockId + "__" + Port;
  };
  Os << "@cocotb.test()\n";
  Os << "async def sil(dut):\n";
  Os << "    \"\"\"matlabc-generated SIL test. Drives the DUT(s) "
        "each tick, samples\n";
  Os << "    their outputs, compares against the host reference, "
        "and writes a CSV\n";
  Os << "    of (t, dut_y*, ref_y*, err*) rows.\n";
  Os << "    \"\"\"\n";
  Os << "    Ts = " << pyDouble(Ts) << "\n";
  Os << "    N  = " << NTicks << "\n";
  Os << "    L  = " << Opts.Latency
     << "  # pipeline latency: compare cycle k against drive cycle k-L\n";
  Os << "    TOL = " << pyDouble(Opts.Tolerance) << "\n";
  Os << "    fi_signed, fi_w, fi_f = "
     << (Opts.FiSigned ? "True" : "False") << ", " << Opts.FiWidth
     << ", " << Opts.FiFrac << "\n";
  if (AnySequential) {
    // Any sequential DUT triggers the shared clock + reset prologue.
    // The wrapper SV (multi-DUT) fans clk / rst_n / reset out to
    // each DUT instance; single-DUT runs hit the DUT module directly.
    Os << "    cocotb.start_soon(Clock(dut.clk, 10, units=\"ns\")"
          ".start())\n";
    Os << "    if hasattr(dut, \"rst_n\"): dut.rst_n.value = 0\n";
    Os << "    if hasattr(dut, \"reset\"): dut.reset.value = 1\n";
    Os << "    await RisingEdge(dut.clk); await RisingEdge(dut.clk)\n";
    Os << "    if hasattr(dut, \"rst_n\"): dut.rst_n.value = 1\n";
    Os << "    if hasattr(dut, \"reset\"): dut.reset.value = 0\n";
  }
  Os << "    host = HostModel()\n";
  for (auto &L : Logs)
    Os << "    host." << L.Name << " = []\n";
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    Os << "    ref_" << D << " = " << Opts.Duts[D].RefClass << "()\n";
  }
  Os << "    log_t = []\n";
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    for (size_t I = 0; I < Opts.Duts[D].OutputPorts.size(); ++I) {
      Os << "    log_d" << D << "_y" << (I + 1) << " = []\n";
      Os << "    log_d" << D << "_ref" << (I + 1) << " = []\n";
      Os << "    log_d" << D << "_err" << (I + 1) << " = []\n";
    }
  }
  // One pending-ref FIFO per DUT — independent latency-aligned
  // compare windows.
  for (size_t D = 0; D < Opts.Duts.size(); ++D)
    Os << "    pending_" << D << " = []\n";
  Os << "    fail_count = 0\n";
  Os << "    for k in range(N + L):\n";
  Os << "        if k < N:\n";
  Os << "            t = k * Ts\n";
  Os << "            u_all = host.pre_dut(t)\n";
  // Slice u_all per-DUT and compute per-DUT ref outputs.
  {
    size_t Cur = 0;
    for (size_t D = 0; D < Opts.Duts.size(); ++D) {
      size_t NI = Opts.Duts[D].InputPorts.size();
      Os << "            u_" << D << " = u_all[" << Cur << ":"
         << (Cur + NI) << "]\n";
      Os << "            _ref_out_" << D << " = ref_" << D
         << ".step(*u_" << D << ")\n";
      Os << "            ref_tuple_" << D
         << " = _ref_out_" << D << " if isinstance(_ref_out_" << D
         << ", tuple) else (_ref_out_" << D << ",)\n";
      Os << "            pending_" << D << ".append((t, u_" << D
         << ", ref_tuple_" << D << "))\n";
      Cur += NI;
    }
  }
  Os << "        else:\n";
  Os << "            t = float(\"nan\")\n";
  Os << "            u_all = tuple(0.0 for _ in range("
     << TotalIns << "))\n";
  {
    size_t Cur = 0;
    for (size_t D = 0; D < Opts.Duts.size(); ++D) {
      size_t NI = Opts.Duts[D].InputPorts.size();
      Os << "            u_" << D << " = u_all[" << Cur << ":"
         << (Cur + NI) << "]\n";
      Cur += NI;
    }
  }
  // Drive every DUT's inputs in port order.
  {
    size_t Cur = 0;
    for (size_t D = 0; D < Opts.Duts.size(); ++D) {
      for (size_t I = 0; I < Opts.Duts[D].InputPorts.size(); ++I) {
        Os << "        dut." << portName(D, Opts.Duts[D].InputPorts[I])
           << ".value = pack_fi(u_all[" << Cur
           << "], fi_signed, fi_w, fi_f)\n";
        ++Cur;
      }
    }
  }
  // Sample DUT outputs. Sequential DUTs sample BEFORE the rising
  // edge so the FF output reflects the pre-edge state (matching
  // MATLAB unit-delay y[k]=u[k-1]); combinational DUTs settle for
  // 1 ns and read directly. Multi-DUT with mixed sequential /
  // combinational: a small Timer covers both cases — the
  // RisingEdge below advances the FF for sequential DUTs.
  Os << "        await Timer(1, units=\"ns\")\n";
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    for (size_t I = 0; I < Opts.Duts[D].OutputPorts.size(); ++I) {
      Os << "        y" << D << "_" << (I + 1)
         << " = unpack_fi(int(dut."
         << portName(D, Opts.Duts[D].OutputPorts[I])
         << ".value), fi_signed, fi_w, fi_f)\n";
    }
  }
  if (AnySequential) {
    // Advance the clock once per tick — the wrapper SV fans clk to
    // all DUT instances, so combinational DUTs in the same harness
    // simply ignore the edge.
    Os << "        await RisingEdge(dut.clk)\n";
  }
  Os << "        if k < L:\n";
  Os << "            continue  # pipeline still filling\n";
  // Per-DUT compare.
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    Os << "        if pending_" << D << ":\n";
    Os << "            t_cmp_" << D << ", u_cmp_" << D
       << ", ref_tuple_" << D << " = pending_" << D << ".pop(0)\n";
    for (size_t I = 0; I < Opts.Duts[D].OutputPorts.size(); ++I) {
      Os << "            err_" << D << "_" << (I + 1)
         << " = y" << D << "_" << (I + 1)
         << " - ref_tuple_" << D << "[" << I << "]\n";
      Os << "            if abs(err_" << D << "_" << (I + 1)
         << ") > TOL:\n";
      Os << "                fail_count += 1\n";
      Os << "                cocotb.log.error(\n";
      Os << "                    f\"sil[d" << D << " t={t_cmp_" << D
         << ":.6g}] y" << (I + 1) << ": dut={y" << D << "_" << (I + 1)
         << "} ref={ref_tuple_" << D << "[" << I << "]}"
         << " err={err_" << D << "_" << (I + 1)
         << "} (tol={TOL})\")\n";
      Os << "            log_d" << D << "_y" << (I + 1)
         << ".append(y" << D << "_" << (I + 1) << ")\n";
      Os << "            log_d" << D << "_ref" << (I + 1)
         << ".append(ref_tuple_" << D << "[" << I << "])\n";
      Os << "            log_d" << D << "_err" << (I + 1)
         << ".append(err_" << D << "_" << (I + 1) << ")\n";
    }
  }
  // Log time from DUT 0 (all DUTs see the same tick — they're
  // driven in lockstep by `for k in range(N + L)`).
  Os << "        if pending_0 is None or True:\n";
  Os << "            log_t.append(t if k < N else float(\"nan\"))\n";
  // Update host downstream + state on the compare cycle. The flat
  // tuple concatenates every DUT's outputs in `Opts.Duts` order.
  Os << "        host.post_dut(t, (";
  {
    size_t IdxO = 0;
    for (size_t D = 0; D < Opts.Duts.size(); ++D) {
      for (size_t I = 0; I < Opts.Duts[D].OutputPorts.size(); ++I) {
        if (IdxO) Os << ", ";
        Os << "y" << D << "_" << (I + 1);
        ++IdxO;
      }
    }
    if (IdxO == 1) Os << ",";
  }
  Os << "))\n";
  Os << "    csv_path = os.environ.get(\"SIL_CSV\", \"sil_log.csv\")\n";
  Os << "    with open(csv_path, \"w\", newline=\"\") as f:\n";
  Os << "        w = csv.writer(f)\n";
  Os << "        header = [\"t\"]\n";
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    for (size_t I = 0; I < Opts.Duts[D].OutputPorts.size(); ++I) {
      Os << "        header += [\"d" << D << "_dut" << (I + 1)
         << "\", \"d" << D << "_ref" << (I + 1)
         << "\", \"d" << D << "_err" << (I + 1) << "\"]\n";
    }
  }
  Os << "        w.writerow(header)\n";
  Os << "        for i, t_i in enumerate(log_t):\n";
  Os << "            row = [t_i]\n";
  for (size_t D = 0; D < Opts.Duts.size(); ++D) {
    for (size_t I = 0; I < Opts.Duts[D].OutputPorts.size(); ++I) {
      Os << "            row += [log_d" << D << "_y" << (I + 1)
         << "[i], log_d" << D << "_ref" << (I + 1)
         << "[i], log_d" << D << "_err" << (I + 1) << "[i]]\n";
    }
  }
  Os << "            w.writerow(row)\n";
  Os << "    cocotb.log.info(\n";
  Os << "        f\"matlabc cocotb-SIL: {len(log_t)} compares, "
        "{fail_count} mismatches, csv={csv_path}\")\n";
  Os << "    assert fail_count == 0, "
        "f\"SIL mismatch on {fail_count} ticks (TOL={TOL})\"\n";
  return Os.str();
}

} // namespace matlab::flowchart
