//===----------------------------------------------------------------------===//
// Monomorphize — see header for design. Implements the fixpoint clone-and-
// rewrite loop. Each iteration:
//
//   1. Analyze the current TU for per-callee signature buckets.
//   2. For each callee F with >1 buckets: pick a canonical signature
//      (alphabetically first), keep F's original body for that signature,
//      and clone F once per remaining signature with name `F__s<i>`.
//      Clones append to TU.Functions.
//   3. Walk every CallOrIndex that resolves to F. If its arg-type
//      signature is non-canonical, rewrite the NameExpr.Name to the
//      mangled clone name (interned via ASTContext).
//   4. If anything changed, re-run Sema via the caller-supplied callback
//      so the clones get bindings and the rewritten call sites resolve
//      to them.
//
// Bookkeeping: we track, per Function*, which signatures have already
// been monomorphized in prior iterations. That prevents an iteration's
// re-analysis from re-cloning the same signature in the freshly-resolved
// TU (each clone introduces its own signature buckets that would
// otherwise look like duplicates of the original to a naive analyzer).
//===----------------------------------------------------------------------===//

#include "matlab/Sema/Monomorphize.h"

#include "matlab/AST/AST.h"
#include "matlab/AST/Cloner.h"
#include "matlab/Sema/CallSiteAnalyzer.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>

namespace matlab {

namespace {

// Per-iteration plan: for one polymorphic callee F, the mapping from
// signature to mangled function name. The canonical bucket maps to F's
// original name; every other bucket maps to a freshly-allocated mangled
// name (the clone is created when the plan is applied).
using SignatureToName = std::map<CallSignature, std::string>;

// Cross-iteration state. We keep, per Function*, the SignatureToName
// table that's been chosen so far. When a re-analysis fires later, we
// reuse the existing mangled names for signatures we've already cloned —
// only NEW signatures (e.g. surfaced inside a clone body) trigger fresh
// clones. The "originals" set tracks the un-cloned root function for
// each clone, so a clone's call sites of itself reuse the chain.
struct ClonePlan {
  // Per root Function*, the signatures we've already assigned a name to.
  std::map<Function *, SignatureToName> ByRoot;
  // Reverse map from clone Function* → root Function*. Used so that
  // when we analyze a clone in a later iteration, we can look up the
  // root's signature table instead of treating the clone as a new root.
  std::map<Function *, Function *> CloneToRoot;
};

// Return the root Function* for F. If F is itself a clone, follows
// CloneToRoot once; otherwise returns F. Plans only nest one level deep
// because clones are made from roots, never from other clones.
Function *rootOf(ClonePlan &Plan, Function *F) {
  auto It = Plan.CloneToRoot.find(F);
  return It == Plan.CloneToRoot.end() ? F : It->second;
}

// Build the next iteration's plan: scan the current TU for polymorphic
// callees, allocate names for any signatures we haven't seen before,
// and create+register the corresponding clones in TU.Functions. Returns
// the number of new clones added.
int growPlan(TranslationUnit &TU, ASTContext &Ctx, ClonePlan &Plan) {
  // Group call sites by their ROOT function. We track, per (root, sig),
  // whether the sig has at least one EXTERNAL caller (caller's root !=
  // callee's root). A sig that only appears at recursive self-call
  // sites doesn't deserve its own clone — Phase 4's fixpoint rewrites
  // the recursive call to match whichever specialisation the external
  // caller settled on, and the would-be canonical clone for the
  // recursive sig ends up dead with un-lowerable matlab.* ops in its
  // body. See fn_call_recursive / fn_call_self_two regressions.
  std::map<Function *, std::map<CallSignature, bool>> SigHasExt;
  // Track which roots have at least one call site whose arity differs
  // from the declared parameter count (i.e. fewer args via nargin
  // semantics), or whose declared output list ends in `varargout`.
  // Those callees rely on per-arity machinery (`matlab.nargin_value`
  // attr + varargout cell unpacking) the existing late MLIR
  // monomorphiser owns; Sema-time cloning would produce duplicate
  // bodies with stale references to undefined trailing params or
  // mis-sized output packing. Defer such callees to the late path.
  std::set<Function *> SkipRoot;
  walkUserCallsWithCaller(
      TU, [&](Function *Caller, CallOrIndex &C, NameExpr & /*N*/,
              Function &F) {
        Function *CalleeRoot = rootOf(Plan, &F);
        Function *CallerRoot = Caller ? rootOf(Plan, Caller) : nullptr;
        bool Recursive = (CallerRoot == CalleeRoot);
        // #40 Class 2/3: arity-varying callees (call args != declared,
        // MATLAB nargin semantics) and varargin callees are now absorbed
        // Sema-side via per-arity clones carrying NarginOverride — no
        // longer deferred to the late MLIR mono. A varargin call's
        // distinct arities already produce distinct CallSignatures (the
        // tail isn't packed into the cell until lowering), so they bucket
        // and clone naturally; the clone's nargin is its call arity.
        //
        // varargout callees still defer: their per-LHS unpacking is
        // handled entirely by the runtime cell helpers at the call site
        // (varargout_basic passes without any cloning), so adding clones
        // here would be pure churn.
        if (!CalleeRoot->Outputs.empty() &&
            CalleeRoot->Outputs.back() == "varargout")
          SkipRoot.insert(CalleeRoot);
        CallSignature Sig = callSignatureFrom(C.ArgTypes);
        // Touch the slot so its presence reflects "this sig was seen
        // for this callee". External sightings flip it to true; purely
        // recursive sightings leave it default-false.
        bool &Slot = SigHasExt[CalleeRoot][Sig];
        if (!Recursive) Slot = true;
      });

  // Build per-root bucket using only sigs that have at least one
  // external caller (or any sig already in the Plan from a prior
  // iteration — those clones must stay live).
  std::map<Function *, std::set<CallSignature>> ByRoot;
  for (auto &KV : SigHasExt) {
    Function *Root = KV.first;
    if (SkipRoot.count(Root)) continue; // defer to late mono
    auto &Bucket = ByRoot[Root];
    for (auto &SigSlot : KV.second)
      if (SigSlot.second) Bucket.insert(SigSlot.first);
    // Preserve any signatures already named in a prior iteration; the
    // recursion-closure step may rewrite recursive calls to those
    // names so they continue to need their clones.
    auto It = Plan.ByRoot.find(Root);
    if (It != Plan.ByRoot.end())
      for (auto &SN : It->second) Bucket.insert(SN.first);
  }

  int NewClones = 0;
  for (auto &KV : ByRoot) {
    Function *Root = KV.first;
    const std::set<CallSignature> &Bucket = KV.second;
    if (Bucket.size() <= 1) continue; // monomorphic

    SignatureToName &Table = Plan.ByRoot[Root];
    size_t DeclArity = Root->Inputs.size();
    bool IsVariadic =
        !Root->Inputs.empty() && Root->Inputs.back() == "varargin";
    // #40 Class 2: when a callee is invoked at several arities, keep the
    // declared-arity signature on the original Function* (canonical) and
    // make the reduced-arity ones clones carrying a NarginOverride. Order
    // the bucket so a full-arity sig is named first and claims suffix 0.
    std::vector<const CallSignature *> Order;
    Order.reserve(Bucket.size());
    for (const CallSignature &Sig : Bucket) Order.push_back(&Sig);
    std::stable_sort(Order.begin(), Order.end(),
        [&](const CallSignature *A, const CallSignature *B) {
          return (A->size() == DeclArity) && (B->size() != DeclArity);
        });
    int NextSuffix = static_cast<int>(Table.size()); // 0 means "canonical name"
    for (const CallSignature *SigP : Order) {
      const CallSignature &Sig = *SigP;
      if (Table.count(Sig)) continue; // already named
      std::string Name;
      Function *Target;
      if (NextSuffix == 0) {
        // Canonical signature — keeps the root's original name. Do NOT
        // clone; the original Function* is reused for this signature.
        Name = std::string(Root->Name);
        Target = Root;
      } else {
        Name = std::string(Root->Name) + "__s" + std::to_string(NextSuffix);
        Function *Clone = cloneFunction(Ctx, *Root, Name);
        TU.Functions.push_back(Clone);
        Plan.CloneToRoot[Clone] = Root;
        NewClones++;
        Target = Clone;
      }
      // Record the per-call-site nargin so the lowerer emits
      // matlab.nargin_value and the body's `if nargin == N` branches fold
      // for this specialisation. For a variadic callee nargin is the
      // actual call arity (Sig.size()); for a fixed-arity callee only a
      // reduced-arity clone needs an override (the full-arity canonical
      // uses its declared count).
      if (IsVariadic)
        Target->NarginOverride = static_cast<int>(Sig.size());
      else if (Sig.size() < DeclArity)
        Target->NarginOverride = static_cast<int>(Sig.size());
      Table[Sig] = std::move(Name);
      NextSuffix++;
    }
  }
  return NewClones;
}

// Rewrite every user-function call site in TU so its NameExpr.Name
// matches the plan's chosen specialization. Returns the number of
// NameExpr.Name rewrites performed (call sites that were already on the
// correct name are NOT counted — they're no-ops).
int applyPlan(TranslationUnit &TU, ASTContext &Ctx, const ClonePlan &Plan) {
  int Rewrites = 0;
  walkUserCalls(TU, [&](CallOrIndex &C, NameExpr &N, Function &F) {
    Function *Root = F.Name.empty() ? &F : &F;
    // Resolve through the clone-to-root map without mutating the input.
    auto CtoR = Plan.CloneToRoot.find(&F);
    if (CtoR != Plan.CloneToRoot.end()) Root = CtoR->second;

    auto It = Plan.ByRoot.find(Root);
    if (It == Plan.ByRoot.end()) return; // not a polymorphic root
    const SignatureToName &Table = It->second;

    CallSignature Sig = callSignatureFrom(C.ArgTypes);
    auto NIt = Table.find(Sig);
    if (NIt == Table.end()) return; // signature absent (would be a fresh
                                     // one — picked up on next growPlan)

    if (N.Name == NIt->second) return; // already on the right name
    N.Name = Ctx.intern(NIt->second);
    Rewrites++;
  });
  return Rewrites;
}

// Phase 5 stamping: record each user Function's surviving call-site
// signature in its ParamTypeStamps vector. The stamp persists across
// Sema re-runs (it lives on the Function, not on transient Bindings),
// so the next TypeInference run reads it as the initial env type for
// each param and the body refines under the specialised arg types.
//
// Post-mono, every call site of F shares the same signature — picking
// any call site's ArgTypes is sufficient. Returns the number of stamp
// slots that actually changed (so the driver can detect a fixpoint).
int stampSignatureTypes(TranslationUnit &TU, TypeContext &TC) {
  // Pre-scan: collect Functions to skip entirely. Only varargout
  // callees remain deferred (their per-LHS unpacking is a runtime-cell
  // concern, not a signature one — see growPlan). Arity-varying (#40
  // Class 2) and varargin (#40 Class 3) callees are NO LONGER skipped:
  // growPlan produces per-arity clones, and stamping below handles the
  // reduced-arity prefix and the varargin cell slot.
  std::set<Function *> Skip;
  walkUserCalls(TU, [&](CallOrIndex & /*C*/, NameExpr & /*N*/, Function &F) {
    if (!F.Outputs.empty() && F.Outputs.back() == "varargout")
      Skip.insert(&F);
  });

  std::map<Function *, std::vector<const Type *>> ChosenSigs;
  walkUserCalls(TU, [&](CallOrIndex &C, NameExpr & /*N*/, Function &F) {
    if (Skip.count(&F)) return;
    if (ChosenSigs.count(&F)) return;
    ChosenSigs[&F] = C.ArgTypes;
  });

  int Stamped = 0;
  for (auto &KV : ChosenSigs) {
    Function *F = KV.first;
    const auto &ArgTypes = KV.second;
    bool Variadic = !F->Inputs.empty() && F->Inputs.back() == "varargin";
    // Leading params that take a positional arg. For a variadic callee
    // the trailing `varargin` slot is a runtime cell (typed !llvm.ptr by
    // the lowerer) and is never stamped — extra call args beyond the
    // leading params are packed into it.
    size_t LeadCount = Variadic ? F->Inputs.size() - 1 : F->Inputs.size();
    // Reduced-arity clones supply fewer args than declared params — stamp
    // just the supplied prefix. For non-variadic callees, bail when there
    // are MORE args than params (a user error the resolver already flags);
    // variadic callees legitimately have extra args (the packed tail).
    if (!Variadic && ArgTypes.size() > F->ParamRefs.size()) continue;
    // Resize stamps vector to match arity; fresh entries start null.
    if (F->ParamTypeStamps.size() != F->Inputs.size())
      F->ParamTypeStamps.assign(F->Inputs.size(), nullptr);
    size_t StampN = std::min(LeadCount, ArgTypes.size());
    for (size_t I = 0; I < StampN; ++I) {
      const Type *NewT = ArgTypes[I];
      if (!NewT) continue;
      // Only refine (don't widen). Refusing to overwrite a concrete
      // stamp with Any keeps the lattice monotonic across iterations.
      if (NewT->K == Type::Kind::Any) continue;
      // Non-scalar Array types: at the AST→MLIR boundary `mirTy` maps
      // these to `tensor<NxF64>`, but the downstream lowering pipeline
      // (LowerTensorOps + LowerIO + LowerUserCalls' iterative loop)
      // is built around matrix args/results flowing as `ptr` to a
      // runtime matlab_mat. Stamping tensor types onto func.func
      // signatures locks in a sig that the pipeline can't bridge to
      // the post-LowerTensorOps ptr operands. Skipping the matrix
      // stamps lets such call sites fall through to the late-MLIR
      // monomorphizer (runMonomorphiseUserCalls) which buckets by the
      // settled ptr type after materialization. The clone itself is
      // still useful (each call site already dispatches to its own
      // function symbol) — only the SIG stays `(none) -> (none)` so
      // the existing pipeline machinery handles the ptr refinement.
      if (NewT->K == Type::Kind::Array) {
        auto &AT = static_cast<const ArrayType &>(*NewT);
        if (AT.S.K != Shape::Rank::Scalar) continue;
      }
      if (F->ParamTypeStamps[I] == NewT) continue;
      // Skip if a different concrete type is already stamped (signals
      // an inconsistency the monomorphizer should have resolved into
      // a clone — leaving it stamped to the first wins keeps the
      // pipeline deterministic).
      if (F->ParamTypeStamps[I] &&
          F->ParamTypeStamps[I]->K != Type::Kind::Any &&
          F->ParamTypeStamps[I] != NewT)
        continue;
      F->ParamTypeStamps[I] = NewT;
      Stamped++;
    }
    // #40 Class 2: a reduced-arity clone's trailing positional params
    // receive no call-site arg. Pad them to scalar double (mirrors the
    // late mono's f64 padding of dead trailing params) so a body branch
    // guarded by `if nargin == N` that references them — dead for this
    // arity but still type-checked by TypeInference — doesn't poison the
    // result type with `none`. Stops before the varargin cell slot.
    for (size_t I = ArgTypes.size(); I < LeadCount; ++I) {
      if (!F->ParamTypeStamps[I]) {
        F->ParamTypeStamps[I] = TC.scalar(Dtype::Double);
        Stamped++;
      }
    }
  }
  return Stamped;
}

} // namespace

MonomorphizeStats runMonomorphize(
    TranslationUnit &TU, ASTContext &Ctx, TypeContext &TC,
    const std::function<void()> &runSemaPass, int MaxIters) {
  MonomorphizeStats Stats;
  ClonePlan Plan;

  for (int I = 0; I < MaxIters; ++I) {
    Stats.Iterations = I + 1;
    bool ChangedThisIter = false;

    // Phase 3 — grow the clone plan and rewrite call sites.
    int NewClones = growPlan(TU, Ctx, Plan);
    int Rewrites = applyPlan(TU, Ctx, Plan);
    Stats.ClonesCreated += NewClones;
    Stats.CallSitesRewritten += Rewrites;
    if (NewClones > 0 || Rewrites > 0) {
      ChangedThisIter = true;
      // Re-run Sema so new clones get bindings and rewritten NameExpr
      // names resolve to the right specialisations on the next analysis.
      runSemaPass();
    }

    // Phase 5 — stamp call-site ArgTypes onto each user Function's
    // current ParamRefs[i]->InferredType. Must run AFTER runSemaPass
    // (each Resolver invocation refreshes F.ParamRefs with new
    // Binding pointers, orphaning earlier stamps). The next Sema pass
    // honors a pre-existing concrete InferredType on each param
    // binding (see TypeInference.cpp's param-init branch), so body
    // expressions refine under the stamped arg types.
    int Stamped = stampSignatureTypes(TU, TC);
    if (Stamped > 0) {
      ChangedThisIter = true;
      // Refresh body types so the stamped params propagate through
      // arithmetic / calls inside the body. This may surface NEW
      // signatures inside clones (e.g. a recursive call whose args
      // were `any` last round are now concrete) — the next iteration
      // picks those up in growPlan.
      runSemaPass();
    }

    if (!ChangedThisIter) {
      // Fixpoint reached: nothing new added, rewritten, or stamped.
      Stats.Converged = true;
      return Stats;
    }
  }

  // Hit the iteration cap — caller should treat this as a soft bug
  // (likely unbounded polymorphic recursion). Stats reflect partial work.
  Stats.Converged = false;
  return Stats;
}

} // namespace matlab
