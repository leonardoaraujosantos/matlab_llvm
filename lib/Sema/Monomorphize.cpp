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
  CallSiteAnalysis Analysis = analyzeCallSites(TU);

  // Group call sites by their ROOT function (so clones-of-the-same-root
  // contribute to the same signature table). For each root, we accumulate
  // the union of all signatures seen at any of its (root or clone) calls.
  std::map<Function *, std::set<CallSignature>> ByRoot;
  for (auto &KV : Analysis.Sigs) {
    Function *F = KV.first;
    Function *Root = rootOf(Plan, F);
    auto &Bucket = ByRoot[Root];
    for (const CallSignature &Sig : KV.second) Bucket.insert(Sig);
  }

  int NewClones = 0;
  for (auto &KV : ByRoot) {
    Function *Root = KV.first;
    const std::set<CallSignature> &Bucket = KV.second;
    if (Bucket.size() <= 1) continue; // monomorphic

    SignatureToName &Table = Plan.ByRoot[Root];
    int NextSuffix = static_cast<int>(Table.size()); // 0 means "canonical name"
    for (const CallSignature &Sig : Bucket) {
      if (Table.count(Sig)) continue; // already named
      std::string Name;
      if (NextSuffix == 0) {
        // Canonical signature — keeps the root's original name. Do NOT
        // clone; the original Function* is reused for this signature.
        Name = std::string(Root->Name);
      } else {
        Name = std::string(Root->Name) + "__s" + std::to_string(NextSuffix);
        Function *Clone = cloneFunction(Ctx, *Root, Name);
        TU.Functions.push_back(Clone);
        Plan.CloneToRoot[Clone] = Root;
        NewClones++;
      }
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
int stampSignatureTypes(TranslationUnit &TU) {
  std::map<Function *, std::vector<const Type *>> ChosenSigs;
  walkUserCalls(TU, [&](CallOrIndex &C, NameExpr & /*N*/, Function &F) {
    if (ChosenSigs.count(&F)) return;
    ChosenSigs[&F] = C.ArgTypes;
  });

  int Stamped = 0;
  for (auto &KV : ChosenSigs) {
    Function *F = KV.first;
    const auto &ArgTypes = KV.second;
    if (F->ParamRefs.size() != ArgTypes.size()) continue;
    // Resize stamps vector to match arity; fresh entries start null.
    if (F->ParamTypeStamps.size() != F->Inputs.size())
      F->ParamTypeStamps.assign(F->Inputs.size(), nullptr);
    for (size_t I = 0; I < ArgTypes.size(); ++I) {
      const Type *NewT = ArgTypes[I];
      if (!NewT) continue;
      // Only refine (don't widen). Refusing to overwrite a concrete
      // stamp with Any keeps the lattice monotonic across iterations.
      if (NewT->K == Type::Kind::Any) continue;
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
  }
  return Stamped;
}

} // namespace

MonomorphizeStats runMonomorphize(
    TranslationUnit &TU, ASTContext &Ctx,
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
    int Stamped = stampSignatureTypes(TU);
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
