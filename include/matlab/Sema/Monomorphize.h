#pragma once

//===----------------------------------------------------------------------===//
// Monomorphize — Phases 3 and 4 of the Sema-time monomorphization epic (#38).
//
// Phase 3 (specialization driver): given a Sema'd TU, find every user
// Function* whose call sites use >1 distinct arg-type signatures. For
// each non-canonical signature, deep-clone the function with a mangled
// name (`<name>__s1`, `__s2`, …), append the clone to TU.Functions, and
// rewrite the corresponding call sites' NameExpr.Name to the mangled
// name. After a subsequent Resolver re-run, each call site's
// NameExpr.Ref points at the right specialization.
//
// Phase 4 (recursion closure): the bodies of clones may contain calls
// to other polymorphic helpers (or to themselves). Each iteration's
// rewrite may surface new signatures, so we iterate to a fixpoint —
// bounded by MaxIters to guard against unbounded polymorphic recursion.
//
// API contract:
//   - Caller has already run Resolver + TypeInference on TU before the
//     first invocation (the analyzer reads ArgTypes annotated by the
//     prior TypeInference run).
//   - Caller provides `runSemaPass` — a closure that re-runs Resolver +
//     TypeInference over TU. Invoked after each iteration that creates
//     clones, so the next iteration's analyzer sees fresh Refs / ArgTypes
//     on cloned bodies.
//   - On return, TU contains clones and the caller's Sema state reflects
//     the final pass (the last `runSemaPass` invocation).
//===----------------------------------------------------------------------===//

#include <functional>

namespace matlab {

class TranslationUnit;
class ASTContext;
class TypeContext;

struct MonomorphizeStats {
  int Iterations = 0;        // number of fixpoint iterations actually run
  int ClonesCreated = 0;     // total clones appended to TU.Functions
  int CallSitesRewritten = 0; // total NameExpr.Name rewrites
  bool Converged = true;     // false if MaxIters reached without convergence
};

MonomorphizeStats runMonomorphize(
    TranslationUnit &TU,
    ASTContext &Ctx,
    TypeContext &TC,
    const std::function<void()> &runSemaPass,
    int MaxIters = 8);

} // namespace matlab
