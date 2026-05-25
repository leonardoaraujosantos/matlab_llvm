#pragma once

//===----------------------------------------------------------------------===//
// CallSiteAnalyzer — Phase 1 of the Sema-time monomorphization epic (#38).
//
// Walks a TranslationUnit after TypeInference has run and groups every
// resolved user-function Call by the per-arg type signature seen at that
// call site. The result is a CallSiteAnalysis: a map from each user
// Function* to the set of distinct signatures it is called with.
//
// This pass is a pure analysis — it does not mutate the AST. Phase 3 of
// the epic consumes the analysis to drive AST cloning (one clone per
// signature variant).
//
// The signature key is the full Type::toString() of each arg, so two
// call sites with identical inferred types share a bucket but call sites
// with different shapes (e.g. scalar vs 1×3 vs 2×2) land in distinct
// buckets. That fidelity matters for Phase 3 — a later pass can choose
// to coalesce buckets that lower to the same MLIR type, but Phase 1
// preserves what TypeInference observed.
//===----------------------------------------------------------------------===//

#include "matlab/AST/AST.h"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace matlab {

class TranslationUnit;
class Function;
class NameExpr;

// A signature is one string per call-site arg. nullptr-typed args render
// as "?" so two calls with unresolved-but-equal arg slots dedupe cleanly.
using CallSignature = std::vector<std::string>;

struct CallSiteAnalysis {
  // Insertion-ordered map keyed by user Function*. Each callee maps to the
  // ordered set of distinct signatures seen across its call sites. We
  // intentionally keep the set sorted by signature text so dump output is
  // deterministic regardless of TU traversal order.
  std::map<Function *, std::set<CallSignature>> Sigs;

  // For golden / -dump-call-sites output: emit one line per callee, with
  // its signatures listed sorted underneath. Functions are emitted in
  // alphabetical name order for determinism.
  std::string dump() const;
};

// Render a single Type* as the per-arg signature string. Public so future
// phases (e.g. Phase 3's clone-name mangling) can share the same encoding.
std::string typeSignature(const Type *T);

// Render an entire ArgTypes tuple as a single CallSignature. Shared by
// analyzer and monomorphizer so call-site bucketing stays in lockstep.
CallSignature callSignatureFrom(const std::vector<const Type *> &ArgTypes);

// Run the analyzer over the translation unit. Requires that TypeInference
// has already populated CallOrIndex::ArgTypes on every user-function call.
CallSiteAnalysis analyzeCallSites(TranslationUnit &TU);

// Walk every CallOrIndex in the TU that resolves to a user-defined
// Function* (excluding builtins, indexing, and unresolved sites). The
// action callback receives the Call node, its NameExpr Callee, and the
// resolved Function*. Used by both the analyzer (record signatures) and
// the Phase 3 monomorphizer (rewrite NameExpr.Name to a clone mangling).
using UserCallAction =
    std::function<void(CallOrIndex &, NameExpr &, Function &)>;
void walkUserCalls(TranslationUnit &TU, const UserCallAction &Action);

} // namespace matlab
