// DispatchDesynth.h — #191 P3. Sema-time AST rewrite that de-synthesizes class
// operator/method dispatch.
//
// Operator overloads on class instances (`G + H`, `2 * sys`, ...) are otherwise
// synthesized during lowering (Lowering.cpp), so Sema never sees them as real
// calls — which defeats inter-procedural inference (P2) and the monomorphizer
// (P5). This pass rewrites, in place, a `BinaryOpExpr` whose operand is a class
// instance into an explicit method call `a.<opmethod>(b)` (a FieldAccess-callee
// CallOrIndex), the form that already type-checks (P1.1) and lowers identically
// to the synthesized `Class__<opmethod>`. A non-class operand mixed with a
// class instance is boxed into a one-arg constructor call `Class(value)`.
//
// Gated per class via an allow-list (default empty -> no rewrite, no behavior
// change). The lowering synthesis stays as the fallback for classes not yet on
// the list; once a class is rewritten here, the lowering BinaryOp-on-object path
// no longer matches it, so there is no double-emit.
//
// Runs AFTER a first Resolver + TypeInference pass (it keys off the operand's
// inferred object<Class> type); the caller must re-run Resolver + TypeInference
// afterward so the synthesized nodes are resolved and typed. See
// docs/sema_p3_dispatch_desynth.md.

#pragma once

#include <set>
#include <string>

namespace matlab {
class ASTContext;
class TranslationUnit;

namespace sema {

// Rewrite operator dispatch on instances of the allow-listed classes into
// explicit method-call AST nodes. Returns the number of operator nodes
// rewritten (0 => the TU is unchanged). Safe to call with an empty allow-list.
//
// KeyOffPinnedClass: when true, also recover an operand's class from a
// NameExpr binding's PinnedClass (not just an object<Class> Expr->Ty),
// matching the lowering synthesis path (pinnedFromExpr). Safe and wanted in
// whole-program (AOT) compilation — where P2/P5 run — but must stay OFF for
// cross-turn -repl: there, a pinned-but-not-object operand desynthed into a
// method call whose base is a cross-turn binding crashes the dispatch lowering
// at runtime, and the synthesis fallback (taken when the rewrite is a no-op)
// handles it correctly.
int desynthDispatch(ASTContext &Ctx, TranslationUnit &TU,
                    const std::set<std::string> &AllowClasses,
                    bool KeyOffPinnedClass = false);

} // namespace sema
} // namespace matlab
