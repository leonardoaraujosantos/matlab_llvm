#pragma once

//===----------------------------------------------------------------------===//
// AST Cloner — Phase 2 of the Sema-time monomorphization epic (#38).
//
// Polymorphic deep-copy over the AST hierarchy. Used by the Phase 3
// driver to clone a user `Function` once per signature variant before
// Sema is re-run on the augmented TU. Clones are allocated through
// ASTContext (same bump allocator as the original parse).
//
// What gets cloned:
//   - The structural tree: every Expr / Stmt / Block / Function node.
//   - Owning sub-vectors (Block::Stmts, MatrixLiteral::Rows, etc.).
//   - Source-buffer-backed string_views (Inputs, literal Text, etc.) are
//     shallow-copied — the source buffer outlives the AST so they remain
//     valid in the clone.
//
// What does NOT get cloned (zeroed instead):
//   - Sema-populated pointers: Binding* (NameExpr::Ref, ForStmt::VarRef,
//     TryStmt::CatchVarRef, Function::Self / FnScope / ParamRefs /
//     OutputRefs / Nested-binding info, ClassDef::Self).
//   - TypeInference annotations: Expr::Ty and CallOrIndex::ArgTypes.
//   - The CallKind on CallOrIndex (re-resolved by Resolver).
//
// Re-running Resolver + TypeInference on a TU containing the clone is
// expected to yield the same Sema state on the clone as on the original.
// The Phase 2 acceptance test exercises that round-trip.
//===----------------------------------------------------------------------===//

#include "matlab/AST/AST.h"

#include <string_view>

namespace matlab {

// Deep-copy a Function and assign it `NewName` (interned through Ctx so
// the clone outlives the caller's local std::string). All Sema-populated
// pointers on the clone are nullptr; Expr::Ty is cleared. Caller is
// responsible for re-running Resolver + TypeInference on the TU.
Function *cloneFunction(ASTContext &Ctx, const Function &Src,
                        std::string_view NewName);

// Deep-copy an arbitrary Expr / Stmt sub-tree. Exposed for tests; the
// monomorphizer should normally go through cloneFunction.
Expr *cloneExpr(ASTContext &Ctx, const Expr *E);
Stmt *cloneStmt(ASTContext &Ctx, const Stmt *S);

} // namespace matlab
