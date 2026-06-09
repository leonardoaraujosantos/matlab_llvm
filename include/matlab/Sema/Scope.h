#pragma once

#include "matlab/Basic/SourceManager.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace matlab {

class Function;
class ClassDef;
class Type;

enum class BindingKind : uint8_t {
  Var,        // assigned somewhere in this function's body
  Param,      // input parameter
  Output,     // output parameter (must be assigned before function exits)
  Global,     // declared `global`
  Persistent, // declared `persistent`
  Function,   // TU-level or nested function (user-defined)
  Builtin,    // known MATLAB builtin (extern)
  Import,     // `import pkg.name` alias (unused for now)
  Class,      // user-defined classdef; ClassName(args) = constructor call
};

const char *bindingKindName(BindingKind K);

struct Binding {
  BindingKind Kind = BindingKind::Var;
  std::string_view Name;
  SourceLocation FirstUse;
  matlab::Function *FuncDef = nullptr; // for Kind=Function; null for Builtin
  matlab::ClassDef *ClassDef = nullptr; // for Kind=Class
  const Type *DeclaredType = nullptr;  // optional, e.g. for builtins
  // Type inference fills in the current inferred type here after analysis.
  const Type *InferredType = nullptr;
  /* For ordinary variables whose value is a class instance, the resolver
   * pins the ClassDef here so `obj.prop` / `obj.method(args)` can find
   * the property / method without dynamic dispatch. Populated when a Var
   * is the output of a `ClassName(args)` constructor call. */
  matlab::ClassDef *PinnedClass = nullptr;
  bool WrittenTo = false;
  bool ReadFrom = false;
  /* Phase 6 — Symbolic Math Toolbox. Stamped by the Resolver when the
   * REPL workspace-kind hook reports kind=7 (matlab_sym*). Read by the
   * MLIR lowering's NameExpr / BinaryOp / disp dispatch sites so a
   * cross-TU sym binding (declared in an earlier REPL input) routes
   * through matlab_sym_* the same as same-TU SymBindings tracking. */
  bool IsSym = false;
  /* Phase 6.1 — symbolic matrix (kind=8). Same cross-TU REPL story as
   * IsSym, but routes through matlab_ws_get_symmat / matlab_symmat_*. */
  bool IsSymmat = false;
  /* Plain matlab_struct* (kind=9). Set when the REPL workspace-kind
   * hook reports a struct binding from a prior input; consumed by the
   * MLIR lowering to re-populate StructInitialised so field accesses
   * (`s.x`) and re-assignments route through the struct-aware path
   * instead of the matrix one. */
  bool IsStruct = false;
  /* Function handle (kind=13).  Stamped by the Resolver when the REPL
   * workspace-kind hook reports a prior input bound this name to a
   * function handle (`f = @sin`).  The MLIR lowering reads it to route
   * the cross-turn workspace load through matlab_ws_get_handle and the
   * `f(x)` call through the matlab_call_handle_s* trampoline instead of
   * a matrix subscript. */
  bool IsHandle = false;
  /* Return-kind of the function the handle points at, recovered from the
   * kind=13 workspace signature on a later REPL turn (#119): -1 unknown,
   * 0 scalar (double), 1 matrix (matlab_mat*).  Lets a cross-turn
   * `f(vec)` call with a MATRIX argument dispatch to the matrix-argument
   * trampoline (matlab_call_handle_m* / _mm*) with the right result type,
   * instead of falling through to the scalar-only path and mis-lowering
   * the code pointer into a matrix subscript (SIGSEGV). */
  int8_t HandleRetKind = -1;
  /* Table (kind=6).  Stamped by the Resolver when the REPL workspace-kind
   * hook reports a prior input bound this name to a table (`T =
   * readtable(...)`).  The MLIR lowering reads it to re-populate
   * TableBindings so a cross-turn `height(T)` / `width(T)` / `T.col` /
   * `disp(T)` dispatches to the table runtime entries instead of the
   * generic matrix path (which fails the call-shape detector / crashes). */
  bool IsTable = false;
  /* VideoWriter handle (#236).  Stamped by the Resolver when the REPL
   * workspace-kind hook reports a prior input bound this name to a
   * VideoWriter (`v = VideoWriter(...)`).  The MLIR lowering reads it to
   * re-populate VideoWriterBindings so a cross-turn `v.FrameRate = ...` /
   * `v.Quality = ...` routes to the dedicated setter instead of the
   * struct-field path, which corrupted the opaque handle and crashed the
   * encoder on the next writeVideo. */
  bool IsVideoWriter = false;
  /* Timetable (#259).  Stamped by the Resolver when the REPL workspace-kind
   * hook reports a prior input bound this name to a timetable (a timetable is
   * stored generically and would otherwise be mis-stamped as a plain matrix /
   * table cross-turn).  The MLIR lowering reads it (via isTimetableBinding) so
   * a cross-turn `summary(TT)` / `head(TT)` / `TT.col` / disp dispatches to
   * the timetable path instead of backing off to "unsupported call shape". */
  bool IsTimetable = false;
  /* Struct array (kind=14).  Stamped by the Resolver when the REPL
   * workspace-kind hook reports a prior input bound this name to a
   * matlab_struct_arr* (`a(i).x = v`).  The MLIR lowering reads it to
   * re-populate StructArrayBindings and rehydrate the array pointer from
   * the workspace, so a cross-turn `a(i).x` / `length(a)` dispatches to
   * the struct-array path instead of allocating a fresh empty array. */
  bool IsStructArray = false;
  /* #116: true when a cross-REPL-turn binding holds a 3-D array
   * (matlab_mat3).  A 3-D value round-trips through the workspace under the
   * generic mat kind, so the next turn's Sema loses its rank and the N-D
   * subscript store/read detectors (`__subscript_store` 5-arg,
   * `A(:,:,k)`) back off to "unsupported call shape".  The Resolver stamps
   * this from the workspace-kind hook (kind=16) and the MLIR lowering reads
   * it (via isThreeDBinding) to re-seed the 3-D dispatch the way
   * ThreeDBindings does for same-TU 3-D values. */
  bool IsThreeD = false;
};

class Scope {
public:
  explicit Scope(Scope *Parent = nullptr, std::string Name = "")
      : Parent(Parent), Name(std::move(Name)) {}

  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

  // Lookup walks up the parent chain.
  Binding *lookup(std::string_view N);
  Binding *lookupLocal(std::string_view N);

  // Declare a new binding in this scope (no-op if already present with same
  // kind — returns existing binding in that case, promoting Var to keep Kind).
  Binding *declare(std::string_view N, BindingKind K, Binding *Owned);

  // Retrieve or auto-declare as Var (used at first assignment).
  Binding *getOrDeclareVar(std::string_view N, Binding *Owned);

  Scope *parent() const { return Parent; }
  const std::string &name() const { return Name; }

  // Iteration for the dumper.
  const std::unordered_map<std::string, Binding *> &locals() const {
    return Bindings;
  }

private:
  Scope *Parent;
  std::string Name;
  std::unordered_map<std::string, Binding *> Bindings;
};

// Owns all Sema-created objects (Scopes, Bindings, Types).
class SemaContext {
public:
  SemaContext();
  ~SemaContext();
  SemaContext(const SemaContext &) = delete;
  SemaContext &operator=(const SemaContext &) = delete;

  Scope *newScope(Scope *Parent, std::string Name = "");
  Binding *newBinding();

  // String interning for names whose storage must outlive the source buffer
  // (generally unnecessary since source outlives Sema, but available).
  std::string_view intern(std::string_view S);

private:
  std::vector<std::unique_ptr<Scope>> Scopes;
  std::vector<std::unique_ptr<Binding>> Bindings;
  std::vector<std::unique_ptr<std::string>> Strings;
};

} // namespace matlab
