#include "matlab/MLIR/Lowering.h"

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/MLIR/Context.h"
#include "matlab/MLIR/TypeMapper.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_os_ostream.h"

#include <cmath>
#include <limits>

#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace matlab {
namespace mlirgen {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

int64_t foldInt(const Expr *E) {
  if (!E) return 0;
  if (auto *L = dynamic_cast<const IntegerLiteral *>(E)) {
    try { return std::stoll(std::string(L->Text)); } catch (...) { return 0; }
  }
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(E)) {
    if (U->Op == UnOp::Minus) return -foldInt(U->Operand);
    if (U->Op == UnOp::Plus)  return  foldInt(U->Operand);
  }
  return 0;
}

double foldFloat(const Expr *E) {
  if (!E) return 0.0;
  if (auto *L = dynamic_cast<const FPLiteral *>(E)) {
    try { return std::stod(std::string(L->Text)); } catch (...) { return 0.0; }
  }
  if (dynamic_cast<const IntegerLiteral *>(E)) return (double)foldInt(E);
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(E)) {
    if (U->Op == UnOp::Minus) return -foldFloat(U->Operand);
    if (U->Op == UnOp::Plus)  return  foldFloat(U->Operand);
  }
  return 0.0;
}

//===----------------------------------------------------------------------===//
// Fixed-Point Designer (fi) helpers — see docs/emit_fixed_point.md.
//===----------------------------------------------------------------------===//

// Build the named-attribute set that every fi-tagged op carries.
// LowerFixedPoint reads these to drive its rewrite. Keys (all integer-
// valued except the marker):
//   "fi"        : i1   marker (always 1)
//   "fi_signed" : i1
//   "fi_wl"     : i32  word length
//   "fi_fl"     : i32  fraction length
//   "fi_of"     : i32  overflow mode (0=Wrap, 1=Saturate)
//   "fi_rm"     : i32  rounding mode (0=Floor, 1=Nearest, ...)
llvm::SmallVector<mlir::NamedAttribute, 6>
buildFixedAttrs(mlir::MLIRContext *Ctx, const FixedSpec &S) {
  auto I1 = mlir::IntegerType::get(Ctx, 1);
  auto I32 = mlir::IntegerType::get(Ctx, 32);
  llvm::SmallVector<mlir::NamedAttribute, 6> Attrs;
  Attrs.emplace_back(mlir::StringAttr::get(Ctx, "fi"),
                     mlir::IntegerAttr::get(I1, 1));
  Attrs.emplace_back(mlir::StringAttr::get(Ctx, "fi_signed"),
                     mlir::IntegerAttr::get(I1, S.Signed ? 1 : 0));
  Attrs.emplace_back(mlir::StringAttr::get(Ctx, "fi_wl"),
                     mlir::IntegerAttr::get(I32, (int64_t)S.WordLength));
  Attrs.emplace_back(mlir::StringAttr::get(Ctx, "fi_fl"),
                     mlir::IntegerAttr::get(I32, (int64_t)S.FractionLength));
  Attrs.emplace_back(mlir::StringAttr::get(Ctx, "fi_of"),
                     mlir::IntegerAttr::get(I32, (int64_t)S.OF));
  Attrs.emplace_back(mlir::StringAttr::get(Ctx, "fi_rm"),
                     mlir::IntegerAttr::get(I32, (int64_t)S.RM));
  return Attrs;
}

// Compile-time quantize: real-world value -> stored integer for the spec.
// Mirrors matlab_fi_quantize_s/u in the runtime; used by Lowering.cpp's
// `fi(literal, …)` constant fold so the emitted IR has an `arith.constant`
// of the stored value rather than a runtime call.
int64_t quantizeFixedSigned(double v, const FixedSpec &S) {
  double scaled = std::ldexp(v, S.FractionLength);
  int64_t stored;
  switch (S.RM) {
  case FixedSpec::Rounding::Nearest:
    stored = (int64_t)std::floor(scaled + 0.5);
    break;
  case FixedSpec::Rounding::Zero:
    stored = (int64_t)std::trunc(scaled);
    break;
  case FixedSpec::Rounding::Ceiling:
    stored = (int64_t)std::ceil(scaled);
    break;
  case FixedSpec::Rounding::Convergent: {
    double frac = scaled - std::floor(scaled);
    if (frac == 0.5) {
      int64_t lo = (int64_t)std::floor(scaled);
      stored = (lo % 2 == 0) ? lo : lo + 1;
    } else {
      stored = (int64_t)std::round(scaled);
    }
    break;
  }
  case FixedSpec::Rounding::Floor:
  default:
    stored = (int64_t)std::floor(scaled);
    break;
  }
  if (S.OF == FixedSpec::Overflow::Saturate) {
    if (S.WordLength >= 64) return stored;
    int64_t hi = ((int64_t)1 << (S.WordLength - 1)) - 1;
    int64_t lo = -((int64_t)1 << (S.WordLength - 1));
    if (stored > hi) stored = hi;
    if (stored < lo) stored = lo;
    return stored;
  }
  // Wrap: mask to WL bits then sign-extend.
  if (S.WordLength == 0) return 0;
  if (S.WordLength >= 64) return stored;
  uint64_t mask = ((uint64_t)1 << S.WordLength) - 1u;
  uint64_t bits = ((uint64_t)stored) & mask;
  if (bits & ((uint64_t)1 << (S.WordLength - 1))) bits |= ~mask;
  return (int64_t)bits;
}

uint64_t quantizeFixedUnsigned(double v, const FixedSpec &S) {
  double scaled = std::ldexp(v, S.FractionLength);
  if (scaled < 0.0) scaled = 0.0;
  uint64_t stored;
  switch (S.RM) {
  case FixedSpec::Rounding::Nearest:
    stored = (uint64_t)std::floor(scaled + 0.5);
    break;
  case FixedSpec::Rounding::Zero:
    stored = (uint64_t)std::trunc(scaled);
    break;
  case FixedSpec::Rounding::Ceiling:
    stored = (uint64_t)std::ceil(scaled);
    break;
  case FixedSpec::Rounding::Convergent: {
    double frac = scaled - std::floor(scaled);
    if (frac == 0.5) {
      uint64_t lo = (uint64_t)std::floor(scaled);
      stored = (lo % 2 == 0) ? lo : lo + 1;
    } else {
      stored = (uint64_t)std::round(scaled);
    }
    break;
  }
  case FixedSpec::Rounding::Floor:
  default:
    stored = (uint64_t)std::floor(scaled);
    break;
  }
  if (S.OF == FixedSpec::Overflow::Saturate) {
    if (S.WordLength >= 64) return stored;
    uint64_t hi = ((uint64_t)1 << S.WordLength) - 1u;
    if (stored > hi) stored = hi;
    return stored;
  }
  if (S.WordLength == 0) return 0;
  if (S.WordLength >= 64) return stored;
  uint64_t mask = ((uint64_t)1 << S.WordLength) - 1u;
  return stored & mask;
}

/* Walk an anon function body collecting NameExpr bindings that refer to
 * values defined OUTSIDE the anon — i.e. captures. Params are filtered
 * out (they resolve against the block args), and so are builtins and
 * user functions (which don't need a capture slot; their call lowering
 * routes through the @name path or a direct call).
 *
 * Out is populated with bindings in first-seen order; Seen deduplicates
 * across multiple references to the same capture. Unknown expression
 * kinds simply aren't recursed into — a capture hiding inside an
 * unrecognised expr will still be lowered as a fresh lazy slot at the
 * read site, which loses the value but doesn't crash. */
void collectCaptures(const Expr *E,
                     const std::vector<Binding *> &Params,
                     std::vector<Binding *> &Out,
                     std::unordered_set<Binding *> &Seen) {
  if (!E) return;
  switch (E->Kind) {
  case NodeKind::NameExpr: {
    auto *N = static_cast<const NameExpr *>(E);
    if (!N->Ref) return;
    for (Binding *P : Params) if (P == N->Ref) return;
    if (N->Ref->Kind == BindingKind::Builtin ||
        N->Ref->Kind == BindingKind::Function) return;
    if (!Seen.insert(N->Ref).second) return;
    Out.push_back(N->Ref);
    return;
  }
  case NodeKind::BinaryOp: {
    auto *B = static_cast<const BinaryOpExpr *>(E);
    collectCaptures(B->LHS, Params, Out, Seen);
    collectCaptures(B->RHS, Params, Out, Seen);
    return;
  }
  case NodeKind::UnaryOp: {
    auto *U = static_cast<const UnaryOpExpr *>(E);
    collectCaptures(U->Operand, Params, Out, Seen);
    return;
  }
  case NodeKind::PostfixOp: {
    auto *P = static_cast<const PostfixOpExpr *>(E);
    collectCaptures(P->Operand, Params, Out, Seen);
    return;
  }
  case NodeKind::RangeExpr: {
    auto *R = static_cast<const RangeExpr *>(E);
    collectCaptures(R->Start, Params, Out, Seen);
    collectCaptures(R->Step,  Params, Out, Seen);
    collectCaptures(R->End,   Params, Out, Seen);
    return;
  }
  case NodeKind::CallOrIndex: {
    auto *C = static_cast<const CallOrIndex *>(E);
    collectCaptures(C->Callee, Params, Out, Seen);
    for (const Expr *A : C->Args) collectCaptures(A, Params, Out, Seen);
    return;
  }
  case NodeKind::MatrixLiteral: {
    auto *M = static_cast<const MatrixLiteral *>(E);
    for (auto &Row : M->Rows)
      for (const Expr *A : Row) collectCaptures(A, Params, Out, Seen);
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// Lowerer
//===----------------------------------------------------------------------===//

class Lowerer {
public:
  Lowerer(mlir::MLIRContext &MCtx, TypeContext &TC, DiagnosticEngine &Diag,
          const SourceManager *SM = nullptr, bool ReplMode = false,
          bool DebugMode = false)
      : MCtx(MCtx), TC(TC), Diag(Diag), SM(SM), B(&MCtx),
        ReplMode(ReplMode), DebugMode(DebugMode) {
    (void)this->Diag;
    (void)this->SM;
  }

  mlir::ModuleOp lower(const TranslationUnit &TU);

private:
  mlir::MLIRContext &MCtx;
  TypeContext &TC;
  DiagnosticEngine &Diag;
  const SourceManager *SM;
  mlir::OpBuilder B;
  /* When true, script-level Var reads/writes route through
   * matlab_ws_get_* / matlab_ws_set_* so variables persist across
   * JIT invocations. Function bodies inside the same TU still use
   * local slots. */
  bool ReplMode = false;
  /* True while lowering the script body (not inside any user function).
   * REPL rerouting only fires when this is true AND ReplMode is on,
   * so Vars declared inside a user function keep their slot-local
   * semantics. */
  bool InScriptBody = false;
  /* #286: the CallOrIndex node currently being lowered as the bare,
   * non-suppressed top-level expression of an ExprStmt (the implicit
   * `ans = ...` display path). Only this exact node gets its open
   * (NoneType) user-function result type nudged to a concrete type so the
   * display fires — every other call site is left untouched, keeping the
   * fix's blast radius to one statement. nullptr outside that path. */
  const void *BareDisplayCall = nullptr;
  /* Set during lower() so lowerScript can iterate the TU's classdefs
   * to emit class-name registrations. nullptr outside lower(). */
  const TranslationUnit *CurTU = nullptr;
  /* When true, inject matlab_dbg_hook(file_id, line) at every
   * top-level / function-body statement. The file_id comes from the
   * SourceManager's FileID so the DAP server can resolve back to the
   * source path via matlab_dbg_register_file. */
  bool DebugMode = false;

  /* Push/pop a runtime frame around the body of a user function. The
   * lowered call carries a const_char with the displayed name (e.g.
   * "fact" or "MyClass.foo"); LowerTensorOps converts that into a
   * (ptr, length) pair for matlab_dbg_enter_frame. The leave call is
   * a zero-arg builtin. Both are no-ops outside DebugMode. */
  void emitDbgEnterFrame(llvm::StringRef Name, mlir::Location L);
  void emitDbgLeaveFrame(mlir::Location L);

  // Per-function: binding -> slot (Value result of matlab.alloc).
  std::unordered_map<Binding *, mlir::Value> Slots;

  // Bindings known to hold function handles. Populated when we see an
  // assignment whose RHS is an AnonFunction / FuncHandle. Used at
  // CallOrIndex lowering time to emit matlab.call_indirect instead of
  // matlab.subscript for `f(x)` where f is a handle variable.
  //
  // The vector is the list of capture SPILL SLOTS (in the outer function)
  // that must be loaded and prepended to each call_indirect's argument
  // list — so that @(x) x + k calls still see the value k had at @ time.
  // Empty vector = no captures (plain @name handles or capture-free anons).
  std::unordered_map<Binding *, std::vector<mlir::Value>> HandleBindings;

  // #77: a handle/anon binding ASSIGNED in this compilation unit (an entry
  // in HandleBindings, capture-free or not) must stay on the local-slot
  // lane — the same lane the static (AOT) path uses. That lane already
  // lowers every handle shape correctly: direct calls, matrix-argument
  // calls (`f(vec)`), passing a handle to a solver builtin (`lsqnonlin(f,…)`),
  // and captured closures (capture spill slots threaded onto each
  // call_indirect). In ReplMode the binding would otherwise be routed
  // through the workspace (matlab_ws_set/get_handle for capture-free
  // handles — #87 — or _mat for captured ones), which severs the
  // make_anon→addressof→call_indirect chain and breaks all of the above.
  // A workspace round-trip is only needed to recover a handle DEFINED in a
  // PRIOR REPL turn — that binding has no HandleBindings entry this turn
  // (it carries Binding::IsHandle from the kind=13 workspace lookup
  // instead), so it correctly stays on the kind=13 path.
  // Only *anonymous* closures defined this unit take the slot lane. A
  // NAMED handle (`@sin`, `@myCube`) is tracked in HandleTargetRef and
  // must keep the #87 path: it resolves the call by callee NAME (a
  // direct matlab.call for a user function, the kind=13 trampoline for a
  // builtin) — lowering `@myCube` to a `func.constant` in a slot would
  // freeze the callee's pre-refinement `(none)->none` type and fail
  // verification once RefineFuncSigs rewrites it to `(f64)->f64`.
  bool isLocalHandle(Binding *B) const {
    return B && HandleBindings.find(B) != HandleBindings.end() &&
           HandleTargetRef.find(B) == HandleTargetRef.end();
  }

  // #77: script-scope Var bindings whose last workspace store routed
  // through matlab_ws_set_mat (a matrix value). A workspace-backed matrix
  // has no local slot, so an anon that captures it (`@(s) M*s`) can't read
  // the slot's concrete tensor type — Sema's InferredType is often still
  // `any`/scalar for a `M = [..]` script assignment, so the capture
  // defaulted to f64 and the outlined anon got an f64 capture arg with a
  // tensor body (llvm.return type mismatch, #4). Tracking the matrix store
  // lets the capture load via matlab_ws_get_mat (ptr) instead.
  std::unordered_set<Binding *> MatrixWsBindings;

  // #77: a fixed-point (`fi`) script var holds an integer-encoded Q-format
  // value; its arithmetic lowers to integer shifts/muls (LowerFixedPoint).
  // Routing it through the workspace in ReplMode stores/loads it as a
  // matrix ptr (matlab_ws_get_mat), so a later fi op gets a !llvm.ptr
  // operand where it needs an integer (`arith.shrsi(!llvm.ptr, i32)` ->
  // verifier failure, #77). Like a captured closure, a fi binding has no
  // workspace representation that round-trips, so keep it on the
  // local-slot lane (the static/AOT path) in ReplMode.
  static const FixedSpec *fixedSpecOf(const Type *T) {
    if (!T || T->K != Type::Kind::Array) return nullptr;
    auto &AT = static_cast<const ArrayType &>(*T);
    return (AT.Elt == Dtype::Fixed && AT.FxSpec) ? &(*AT.FxSpec) : nullptr;
  }
  // Script vars proven to hold a fixed-point value. A binding's
  // InferredType/DeclaredType is often NOT fi-typed even for `x = fi(...)`
  // (Sema leaves it `any`/scalar, same unreliability as the matrix case),
  // so the spec is only reliably visible on the assignment LHS type (N.Ty)
  // at the store. We record the binding there and consult this set on the
  // read side, where N.Ty is unavailable.
  std::unordered_set<Binding *> FiBindings;
  bool isFiBinding(Binding *B) const {
    return B && (FiBindings.count(B) || fixedSpecOf(B->InferredType) ||
                 fixedSpecOf(B->DeclaredType));
  }
  // Resolved target binding for a binding that holds a *named* function
  // handle (`h = @inc` -> inc's binding).  Lets a handle stored into a
  // struct field / property be resolved back to its callee (#81), and
  // gives the callee's declared output arity for multi-return handle
  // calls (#80).  Only named handles (FuncHandle, or a copy of another
  // tracked named-handle binding) are recorded; anon closures with
  // captures are not.  Ref->Name is the callee; Ref->FuncDef (when set)
  // carries the output refs.
  std::unordered_map<Binding *, Binding *> HandleTargetRef;

  // (struct/obj binding, field name) -> resolved callee binding, for a
  // named function handle stored in a field (#81: `s.h = @inc;
  // v = s.h(5)` / `obj.StepFcn = @step`).  At the call site `s.h(args)`
  // resolves to a direct `matlab.call @<name>` instead of leaving an
  // unconverted matlab.subscript on the field-loaded handle.
  std::map<std::pair<Binding *, std::string>, Binding *> FieldHandleBindings;

  // Side map populated inside the AnonFunction lowering so the enclosing
  // AssignStmt can link the resulting capture slot list to the LHS binding.
  // Keyed by the AnonFunction AST node; cleared after use.
  std::unordered_map<const AnonFunction *,
                     std::vector<mlir::Value>> PendingCaptures;

  // Map from global/persistent BINDING to a slot-ID used by the runtime's
  // matlab_global_{get,set}_f64 helpers. IDs are assigned in first-seen
  // order and are module-global so every function that declares the same
  // global shares its slot. Persistent bindings are namespaced per
  // declaring function; the map key is the distinct Binding instance and
  // the ID space is shared with globals — both go through the same
  // runtime table.
  std::unordered_map<Binding *, int32_t> GlobalIds;
  // Name -> ID for global bindings so different functions declaring
  // the same `global x` share a slot even though each function has its
  // own Binding for x.
  std::unordered_map<std::string, int32_t> GlobalIdByName;
  int32_t NextGlobalId = 0;

  // Stack of (base, dim) contexts for `end` resolution inside subscripts.
  // Each entry represents the subscript arg currently being lowered:
  //   base = the matrix being indexed (already-lowered SSA value)
  //   dim  = 1-based position of this arg in the subscript.
  // When an EndExpr is lowered, the top of the stack provides operands for
  // the emitted matlab.end op so the tensor-ops pass can rewrite it to a
  // runtime matlab_end_of_dim call.
  std::vector<std::pair<mlir::Value, int64_t>> SubscriptCtx;

  /* Per-loop state for break/continue lowering. Each loop that
   * contains a break or continue allocates two i1 slots (did_break,
   * did_continue). matlab.break / matlab.continue write true to the
   * top-of-stack slot; the body restructuring wraps statements after
   * a break-/continue-containing stmt in an scf.if guarded by
   * !did_break && !did_continue so their side effects are skipped.
   * The enclosing loop's cond consumes did_break to exit. */
  struct LoopCtx {
    mlir::Value BreakSlot;
    mlir::Value ContinueSlot;
  };
  std::vector<LoopCtx> LoopStack;

  /* GPU Coder pragma state.  Set by `coder.gpu.kernelfun()` (whole
   * function) or `coder.gpu.kernel` (next for-loop only).  Consulted
   * in the ForStmt arm to emit `matlab.gpu.kernel` instead of
   * `matlab.for`.  Reset at lowerFunction entry. */
  bool InGpuKernelfun     = false;
  bool NextForIsGpuKernel = false;

  //--- location / type helpers
  mlir::Location loc(SourceLocation L) const;
  mlir::Location loc(SourceRange R) const { return loc(R.Begin); }
  mlir::Type mirTy(const Type *T) const;

  //--- emission helpers
  mlir::Value emitUnreg(llvm::StringRef OpName,
                        llvm::ArrayRef<mlir::Value> Operands,
                        mlir::Type ResultType, mlir::Location Loc,
                        llvm::ArrayRef<mlir::NamedAttribute> Attrs = {});

  mlir::Operation *emitUnregOp(llvm::StringRef OpName,
                               llvm::ArrayRef<mlir::Value> Operands,
                               llvm::ArrayRef<mlir::Type> ResultTypes,
                               mlir::Location Loc,
                               llvm::ArrayRef<mlir::NamedAttribute> Attrs = {},
                               unsigned NumRegions = 0);

  mlir::Value emitAlloc(const Type *T, llvm::StringRef Name, mlir::Location Loc);
  mlir::Value emitLoad(mlir::Value Slot, mlir::Type Ty, mlir::Location Loc);
  void        emitStore(mlir::Value V, mlir::Value Slot, mlir::Location Loc);

  //--- top-level
  void lowerScript(const Script &S, mlir::ModuleOp M);
  void lowerFunction(const Function &F, mlir::ModuleOp M,
                     const ClassDef *Owner = nullptr,
                     bool IsStatic = false);
  void lowerClass(const ClassDef &C, mlir::ModuleOp M);

  //--- blocks / stmts / exprs
  void lowerBlock(const ::matlab::Block &B);
  void lowerStmt(const Stmt &St);
  /* Walk a statement (including nested if/for/while bodies) for
   * matlab.break or matlab.continue. Used by ForStmt/WhileStmt
   * lowering to decide whether to emit the did_break/did_continue
   * flag plumbing. */
  bool stmtContainsBreakOrContinue(const Stmt &St);
  bool blockContainsBreakOrContinue(const ::matlab::Block &Blk);
  /* Lower statements of a loop body, inserting scf.if-guarded tails
   * after any stmt that contains break/continue so remaining work is
   * skipped once a flag is set. */
  void lowerLoopBody(const ::matlab::Block &Blk);
  mlir::Value lowerExpr(const Expr &E);
  void lowerLValueStore(const Expr &LHS, mlir::Value Rhs);
  /* Coerce an scf.if condition value to i1, regardless of the source
   * type. Float / wide-integer conds get an explicit cmpf-or-cmpi
   * against zero. `none`-typed conds (which arise when the cond is a
   * load of an `any`-typed slot, or a function param whose type
   * refines later) get an unrealized_conversion_cast as a verifier
   * placeholder. The RefineIfConds fixup pass that runs in the SV
   * pipeline after type-flow refinement replaces the placeholder
   * with a real cmpi/cmpf once the source type lands. */
  mlir::Value fixupIfCond(mlir::OpBuilder &B, mlir::Value Cond,
                          mlir::Location LC);

  //--- op-kind translation
  llvm::StringRef binOpName(BinOp O);
  llvm::StringRef unOpName(UnOp O);
  llvm::StringRef postfixName(PostfixOp O);

  mlir::Value loadBinding(Binding *Bnd, const Type *ValTy, mlir::Location L);

  int32_t globalSlotId(Binding *Bnd);
  mlir::Value ensureStructSlot(Binding *Bnd, std::string_view Name,
                                mlir::Location L);
  mlir::Value emitFieldNameChar(std::string_view Name, mlir::Location L);
  /* Resolve a struct-valued base expression to a ptr-typed struct
   * pointer. Handles NameExpr (via ensureStructSlot + load) and
   * chained FieldAccess (via matlab_struct_get_child_struct so
   * intermediate struct fields auto-allocate for s.a.b = v). Returns
   * a null Value when the base isn't resolvable to a struct. */
  mlir::Value resolveStructBase(const Expr *E, mlir::Location L);
  /* Bindings that have been initialised to a fresh matlab_struct_new().
   * Tracked per-Binding so a function with multiple FieldAccess sites
   * only initialises once. */
  std::unordered_set<Binding *> StructInitialised;
  /* Bindings introduced by `catch ME` — when `ME.<field>` is accessed
   * we route known fields (like `message`) to dedicated runtime
   * entries instead of the generic struct-get path, since the error
   * info lives outside a real matlab_struct. */
  std::unordered_set<Binding *> CatchBindings;
  /* Phase 2: bindings that hold a matlab_struct_arr * (any binding
   * that has been the base of an `s(i).x = ...` assignment). The
   * presence in this set switches read paths (`s(i).x`, `length(s)`,
   * `numel(s)`) over to the struct_arr runtime entries. */
  std::unordered_set<Binding *> StructArrayBindings;
  /* True if a binding holds a struct array — same-turn (in
   * StructArrayBindings, set by the `a(i).x = v` store) OR cross-turn
   * (Binding::IsStructArray, stamped by the Resolver from the kind=14
   * workspace lookup, #133). Lets the dispatch / rehydrate sites recover
   * the struct-array-ness a later REPL turn would otherwise lose. */
  bool isStructArrayBinding(Binding *B) const {
    return B && (StructArrayBindings.count(B) || B->IsStructArray);
  }
  /* Phase 4: bindings holding a matlab_dict * (assigned from
   * `containers.Map()` or `dictionary(...)`). Indexing reads /
   * writes route through the matlab_dict_* runtime entries. */
  std::unordered_set<Binding *> DictBindings;
  /* Phase 5.1: bindings holding a matlab_datetime * / matlab_duration *
   * pointer. Used by disp / arithmetic dispatch. */
  std::unordered_set<Binding *> DatetimeBindings;
  std::unordered_set<Binding *> DurationBindings;
  /* Phase 5.4: bindings holding a matlab_datetime_vec * /
   * matlab_duration_vec *. Produced by matrix-typed unit
   * constructors (`days(0:251)`, `hours(v)`, ...), scalar+vec or
   * vec+vec arithmetic, and timetable RowTimes access. Arithmetic
   * dispatch routes the `_vec` runtime entries; disp / length /
   * indexing follow the matlab_*_vec_ family. */
  std::unordered_set<Binding *> DatetimeVecBindings;
  std::unordered_set<Binding *> DurationVecBindings;
  /* Phase 5.2: bindings holding a matlab_categorical * — used to
   * dispatch disp / categories / iscategory / equality through the
   * dedicated runtime entries. */
  std::unordered_set<Binding *> CategoricalBindings;
  /* Bindings holding a matlab_videowriter * (from `v = VideoWriter(...)`).
   * Used so a `v.FrameRate = ...` / `v.Quality = ...` property store routes
   * to the video-writer setters instead of the generic struct field path
   * (which would misread the opaque handle as a struct). */
  std::unordered_set<Binding *> VideoWriterBindings;
  /* #236: same-TU (VideoWriterBindings, set at the `v = VideoWriter(...)`
   * assignment) OR cross-turn (Binding::IsVideoWriter, stamped by the Resolver
   * from the kind=15 workspace hook). */
  bool isVideoWriterBinding(Binding *B) const {
    return B && (VideoWriterBindings.count(B) || B->IsVideoWriter);
  }
  /* Phase 5.3: bindings holding a matlab_table * — used to dispatch
   * column accessors (`T.x`), shape (height/width/size), and disp(T). */
  std::unordered_set<Binding *> TableBindings;
  /* True if a binding holds a table — same-turn (in TableBindings, set by
   * the `T = readtable(...)` assignment) OR cross-turn (Binding::IsTable,
   * stamped by the Resolver from the kind=6 workspace lookup, #116). The
   * read path already returns the table ptr via matlab_ws_get_mat (kind=6
   * pass-through); this lets the dispatch sites recover the table-ness a
   * later REPL turn would otherwise lose. */
  bool isTableBinding(Binding *B) const {
    return B && (TableBindings.count(B) || B->IsTable);
  }
  /* Phase 5.4 (cont.): bindings holding a matlab_timetable * — same
   * column-store ABI as table, plus a RowTimes axis. Constructed by
   * `timetable(col1, ..., 'RowTimes', dt)` or `table2timetable(T,
   * 'RowTimes', dt)`. */
  std::unordered_set<Binding *> TimetableBindings;
  /* #259: same-TU (TimetableBindings, set at the timetable-producing
   * assignment) OR cross-turn (Binding::IsTimetable, stamped by the Resolver
   * from the timetable kind hook).  Lets summary/head/disp/column dispatch
   * recover the timetable-ness a later REPL turn would otherwise lose (a
   * timetable is stored generically, so it'd read back as a plain matrix). */
  bool isTimetableBinding(Binding *B) const {
    return B && (TimetableBindings.count(B) || B->IsTimetable);
  }
  /* matlab_timerange * — time-interval row subscript. Produced by
   * `tr = timerange(t1, t2, 'closed')`; consumed by `TT(tr, :)`. */
  std::unordered_set<Binding *> TimerangeBindings;
  /* Bindings tagged as holding a plain matlab_struct* (vs class
   * instance or matrix).  Populated by the AssignStmt RhsIsStruct
   * tagging block when the RHS is a known struct-returning builtin
   * (struct(...), linkBudget(...)) or a NameExpr referencing a
   * previously-tagged struct binding (including cross-REPL via
   * Binding->IsStruct).  Consumed only by the REPL-mode workspace-
   * setter routing — the same-TU struct lowering keeps using
   * StructInitialised + ensureStructSlot for the matlab_struct_new
   * init dance. */
  std::unordered_set<Binding *> StructBindings;
  /* Phase 6: bindings holding a matlab_sym * (Symbolic Math Toolbox
   * via SymPP). Triggers sym-typed arithmetic dispatch + disp routing,
   * mirrors how DatetimeBindings drive the datetime arithmetic family. */
  std::unordered_set<Binding *> SymBindings;
  /* Phase 6.1: bindings holding a matlab_symmat * (symbolic matrix).
   * Distinct from SymBindings because the runtime entries are a
   * separate set (matlab_symmat_*) and disp routes to a different
   * pretty-printer. */
  std::unordered_set<Binding *> SymmatBindings;

  /* Recursive sym-typed expression predicate. Returns true if the
   * expression's value is a matlab_sym* at runtime — covers:
   *   - NameExpr referencing a SymBindings/IsSym binding
   *   - CallOrIndex to a sym-producing builtin or sym-overloaded one
   *     (diff/int) where the first arg is sym
   *   - BinaryOp / UnaryOp where any operand is sym (transitive)
   * Used at every dispatch site that needs to know "should I route
   * through matlab_sym_*?": disp dispatch, RhsIsSym tagging, BinaryOp
   * lowering, sym-overloaded call detection. Same predicate everywhere
   * so the rules don't drift between sites. */
  bool exprIsSym(const Expr *X) const;
  /* Same predicate for matlab_symmat* — symbolic matrix. Distinct
   * dispatch path: disp routes to matlab_symmat_disp. Operator
   * arithmetic on symmat is not yet wired (would need detecting
   * symmat-typed BinaryOp). */
  bool exprIsSymmat(const Expr *X) const;
  /* Per-struct-array binding: the slot was already initialised with
   * matlab_struct_arr_new() at function entry. Avoids re-initialising
   * on every `s(i).x = ...` assignment. */
  std::unordered_set<Binding *> StructArrayInitialised;
  /* Phase 2 helper: ensure the binding has a ptr slot pre-initialised
   * to a fresh matlab_struct_arr_new(). Returns the slot ptr. */
  mlir::Value ensureStructArraySlot(Binding *Bnd, std::string_view Name,
                                     mlir::Location L);

  /* Phase 3 — value-class detection. A class is a value class iff
   * none of its ancestors is `handle`. Walks Super for inheritance.
   * MATLAB's classdef header `< handle` triggers reference semantics;
   * any other base (or no base at all) means value semantics. */
  static bool isValueClass(const ClassDef *C) {
    if (!C) return false;
    /* Resolver dropped the SuperName when it was "handle"; the
     * remaining empty SuperName means a value class. If the parser
     * preserved "handle" anywhere in the chain, that's a handle
     * class. We walk both Super (resolved) and SuperName (textual)
     * so the check is robust to the resolver's normalisation. */
    for (const ClassDef *CC = C; CC; CC = CC->Super)
      if (CC->SuperName == "handle") return false;
    return true;
  }
  /* Phase 3 helper: wrap a matlab_obj* value with matlab_obj_clone if
   * the originating expression is a value-class binding (so the
   * receiving slot gets its own copy). Returns Rhs unchanged when
   * cloning isn't required (handle class, non-class type, RHS is a
   * fresh constructor return, etc.). */
  mlir::Value maybeCloneObjForAssign(mlir::Value Rhs, const Expr *RhsExpr,
                                      mlir::Location L);
  /* Bindings assigned from a CellLiteral — tracked so calls like
   * numel(C) / length(C) / iscell(C) can dispatch to the matlab_cell_*
   * runtime entries instead of the matrix path. */
  std::unordered_set<Binding *> CellBindings;
  /* For a cell-literal binding, the 1-based linear element indices whose
   * stored value is a matrix or string (kind 1/3 — anything that the
   * runtime keeps as a ptr, not an f64 scalar). A constant-index brace
   * read `c{k}` of such an element must dispatch to matlab_cell_get_mat;
   * Sema can't carry per-element types, so without this `disp(c{k})`
   * defaults to matlab_cell_get_f64, which returns 0 for a >1-element
   * matrix (the ptr is reinterpreted as a scalar). Mirrors the
   * MatStructFields trick for struct fields. */
  std::map<Binding *, std::set<int64_t>> CellMatElems;
  /* String-typed cell elements (subset of CellMatElems): the 1-based indices
   * whose value is a string. Lets `c{i}` / disp(c{i}) recover string-ness
   * (the brace-read returns a matlab_string* via get_mat, but without this the
   * disp dispatch would treat it as a numeric matrix and print char codes). */
  std::map<Binding *, std::set<int64_t>> CellStrElems;
  /* #233: whole-cell "every element is a string" bindings (e.g. from
   * `parts = strsplit(...)`). Unlike CellStrElems (per constant index), this
   * routes ANY `parts{i}` brace read — including a runtime/variable index — to
   * matlab_cell_get_str so the element comes back as a real matlab_string*. */
  std::set<Binding *> CellAllStrBindings;
  /* Total element count of a cell-literal binding — lets a bare-`end` brace
   * read `c{end}` (effective index = count) consult CellMatElems for the
   * last element's matrix-ness. */
  std::map<Binding *, int64_t> CellElemCount;
  /* (struct binding, field name) pairs assigned a matrix value, so a later
   * read `s.field` fetches via matlab_struct_get_mat (ptr) instead of
   * defaulting to get_f64 — Sema can't specialise through struct fields, so
   * without this `sum(s.v)` / `numel(s.v)` see a scalar and bail. */
  std::set<std::pair<Binding *, std::string>> MatStructFields;
  /* (struct binding, field name) pairs assigned a char/string value
   * (#79.2: `s.name = 'hello'`).  The field holds a matlab_string*
   * (kind=3); reads still go through matlab_struct_get_mat (so the
   * pair is also in MatStructFields), but this set lets isStringExpr /
   * the fprintf str_mask tag the read as a string so `%s` routes the
   * descriptor through the string path instead of mis-reading it as a
   * numeric matrix (SIGSEGV). */
  std::set<std::pair<Binding *, std::string>> StringStructFields;
  /* Bindings whose current value is a matlab_string (from a "..."
   * literal or a matlab_string_concat result). Tracked so `a + b`
   * on two string operands routes to matlab_string_concat rather
   * than numeric addition, disp(s) routes to matlab_string_disp,
   * and strlen(s) routes to matlab_string_len. */
  std::unordered_set<Binding *> StringBindings;
  /* Bindings whose current value is a 3-D matlab_mat3 descriptor.
   * Populated when the RHS is a 3-arg zeros / ones, so A(i,j,k),
   * A(i,j,k) = v, and size(A, 3) all route to matlab_mat3 runtime
   * entries instead of the 2-D path. */
  std::unordered_set<Binding *> ThreeDBindings;
  /* True if a binding holds a 3-D value — same-turn (in ThreeDBindings, set
   * by the RhsIsThreeD assignment path) OR cross-turn (Binding::IsThreeD,
   * stamped by the Resolver from the kind=16 workspace lookup, #116).  Lets
   * the N-D subscript store/read dispatch sites recover the 3-D-ness a later
   * REPL turn would otherwise lose when the mat3 round-trips under the
   * generic mat kind. */
  bool isThreeDBinding(Binding *B) const {
    return B && (ThreeDBindings.count(B) || B->IsThreeD);
  }
  /* (struct binding, field name) pairs holding a 3-D matlab_mat3 value
   * (#78: `s.T = zeros(3,3,2)`).  Lets `s.T(i,j,k)=v` / `s.T(:,:,k)=…`
   * stores and `s.T(i,j,k)` reads route through the matlab_subscript3_*
   * helpers (load the field's mat3 ptr, mutate/read in place) the same
   * way ThreeDBindings does for plain variables. */
  std::set<std::pair<Binding *, std::string>> ThreeDStructFields;
  /* Memoized interprocedural "does this user function return a 3-D value,
   * given which of its params are 3-D?" so `A = f(...)` marks A 3-D when
   * f's output is 3-D (now that the func-boundary tensor->ptr fix makes the
   * matrix result usable).  Keyed by (function, arg-3-D bitmask) so a
   * param-dependent helper like `proc(x)=imadd(x,x)` is 3-D exactly when
   * called with a 3-D argument.  Value: 1 = in-progress (recursion cycle
   * guard), 2 = no, 3 = yes. */
  std::unordered_map<const Function *,
                     std::unordered_map<unsigned long long, int>> Func3DMemo;
  /* True when E evaluates to a 3-D matlab_mat3, given `Set` as the set of
   * locally-known 3-D bindings (the current fn's ThreeDBindings during
   * lowering, or a fresh local set during funcReturns3D analysis). */
  bool exprIsThreeD(const Expr *E, const std::unordered_set<Binding *> &Set);
  bool funcReturns3D(const Function *F, unsigned long long Arg3DMask);
  std::string CurFnName;
  /* Declared arity of the currently-lowered function — used to fold
   * references to the `nargin` / `nargout` builtins into compile-time
   * constants. Per-call-site arity would need LHS-threaded
   * monomorphisation; this v1 uses the declared counts. */
  unsigned CurFnNargin = 0;
  unsigned CurFnNargout = 0;
  mlir::Value getOrCreateSlot(Binding *Bnd, const Type *T, llvm::StringRef N,
                              mlir::Location L);
  /* True for an Expr whose runtime value is a matlab_string*. Sees:
   * "..." literals, NameExprs in StringBindings, calls to string-
   * returning builtins (num2str / sprintf / upper / lower / strtrim /
   * strrep / strcat / fgetl / bin / hex / dec), and any `+` whose
   * either operand satisfies the predicate (the other side is
   * coerced via num2str at lowering time). */
  bool isStringExpr(const Expr *E) const;
  /* True for the names of string-returning builtins listed above. */
  static bool isStringReturningBuiltin(llvm::StringRef N);
  /* Phase 1.1.C: pick the typed-int matrix runtime suffix for an
   * expression. Returns "i32" when E is a non-scalar Int32 array, "u8"
   * for non-scalar UInt8 array, empty otherwise. Used by the disp /
   * matrix-builtin emission sites to swap callee names so downstream
   * lowering reaches the typed runtime entry points (matlab_mat_i32_disp
   * etc.) without needing to thread attributes through opaque ptr SSA. */
  static llvm::StringRef intDtypeSuffixOf(const Expr *E);
  static llvm::StringRef intDtypeSuffixOfType(const Type *T);
};

//===----------------------------------------------------------------------===//
// Helpers impl
//===----------------------------------------------------------------------===//

mlir::Location Lowerer::loc(SourceLocation L) const {
  if (!SM || !L.isValid())
    return mlir::UnknownLoc::get(&MCtx);
  auto LC = SM->getLineColumn(L);
  mlir::StringAttr File = mlir::StringAttr::get(&MCtx, SM->getName(L.File));
  return mlir::FileLineColLoc::get(File, LC.Line, LC.Column);
}

mlir::Type Lowerer::mirTy(const Type *T) const {
  return mapType(MCtx, T);
}

mlir::Operation *Lowerer::emitUnregOp(llvm::StringRef OpName,
                                     llvm::ArrayRef<mlir::Value> Operands,
                                     llvm::ArrayRef<mlir::Type> ResultTypes,
                                     mlir::Location Loc,
                                     llvm::ArrayRef<mlir::NamedAttribute> Attrs,
                                     unsigned NumRegions) {
  mlir::OperationState State(Loc, OpName);
  State.addOperands(Operands);
  State.addTypes(ResultTypes);
  for (auto &A : Attrs) State.addAttribute(A.getName(), A.getValue());
  for (unsigned i = 0; i < NumRegions; ++i) State.addRegion();
  return B.create(State);
}

mlir::Value Lowerer::emitUnreg(llvm::StringRef OpName,
                               llvm::ArrayRef<mlir::Value> Operands,
                               mlir::Type ResultType, mlir::Location Loc,
                               llvm::ArrayRef<mlir::NamedAttribute> Attrs) {
  mlir::Operation *Op = emitUnregOp(OpName, Operands, {ResultType}, Loc, Attrs);
  return Op->getResult(0);
}

bool Lowerer::isStringReturningBuiltin(llvm::StringRef N) {
  return N == "fgetl" || N == "sprintf" || N == "num2str" ||
         N == "upper" || N == "lower" || N == "strtrim" ||
         N == "strrep" || N == "strcat" || N == "char" ||
         N == "deblank" || N == "blanks" || N == "strjoin" ||
         N == "regexprep" ||  /* #235 — regexp is numeric, regexprep a string */
         N == "bin" || N == "hex" || N == "dec" ||
         /* Bioinformatics sequence transforms return a char sequence
          * (matlab_string*); tag so length / disp / strcmp on the result
          * route through the string lane rather than reading it as a matrix. */
         N == "int2nt" || N == "int2aa" || N == "nt2aa" ||
         N == "dna2rna" || N == "rna2dna" || N == "seqcomplement" ||
         N == "seqrcomplement" || N == "seqreverse" || N == "randseq" ||
         /* Phase B — MSA / consensus / Newick are char-string returns. */
         N == "multialign" || N == "profalign" || N == "seqconsensus" ||
         N == "getnewickstr" || N == "matlab_bioinfo_phytree_newick" ||
         /* Phase C — aminolookup / cleave / restrict return char strings. */
         N == "aminolookup" || N == "cleave" || N == "restrict" ||
         /* pwd returns the current directory as a char string. */
         N == "pwd";
}

llvm::StringRef Lowerer::intDtypeSuffixOfType(const Type *T) {
  if (!T || T->K != Type::Kind::Array) return {};
  auto &A = static_cast<const ArrayType &>(*T);
  /* Scalar typed ints are represented at MLIR level as native i32 / i8
   * values, which the existing scalar-disp path handles via SIToFP /
   * UIToFP -> matlab_disp_f64. Only matrix-shaped values need the typed
   * runtime descriptor entry points. */
  if (A.S.K == Shape::Rank::Scalar) return {};
  if (A.Elt == Dtype::Int32) return "i32";
  if (A.Elt == Dtype::UInt8) return "u8";
  return {};
}

llvm::StringRef Lowerer::intDtypeSuffixOf(const Expr *E) {
  return intDtypeSuffixOfType(E ? E->Ty : nullptr);
}

bool Lowerer::isStringExpr(const Expr *E) const {
  if (!E) return false;
  if (E->Kind == NodeKind::StringLiteral) return true;
  if (auto *N = dynamic_cast<const NameExpr *>(E))
    return (N->Ref && StringBindings.count(N->Ref) > 0) ||
           /* bare `pwd` reads as a char string (it has no parens, so it
            * never reaches the CallOrIndex branch below). */
           (N->Ref && N->Ref->Kind == BindingKind::Builtin && N->Name == "pwd");
  /* A char/string-valued struct field (#79.2: `s.name='hello'`). */
  if (auto *F = dynamic_cast<const FieldAccess *>(E))
    if (auto *BN = dynamic_cast<const NameExpr *>(F->Base))
      if (BN->Ref &&
          StringStructFields.count({BN->Ref, std::string(F->Field)}))
        return true;
  if (auto *C = dynamic_cast<const CallOrIndex *>(E)) {
    if (auto *NX = dynamic_cast<const NameExpr *>(C->Callee))
      if (NX->Ref && NX->Ref->Kind == BindingKind::Builtin &&
          isStringReturningBuiltin(NX->Name))
        return true;
  }
  /* A string-typed cell element read `c{k}` (#206): the constant index k is
   * recorded in CellStrElems by the cell-literal assignment. */
  if (auto *CI = dynamic_cast<const CellIndex *>(E)) {
    if (auto *NX = dynamic_cast<const NameExpr *>(CI->Callee))
      if (NX->Ref && CI->Args.size() == 1) {
        /* #233: a brace read of a whole cell-of-strings binding is a string
         * for any index (e.g. `parts{i}` after `parts = strsplit(...)`). */
        if (CellAllStrBindings.count(NX->Ref)) return true;
        auto It = CellStrElems.find(NX->Ref);
        if (It != CellStrElems.end())
          if (auto *IL = dynamic_cast<const IntegerLiteral *>(CI->Args[0])) {
            int64_t k = 0;
            try { k = std::stoll(std::string(IL->Text)); } catch (...) { k = 0; }
            if (k > 0 && It->second.count(k)) return true;
          }
      }
  }
  if (auto *Bi = dynamic_cast<const BinaryOpExpr *>(E))
    if (Bi->Op == BinOp::Add)
      return isStringExpr(Bi->LHS) || isStringExpr(Bi->RHS);
  /* Single-row bracket literals containing any char/string element are
   * char-array concatenations (MATLAB's classic `['x = ', num2str(v)]`
   * idiom) — the MatrixLiteral lowering reroutes them to a chain of
   * matlab_string_concat calls and the result is a matlab_string*. */
  if (auto *M = dynamic_cast<const MatrixLiteral *>(E)) {
    if (M->Rows.size() == 1)
      for (const Expr *Cx : M->Rows[0])
        if (Cx && (Cx->Kind == NodeKind::CharLiteral ||
                   Cx->Kind == NodeKind::StringLiteral ||
                   isStringExpr(Cx)))
          return true;
  }
  return false;
}

/* True when E evaluates to a 3-D matlab_mat3.  `Set` is the set of bindings
 * known to hold 3-D values in the relevant scope (the live ThreeDBindings
 * during lowering, or a fresh local set while analysing a function body in
 * funcReturns3D).  3-D-ness flows through arithmetic / unary ops / aliasing,
 * the 3-D-producing builtins, and user-function returns. */
bool Lowerer::exprIsThreeD(const Expr *E,
                           const std::unordered_set<Binding *> &Set) {
  if (!E) return false;
  if (auto *C = dynamic_cast<const CallOrIndex *>(E)) {
    auto *N = dynamic_cast<const NameExpr *>(C->Callee);
    if (!N) return false;
    if (C->Args.size() == 3 && N->Ref &&
        N->Ref->Kind == BindingKind::Builtin &&
        (N->Name == "zeros" || N->Name == "ones")) {
      /* trailing-singleton: zeros(m,n,1) is 2-D, not 3-D */
      if (auto *PL = dynamic_cast<const IntegerLiteral *>(C->Args[2]))
        if (PL->Text == "1") return false;
      return true;
    }
    if (N->Name == "cat" && C->Args.size() >= 2) {
      if (auto *DimL = dynamic_cast<const IntegerLiteral *>(C->Args[0])) {
        /* cat(3, p1, p2, …) of >=2 planes is 3-D; any cat dim whose
         * operands are already 3-D stays 3-D — EXCEPT dim 4, which
         * promotes to rank-4 (matN) and isn't 3-D anymore. */
        if (DimL->Text == "3" && C->Args.size() >= 3) return true;
        if (DimL->Text == "4") return false;
        for (size_t a = 1; a < C->Args.size(); ++a)
          if (exprIsThreeD(C->Args[a], Set)) return true;
      }
      return false;
    }
    /* Tier-B shape verbs that preserve / produce 3-D.  reshape/repmat with a
     * 3rd target dim produce a mat3 (runtime returns 2-D when that dim is 1 —
     * *3 helpers 2-D-fall-back).  permute/ipermute/squeeze keep 3-D-ness from
     * their input (squeeze may collapse to 2-D; the fall-back handles it). */
    if ((N->Name == "reshape" || N->Name == "repmat") && C->Args.size() == 4)
      return true;
    if ((N->Name == "permute" || N->Name == "ipermute" || N->Name == "squeeze") &&
        !C->Args.empty())
      return exprIsThreeD(C->Args[0], Set);
    /* Depth-preserving image arithmetic: 3-D iff the image arg is 3-D
     * (these loop over depth and return a mat3 — runtime_images.cpp). */
    if ((N->Name == "imadd" || N->Name == "imsubtract" ||
         N->Name == "immultiply" || N->Name == "imdivide" ||
         N->Name == "imabsdiff" || N->Name == "imcomplement") &&
        !C->Args.empty())
      return exprIsThreeD(C->Args[0], Set);
    if (N->Name == "imlincomb" && C->Args.size() >= 2)
      return exprIsThreeD(C->Args[1], Set);
    /* imread may return a 2-D grayscale matrix; the matlab_mat3 runtime
     * helpers fall back to a 2-D view so grayscale files still index/size. */
    if (N->Name == "rgb2hsv" || N->Name == "hsv2rgb" ||
        N->Name == "rgb2ycbcr" || N->Name == "ycbcr2rgb" ||
        N->Name == "rgb2lab" || N->Name == "lab2rgb" ||
        N->Name == "label2rgb" || N->Name == "imread")
      return true;
    /* User-defined function call: 3-D iff the function returns a 3-D value
     * (interprocedural; memoised in funcReturns3D).  Pass which arguments
     * are 3-D in the caller's context so a param-dependent helper (e.g.
     * `proc(x)=imadd(x,x)`) is tracked 3-D when called with a 3-D arg. The
     * func-boundary tensor->ptr fix makes the matrix result usable; this
     * routes its 3-D subscripts / size / numel / ndims through the *3 helpers. */
    if (N->Ref && N->Ref->Kind == BindingKind::Function && N->Ref->FuncDef) {
      unsigned long long Mask = 0;
      for (size_t i = 0; i < C->Args.size() && i < 64; ++i)
        if (C->Args[i] && exprIsThreeD(C->Args[i], Set)) Mask |= (1ull << i);
      return funcReturns3D(N->Ref->FuncDef, Mask);
    }
    return false;
  }
  if (auto *B = dynamic_cast<const BinaryOpExpr *>(E))
    return exprIsThreeD(B->LHS, Set) || exprIsThreeD(B->RHS, Set);
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(E))
    return exprIsThreeD(U->Operand, Set);
  if (auto *N = dynamic_cast<const NameExpr *>(E))
    return N->Ref && (Set.count(N->Ref) || N->Ref->IsThreeD);
  return false;
}

/* Interprocedural: does user function F return a 3-D value?  Analyse F's
 * body once (memoised, with an in-progress cycle guard for recursion),
 * propagating 3-D-ness through its assignments into a local set; F returns
 * 3-D iff any of its outputs lands in that set.  Control-flow bodies are
 * flattened (an over-approximation — safe, since the *3 helpers 2-D-fall-back
 * when a binding flagged 3-D actually holds a 2-D value at runtime). */
bool Lowerer::funcReturns3D(const Function *F, unsigned long long Arg3DMask) {
  if (!F || F->OutputRefs.empty()) return false;
  if (auto FIt = Func3DMemo.find(F); FIt != Func3DMemo.end()) {
    if (auto MIt = FIt->second.find(Arg3DMask); MIt != FIt->second.end()) {
      if (MIt->second == 1) return false;  /* in progress: break the cycle */
      return MIt->second == 3;
    }
  }
  Func3DMemo[F][Arg3DMask] = 1;  /* in progress (don't hold a reference —
                                  * the recursive walk may rehash the map) */
  std::unordered_set<Binding *> Local;
  /* Seed the params the call site passed a 3-D argument for, so 3-D-ness
   * flows from the arguments through the body to the outputs. */
  for (size_t i = 0; i < F->ParamRefs.size() && i < 64; ++i)
    if (((Arg3DMask >> i) & 1ull) && F->ParamRefs[i])
      Local.insert(F->ParamRefs[i]);
  std::function<void(const Block *)> walk = [&](const Block *Blk) {
    if (!Blk) return;
    for (const Stmt *S : Blk->Stmts) {
      if (!S) continue;
      if (auto *AS = dynamic_cast<const AssignStmt *>(S)) {
        if (AS->RHS && exprIsThreeD(AS->RHS, Local))
          for (const Expr *L : AS->LHS)
            if (auto *LN = dynamic_cast<const NameExpr *>(L))
              if (LN->Ref) Local.insert(LN->Ref);
      } else if (auto *Ifs = dynamic_cast<const IfStmt *>(S)) {
        walk(Ifs->Then);
        for (const auto &EI : Ifs->Elseifs) walk(EI.Body);
        walk(Ifs->Else);
      } else if (auto *Fs = dynamic_cast<const ForStmt *>(S)) {
        walk(Fs->Body);
      } else if (auto *Ws = dynamic_cast<const WhileStmt *>(S)) {
        walk(Ws->Body);
      } else if (auto *Sw = dynamic_cast<const SwitchStmt *>(S)) {
        for (const auto &Cse : Sw->Cases) walk(Cse.Body);
      } else if (auto *Ts = dynamic_cast<const TryStmt *>(S)) {
        walk(Ts->TryBody);
        walk(Ts->CatchBody);
      } else if (auto *Nested = dynamic_cast<const Block *>(S)) {
        walk(Nested);
      }
    }
  };
  walk(F->Body);
  bool R = false;
  for (Binding *O : F->OutputRefs)
    if (O && Local.count(O)) { R = true; break; }
  Func3DMemo[F][Arg3DMask] = R ? 3 : 2;
  return R;
}

mlir::Value Lowerer::emitAlloc(const Type *T, llvm::StringRef Name,
                               mlir::Location Loc) {
  mlir::Type MT = mirTy(T);
  mlir::NamedAttribute NameAttr(
      mlir::StringAttr::get(&MCtx, "name"),
      mlir::StringAttr::get(&MCtx, Name));
  return emitUnreg("matlab.alloc", {}, MT, Loc, {NameAttr});
}

mlir::Value Lowerer::emitLoad(mlir::Value Slot, mlir::Type Ty,
                              mlir::Location Loc) {
  return emitUnreg("matlab.load", {Slot}, Ty, Loc);
}

void Lowerer::emitStore(mlir::Value V, mlir::Value Slot, mlir::Location Loc) {
  emitUnregOp("matlab.store", {V, Slot}, {}, Loc);
  /* Mirror the store to the runtime's per-frame Locals table so the
   * DAP server can render `Locals` for any frame in the call stack —
   * not just the script-level workspace. Only fires when DebugMode is
   * on (no overhead in production builds) and only for slots that
   * carry a binding name on their `matlab.alloc` op (skips synthetic
   * BreakSlot / ContinueSlot / spill slots that have no user-visible
   * identity).
   *
   * The mirror dispatches by the stored value's type:
   *   - f64  -> matlab_dbg_frame_set_f64(name, len, val)
   *   - other -> matlab_dbg_frame_set_mat(name, len, ptr)
   * Tensor / matrix types end up as !llvm.ptr after LowerTensorOps,
   * which is what the runtime expects. */
  if (!DebugMode) return;
  mlir::Operation *Def = Slot.getDefiningOp();
  if (!Def) return;
  /* The slot is the result of a matlab.alloc op; everything else
   * (block args, loads, etc.) isn't a store target with a name. */
  auto NameAttr = Def->getAttrOfType<mlir::StringAttr>("name");
  if (!NameAttr) return;
  llvm::StringRef Name = NameAttr.getValue();
  if (Name.empty()) return;
  mlir::Value NameV = emitFieldNameChar(Name, Loc);
  /* Emit a generic matlab_dbg_frame_set builtin and let LowerTensorOps
   * dispatch on the (by then concrete) operand type — at this point in
   * the pipeline V is still `none`-typed, so picking the f64 vs mat
   * variant here would always be wrong.
   *
   * If the slot is tagged with a `matlab.class_id` (set when the
   * binding is pinned to a user classdef), forward the attribute on
   * the call so LowerTensorOps can pick matlab_dbg_frame_set_obj. The
   * obj path keeps the class identity visible in the LOCALS panel
   * instead of falling through to the matrix formatter. */
  llvm::SmallVector<mlir::NamedAttribute, 2> Attrs;
  Attrs.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&MCtx, "callee"),
      mlir::StringAttr::get(&MCtx, "matlab_dbg_frame_set")));
  if (auto ClsId = Def->getAttrOfType<mlir::IntegerAttr>("matlab.class_id"))
    Attrs.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&MCtx, "matlab.class_id"), ClsId));
  emitUnregOp("matlab.call_builtin", {NameV, V},
              {mlir::NoneType::get(&MCtx)}, Loc, Attrs);
}

//===----------------------------------------------------------------------===//
// Op-name mapping
//===----------------------------------------------------------------------===//

llvm::StringRef Lowerer::binOpName(BinOp O) {
  switch (O) {
  case BinOp::Add:          return "matlab.add";
  case BinOp::Sub:          return "matlab.sub";
  case BinOp::Mul:          return "matlab.matmul";
  case BinOp::Div:          return "matlab.matdiv";
  case BinOp::LeftDiv:      return "matlab.matldiv";
  case BinOp::Pow:          return "matlab.matpow";
  case BinOp::ElemMul:      return "matlab.emul";
  case BinOp::ElemDiv:      return "matlab.ediv";
  case BinOp::ElemLeftDiv:  return "matlab.eldiv";
  case BinOp::ElemPow:      return "matlab.epow";
  case BinOp::Eq:           return "matlab.eq";
  case BinOp::Ne:           return "matlab.ne";
  case BinOp::Lt:           return "matlab.lt";
  case BinOp::Le:           return "matlab.le";
  case BinOp::Gt:           return "matlab.gt";
  case BinOp::Ge:           return "matlab.ge";
  case BinOp::And:          return "matlab.and";
  case BinOp::Or:           return "matlab.or";
  case BinOp::ShortAnd:     return "matlab.short_and";
  case BinOp::ShortOr:      return "matlab.short_or";
  }
  return "matlab.add";
}

llvm::StringRef Lowerer::unOpName(UnOp O) {
  switch (O) {
  case UnOp::Plus:  return "matlab.uplus";
  case UnOp::Minus: return "matlab.neg";
  case UnOp::Not:   return "matlab.not";
  }
  return "matlab.neg";
}

llvm::StringRef Lowerer::postfixName(PostfixOp O) {
  switch (O) {
  case PostfixOp::CTranspose: return "matlab.ctranspose";
  case PostfixOp::Transpose:  return "matlab.transpose";
  }
  return "matlab.transpose";
}

//===----------------------------------------------------------------------===//
// Slot handling
//===----------------------------------------------------------------------===//

bool Lowerer::exprIsSym(const Expr *X) const {
  if (!X) return false;
  if (auto *NE = dynamic_cast<const NameExpr *>(X))
    return NE->Ref &&
           (SymBindings.count(NE->Ref) || NE->Ref->IsSym);
  if (auto *CX = dynamic_cast<const CallOrIndex *>(X)) {
    if (auto *CN = dynamic_cast<const NameExpr *>(CX->Callee)) {
      llvm::StringRef Nm = CN->Name;
      static const llvm::StringSet<> Producers = {
          "sym", "syms", "str2sym", "simplify", "expand",
          "subs", "vpa", "taylor", "limit",
          "dsolve", "pdsolve", "pdsolve_heat", "pdsolve_wave",
          "laplace", "ilaplace", "fourier", "ifourier",
          "ztrans", "iztrans",
          "assume", "assumeAlso", "clearAssumptions",
          "nsolve", "vpasolve", "checkodesol",
          "dsolve_ivp", "apply_ivp",
          /* Phase 6.1 — symmat reductions returning sym scalars. */
          "sym_det", "sym_trace"};
      if (Producers.contains(Nm)) return true;
      /* `solve` is overloaded: SymPP's symbolic `solve` *and* the
       * Optimization Toolbox problem-based `solve(prob)`. The latter
       * returns a plain numeric solution vector (matlab_mat*), not a
       * sym — so only treat `solve` as a sym producer when its first
       * argument is NOT pinned to a problem-based classdef. Without
       * this the REPL tags `sol = solve(prob)` as sym, stores it via
       * matlab_ws_set_sym, and the next turn's `sol(1)` reads garbage. */
      if (Nm == "solve") {
        if (!CX->Args.empty()) {
          if (auto *AN = dynamic_cast<const NameExpr *>(CX->Args[0]))
            if (AN->Ref && AN->Ref->PinnedClass &&
                (AN->Ref->PinnedClass->Name == "OptimizationProblem" ||
                 AN->Ref->PinnedClass->Name == "EquationProblem" ||
                 /* PDE Toolbox (#28): solve(femodel) is a FEM solve, not a
                  * symbolic one — let it fall through to the builtin call
                  * that LowerTensorOps maps to matlab_pde_solve. */
                 AN->Ref->PinnedClass->Name == "femodel"))
              return false;
        }
        return true;
      }
      /* Type-overloaded sym builtins — sym only when first arg is sym.
       * Covers diff/int/sin/cos/exp/log/sqrt/abs and the rest of the
       * elementary functions; when the first arg is sym, the result is
       * sym. The matlab.call_builtin's emitted callee is rewritten by
       * the call dispatch (NameExpr CallOrIndex path below) into the
       * matlab_sym_* variant — but the type predicate has to know
       * about it ahead of dispatch. */
      static const llvm::StringSet<> Overloaded = {
          "diff", "int",
          "sin", "cos", "tan", "asin", "acos", "atan",
          "sinh", "cosh", "tanh",
          "exp", "log", "sqrt", "abs",
          /* #235 — factor is symbolic only for a sym arg (factor(expr,var));
           * factor(n) on a number is the numeric prime factorisation. */
          "factor"};
      if (Overloaded.contains(Nm) && !CX->Args.empty())
        return exprIsSym(CX->Args[0]);
    }
    return false;
  }
  if (auto *B2 = dynamic_cast<const BinaryOpExpr *>(X))
    return exprIsSym(B2->LHS) || exprIsSym(B2->RHS);
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(X))
    return exprIsSym(U->Operand);
  return false;
}

bool Lowerer::exprIsSymmat(const Expr *X) const {
  if (!X) return false;
  /* NameExpr referencing a binding tagged via SymmatBindings — set by
   * the AssignStmt path when the RHS is a symmat-producer. Plus the
   * cross-TU REPL flag IsSymmat (kind=8 stamp from the Resolver hook). */
  if (auto *NE = dynamic_cast<const NameExpr *>(X))
    return NE->Ref &&
           (SymmatBindings.count(NE->Ref) || NE->Ref->IsSymmat);
  if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
    if (auto *CN = dynamic_cast<const NameExpr *>(CX->Callee)) {
      static const llvm::StringSet<> MatProducers = {
          "sym_matrix", "sym_eye", "sym_zeros",
          "sym_inv", "sym_transpose", "sym_linsolve",
          "sym_dsolve_system",
          "sym_solve_2x2", "sym_solve_3x3", "sym_solve_sys"};
      return MatProducers.contains(CN->Name);
    }
  /* Phase 6.2 — `[a 1; 2 b]` matrix literal where any entry is sym is
   * lowered via matlab_symmat_zeros + matlab_symmat_set. The result
   * type is matlab_symmat*, so Sema's MatrixLiteral lowering must
   * advertise symmat for the AssignStmt LHS-tagging + disp dispatch
   * to route through the right runtime. */
  if (auto *ML = dynamic_cast<const MatrixLiteral *>(X)) {
    for (auto &Row : ML->Rows)
      for (const Expr *Cx : Row)
        if (Cx && exprIsSym(Cx)) return true;
  }
  return false;
}

mlir::Value Lowerer::getOrCreateSlot(Binding *Bnd, const Type *T,
                                     llvm::StringRef N, mlir::Location L) {
  auto It = Slots.find(Bnd);
  if (It != Slots.end()) return It->second;

  // Allocate at the start of the current func's entry block.
  auto *InsBlock = B.getInsertionBlock();
  mlir::Block *Entry = InsBlock;
  mlir::Operation *P = InsBlock ? InsBlock->getParentOp() : nullptr;
  while (P && !mlir::isa<mlir::func::FuncOp>(P)) {
    auto *PB = P->getBlock();
    P = PB ? PB->getParentOp() : nullptr;
  }
  if (P) {
    auto F = mlir::cast<mlir::func::FuncOp>(P);
    Entry = &F.getBody().front();
  }

  mlir::OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(Entry);
  mlir::Value Slot = emitAlloc(T, N, L);
  /* Carry the class_id forward so the DAP store-mirror knows the slot
   * holds a class instance — same reason as the explicit-alloc sites
   * in lowerFunction, just for slots created lazily on first
   * assignment. */
  if (Bnd && Bnd->PinnedClass && Bnd->PinnedClass->ClassId > 0) {
    auto I32 = mlir::IntegerType::get(&MCtx, 32);
    Slot.getDefiningOp()->setAttr(
        "matlab.class_id",
        mlir::IntegerAttr::get(I32, (int64_t)Bnd->PinnedClass->ClassId));
  }
  Slots[Bnd] = Slot;
  return Slot;
}

mlir::Value Lowerer::ensureStructSlot(Binding *Bnd, std::string_view Name,
                                       mlir::Location L) {
  /* Allocate a ptr slot for the struct and initialise it with a fresh
   * matlab_struct_new() in the function's entry block. Idempotent per
   * binding. The returned value is the slot (ptr-typed matlab.alloc
   * result) — callers matlab.load/store through it. */
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
  auto It = Slots.find(Bnd);
  mlir::Value Slot;
  if (It != Slots.end()) {
    Slot = It->second;
  } else {
    /* emitAlloc wants a Sema Type*; we go around it with a raw
     * matlab.alloc of ptr result so retypeMatrixSlots leaves it alone. */
    mlir::OpBuilder::InsertionGuard G(B);
    auto *InsBlock = B.getInsertionBlock();
    mlir::Operation *P = InsBlock ? InsBlock->getParentOp() : nullptr;
    while (P && !mlir::isa<mlir::func::FuncOp>(P)) {
      auto *PB = P->getBlock();
      P = PB ? PB->getParentOp() : nullptr;
    }
    if (P) B.setInsertionPointToStart(
        &mlir::cast<mlir::func::FuncOp>(P).getBody().front());
    mlir::NamedAttribute NA(
        mlir::StringAttr::get(&MCtx, "name"),
        mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Name)));
    Slot = emitUnreg("matlab.alloc", {}, PtrTy, L, {NA});
    Slots[Bnd] = Slot;
  }
  if (!StructInitialised.count(Bnd)) {
    StructInitialised.insert(Bnd);
    mlir::OpBuilder::InsertionGuard G(B);
    /* Insert the init right after the alloc so the slot has a value
     * before any read/write. Placing in the function entry block
     * works because Slot was allocated there too. */
    auto *SlotOp = Slot.getDefiningOp();
    if (SlotOp) B.setInsertionPointAfter(SlotOp);
    /* REPL mode + binding known to hold a struct: pull the existing
     * pointer from the workspace instead of allocating a fresh empty
     * struct.  Two paths land here:
     *   - cross-input: Resolver stamps Bnd->IsStruct when the
     *     workspace-kind hook reports kind=12 for a prior input's
     *     binding.
     *   - same-input: StructBindings.count(Bnd) is set by the
     *     AssignStmt RhsIsStruct path the moment we lower the
     *     `lb = linkBudget(...)` LHS.  Without this, the same-TU
     *     pair `lb = linkBudget(...); disp(lb.Distance)` would see
     *     ensureStructSlot allocate a fresh matlab_struct_new() and
     *     shadow the just-stored workspace value (which is the only
     *     place the assign wrote to in REPL mode — there's no local
     *     slot store for `lb`). */
    bool LoadFromWorkspace =
        ReplMode && InScriptBody &&
        (Bnd->IsStruct || StructBindings.count(Bnd)) &&
        Bnd->Kind == BindingKind::Var;
    if (LoadFromWorkspace) {
      mlir::Value NameV = emitFieldNameChar(Name, L);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_mat"));
      mlir::Value Ptr = emitUnreg("matlab.call_builtin", {NameV},
                                   PtrTy, L, {Cal});
      emitStore(Ptr, Slot, L);
    } else {
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_struct_new"));
      mlir::Value NewPtr = emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
      emitStore(NewPtr, Slot, L);
    }
  }
  return Slot;
}

mlir::Value Lowerer::ensureStructArraySlot(Binding *Bnd,
                                            std::string_view Name,
                                            mlir::Location L) {
  /* Phase 2: a slot for a matlab_struct_arr*, initialised once with
   * matlab_struct_arr_new() at the function's entry. Mirrors
   * ensureStructSlot but uses the struct_arr runtime constructor. */
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
  auto It = Slots.find(Bnd);
  mlir::Value Slot;
  if (It != Slots.end()) {
    Slot = It->second;
  } else {
    mlir::OpBuilder::InsertionGuard G(B);
    auto *InsBlock = B.getInsertionBlock();
    mlir::Operation *P = InsBlock ? InsBlock->getParentOp() : nullptr;
    while (P && !mlir::isa<mlir::func::FuncOp>(P)) {
      auto *PB = P->getBlock();
      P = PB ? PB->getParentOp() : nullptr;
    }
    if (P) B.setInsertionPointToStart(
        &mlir::cast<mlir::func::FuncOp>(P).getBody().front());
    mlir::NamedAttribute NA(
        mlir::StringAttr::get(&MCtx, "name"),
        mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Name)));
    Slot = emitUnreg("matlab.alloc", {}, PtrTy, L, {NA});
    Slots[Bnd] = Slot;
  }
  if (!StructArrayInitialised.count(Bnd)) {
    StructArrayInitialised.insert(Bnd);
    mlir::OpBuilder::InsertionGuard G(B);
    auto *SlotOp = Slot.getDefiningOp();
    if (SlotOp) B.setInsertionPointAfter(SlotOp);
    /* #133: in ReplMode, a struct array bound in a prior turn
     * (Bnd->IsStructArray, kind=14) — or same-input, already in
     * StructArrayBindings — must rehydrate its pointer from the
     * workspace (matlab_ws_get_mat, kind=14 pass-through) rather than
     * allocate a fresh empty array that would shadow the stored value.
     * Mirrors the ensureStructSlot rehydrate. */
    /* Only rehydrate for a CROSS-TURN array (Resolver-stamped IsStructArray
     * from a prior input's kind=14).  NOT same-turn StructArrayBindings: the
     * `a(i).x = v` store inserts that set *before* calling this, so on the
     * defining turn `a` isn't in the workspace yet — loading it would pull an
     * empty matrix and corrupt the store.  StructArrayInitialised guards
     * re-init within a turn, so a fresh matlab_struct_arr_new() on first
     * touch is correct for the defining turn. */
    bool LoadFromWorkspace =
        ReplMode && InScriptBody && Bnd->Kind == BindingKind::Var &&
        Bnd->IsStructArray;
    if (LoadFromWorkspace) {
      mlir::Value NameV = emitFieldNameChar(Name, L);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_mat"));
      mlir::Value Ptr = emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
      emitStore(Ptr, Slot, L);
    } else {
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_struct_arr_new"));
      mlir::Value NewPtr = emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
      emitStore(NewPtr, Slot, L);
    }
  }
  return Slot;
}

mlir::Value Lowerer::maybeCloneObjForAssign(mlir::Value Rhs,
                                             const Expr *RhsExpr,
                                             mlir::Location L) {
  /* Phase 3: clone the source obj on assign when it's a value class.
   * Heuristic: clone iff RhsExpr is a NameExpr (or FieldAccess on a
   * NameExpr) whose binding is class-pinned to a value class. A
   * CallOrIndex RHS is a fresh return value; calling with no
   * additional clone is always safe (the callee already produced a
   * fresh obj). The same applies to BinaryOp / UnaryOp results. */
  if (!Rhs || !RhsExpr) return Rhs;
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
  const NameExpr *NE = dynamic_cast<const NameExpr *>(RhsExpr);
  const ClassDef *Cls = (NE && NE->Ref) ? NE->Ref->PinnedClass : nullptr;
  if (Cls) {
    if (!isValueClass(Cls)) return Rhs;
    /* Class-instance values may flow through `none`-typed slots in the
     * existing lowering (the alloc carries `matlab.class_id` but the
     * MLIR result type is none). Emit the clone call regardless and
     * let LowerTensorOps retype operands through the runtime call. */
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_obj_clone"));
    return emitUnreg("matlab.call_builtin", {Rhs}, PtrTy, L, {Cal});
  }
  /* Matrix copy-on-assign: a bare `B = A` shallow-copies the `matlab_mat*`
   * pointer (the runtime has no refcount/COW), so a later `B(i) = v` would
   * mutate A's shared buffer.  Clone when the RHS is a plain numeric-matrix
   * variable (double / single / complex element) flowing as a heap pointer.
   * Strings/structs/cells/objects (different Type::Kind or class-pinned) and
   * integer-typed arrays (different runtime descriptor layout) are excluded,
   * and a fresh RHS (call / operator / literal result) is not a NameExpr so
   * never reaches here — only a bare variable reference can alias. */
  if (NE && NE->Ref && !NE->Ref->PinnedClass) {
    const Type *RT = RhsExpr->Ty ? RhsExpr->Ty : NE->Ref->InferredType;
    bool isMat = false;
    if (RT && RT->K == Type::Kind::Array) {
      auto *AT = static_cast<const ArrayType *>(RT);
      bool numeric = AT->Elt == Dtype::Double || AT->Elt == Dtype::Single ||
                     AT->Elt == Dtype::Complex;
      /* Only clone a *definitely multi-element* matrix.  Scalars flow as an
       * f64 (no aliasing) and — crucially — a scalar wrongly wrapped here
       * disrupts the lowering of values flowing into struct-field sets, N-D
       * stores, and function returns (it leaves a `matlab.call_builtin` that
       * downstream type-matched patterns no longer recognise).  Unknown-rank
       * is excluded for the same safety reason. */
      bool multiElem = AT->S.K == Shape::Rank::Vector ||
                       AT->S.K == Shape::Rank::Matrix ||
                       AT->S.K == Shape::Rank::NDArray;
      isMat = numeric && multiElem;
    }
    if (isMat) {
      /* A bare `B = A` shallow-copies the `matlab_mat*` pointer (the runtime
       * has no refcount/COW), so a later `B(i) = v` would mutate A's shared
       * buffer.  Clone the buffer (the LowerTensorOps arm passes a scalar f64
       * through unchanged; the runtime helper deep-copies matlab_mat / mat3 /
       * mat_c via the magic tag).  Gated on a *positive* numeric-matrix static
       * type so non-matrix pointer values are never misread. */
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_mat_clone_cow"));
      return emitUnreg("matlab.call_builtin", {Rhs}, PtrTy, L, {Cal});
    }
  }
  return Rhs;
}

mlir::Value Lowerer::resolveStructBase(const Expr *E, mlir::Location L) {
  if (!E) return {};
  if (auto *N = dynamic_cast<const NameExpr *>(E)) {
    if (!N->Ref) return {};
    /* REPL mode + struct binding: in REPL the assign side bypasses
     * the local slot and routes through matlab_ws_set_struct.  An
     * ensureStructSlot path here would emit a stale matlab_struct_new
     * (or a workspace-load placed at function-entry, before the
     * assign hit the workspace) and shadow the real value.  Read
     * directly from the workspace at the read site so we always see
     * the latest store.
     *
     * The same applies to a class-pinned binding (PinnedClass set):
     * a classdef instance restored across REPL turns lives in the
     * workspace as a kind=2 matlab_obj*, and matlab_obj shares the
     * leading struct layout — so `prob.Constraints.c1 = ...` must walk
     * the child struct of the *workspace* object, not a throwaway
     * matlab_struct_new local slot.  Without this branch the nested
     * field write lands on a local struct that's discarded at end of
     * turn and `solve(prob)` sees an objective-only / empty problem. */
    if (ReplMode && InScriptBody && N->Ref->Kind == BindingKind::Var &&
        (N->Ref->IsStruct || StructBindings.count(N->Ref) ||
         N->Ref->PinnedClass != nullptr)) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::Value NameV = emitFieldNameChar(N->Name, L);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_mat"));
      return emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
    }
    mlir::Value Slot = ensureStructSlot(N->Ref, N->Name, L);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    return emitLoad(Slot, PtrTy, L);
  }
  if (auto *F = dynamic_cast<const FieldAccess *>(E)) {
    mlir::Value Parent = resolveStructBase(F->Base, L);
    if (!Parent) return {};
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Value NameV = emitFieldNameChar(F->Field, L);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_struct_get_child_struct"));
    return emitUnreg("matlab.call_builtin", {Parent, NameV},
                     PtrTy, L, {Cal});
  }
  return {};
}

mlir::Value Lowerer::emitFieldNameChar(std::string_view Name,
                                        mlir::Location L) {
  mlir::NamedAttribute VA(
      mlir::StringAttr::get(&MCtx, "value"),
      mlir::StringAttr::get(&MCtx, std::string(Name)));
  return emitUnreg("matlab.const_char", {},
                   mlir::NoneType::get(&MCtx), L, {VA});
}

void Lowerer::emitDbgEnterFrame(llvm::StringRef Name, mlir::Location L) {
  if (!DebugMode) return;
  mlir::Value NameV = emitFieldNameChar(Name, L);
  mlir::NamedAttribute Cal(
      mlir::StringAttr::get(&MCtx, "callee"),
      mlir::StringAttr::get(&MCtx, "matlab_dbg_enter_frame"));
  emitUnregOp("matlab.call_builtin", {NameV},
              {mlir::NoneType::get(&MCtx)}, L, {Cal});
}

void Lowerer::emitDbgLeaveFrame(mlir::Location L) {
  if (!DebugMode) return;
  mlir::NamedAttribute Cal(
      mlir::StringAttr::get(&MCtx, "callee"),
      mlir::StringAttr::get(&MCtx, "matlab_dbg_leave_frame"));
  emitUnregOp("matlab.call_builtin", {},
              {mlir::NoneType::get(&MCtx)}, L, {Cal});
}

int32_t Lowerer::globalSlotId(Binding *Bnd) {
  auto It = GlobalIds.find(Bnd);
  if (It != GlobalIds.end()) return It->second;
  std::string Key;
  if (Bnd->Kind == BindingKind::Persistent) {
    Key = CurFnName + "." + std::string(Bnd->Name);
  } else {
    Key = std::string(Bnd->Name);
  }
  auto Nit = GlobalIdByName.find(Key);
  int32_t Id;
  if (Nit == GlobalIdByName.end()) {
    Id = NextGlobalId++;
    GlobalIdByName[Key] = Id;
  } else {
    Id = Nit->second;
  }
  GlobalIds[Bnd] = Id;
  return Id;
}

mlir::Value Lowerer::loadBinding(Binding *Bnd, const Type *ValTy,
                                 mlir::Location L) {
  if (!Bnd) return emitUnreg("matlab.undef", {}, mirTy(ValTy), L);
  /* REPL mode: script-level Var reads go through matlab_ws_get_*
   * so state survives across JIT invocations. Function-body Vars
   * still use normal slot lookup. If a local slot already exists
   * for this binding (e.g. because a for-loop init pre-allocated
   * one for its induction variable), prefer the slot — the loop
   * body writes into it per iteration, and reading the workspace
   * would miss the in-flight updates.
   *
   * Routing rule: when Sema concretely typed the read as a scalar
   * Double, fetch via matlab_ws_get_f64 so downstream consumers that
   * require an f64 (matlab.range bounds, scf.if conditions, scalar
   * arith) see a native f64 value. Otherwise — including the common
   * "Sema can't tell scalar vs matrix" case — fall back to
   * matlab_ws_get_mat returning ptr; the runtime auto-boxes stored
   * scalars into a 1x1 matrix so disp / matrix-op paths still work.
   * Without this split, `for i = 1:N` (with N a script-level Var)
   * left a matlab.range with a !llvm.ptr operand that LowerSeqLoops
   * refused to lower, surviving into the JIT as an unconvertible
   * matlab.* op. */
  /* #124: a struct / class-instance binding's local slot is only ever a
   * read-cache in REPL mode — assignments write the workspace (ws_set_*),
   * never the slot, and `ensureStructSlot` blank-inits the slot at function
   * entry (a fresh matlab_struct_new()) the first time a field access needs
   * it.  Under whole-file ReplMode (`-dap`), that blank-init runs *before*
   * the assigning call (`model = createpde()`) executes, so the slot holds a
   * stale empty struct disconnected from the workspace value.  NameExpr reads
   * before the slot is created (re)route to the workspace and see the live,
   * in-place-mutated pointer; reads after the slot exists fall to the stale
   * slot load below — the two diverge (poisson_disk: `geometryFromEdges`
   * mutates the workspace obj, but `solve` reads the blank slot -> u(0)=0).
   * The workspace is the single source of truth for these bindings (the same
   * rule `resolveStructBase` already follows for FieldAccess bases), so force
   * the workspace-read path even when a read-cache slot exists. */
  bool StructReadCache =
      Bnd->Kind == BindingKind::Var &&
      (StructInitialised.count(Bnd) || Bnd->IsStruct ||
       StructBindings.count(Bnd) || Bnd->PinnedClass != nullptr);
  if (ReplMode && InScriptBody && Bnd->Kind == BindingKind::Var &&
      (Slots.find(Bnd) == Slots.end() || StructReadCache) &&
      !isLocalHandle(Bnd) && !isFiBinding(Bnd)) {
    bool ScalarDouble = false;
    if (ValTy && ValTy->K == Type::Kind::Array) {
      auto &VA = static_cast<const ArrayType &>(*ValTy);
      ScalarDouble = VA.Elt == Dtype::Double && VA.S.K == Shape::Rank::Scalar;
    }
    /* String-typed bindings need a dedicated read entry: the workspace
     * stores them under kind=3, and matlab_ws_get_mat falls back to an
     * empty 0x0 matrix for non-mat kinds (after the kind=3 patch in
     * matlab_struct_get_mat it now passes through, but keeping the
     * dedicated entry isolates string reads from any future struct
     * helper changes and lets the DAP read-watchpoint path key on
     * "this is a string read"). Without this, a bare `t` or `disp(t)`
     * silently rendered as nothing because the load returned a fresh
     * empty matrix instead of the matlab_string* the assign stored. */
    bool IsString = false;
    if (StringBindings.count(Bnd)) IsString = true;
    else if (Bnd->InferredType &&
             Bnd->InferredType->K == Type::Kind::StringArray)
      IsString = true;
    else if (ValTy && ValTy->K == Type::Kind::StringArray)
      IsString = true;
    /* Phase 6 — sym binding read. Routes to matlab_ws_get_sym so the
     * stored matlab_sym* pointer comes back unmodified (matlab_ws_get_mat
     * would treat the descriptor as a matrix and return an empty
     * fallback). The IsSym flag is stamped on first declaration via the
     * Resolver hook (kind=7) for cross-input REPL persistence; the
     * SymBindings set covers same-TU references. */
    bool IsSymRead = SymBindings.count(Bnd) || Bnd->IsSym;
    bool IsSymmatRead = SymmatBindings.count(Bnd) || Bnd->IsSymmat;
    /* Function-handle read (kind=13).  Same-turn handles are tracked in
     * HandleBindings; cross-turn ones carry Binding::IsHandle stamped by
     * the Resolver from the workspace kind.  Only capture-free handles
     * round-trip (the stored value is a bare function pointer), so a
     * HandleBindings entry only counts when its capture list is empty.
     * Route to matlab_ws_get_handle so the stored pointer comes back
     * untouched for the call-site trampoline. */
    bool IsHandleRead = Bnd->IsHandle;
    if (!IsHandleRead) {
      auto HIt = HandleBindings.find(Bnd);
      if (HIt != HandleBindings.end() && HIt->second.empty())
        IsHandleRead = true;
    }
    mlir::Value NameV = emitFieldNameChar(Bnd->Name, L);
    if (IsHandleRead) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_handle"));
      return emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
    }
    if (IsSymmatRead) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_symmat"));
      return emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
    }
    if (IsSymRead) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_sym"));
      return emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
    }
    if (IsString) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_string"));
      return emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
    }
    if (ScalarDouble) {
      auto F64 = mlir::Float64Type::get(&MCtx);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_get_f64"));
      return emitUnreg("matlab.call_builtin", {NameV}, F64, L, {Cal});
    }
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_ws_get_mat"));
    return emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
  }
  /* Globals and persistents live in a runtime-backed scalar table.
   * Emit a matlab.call_builtin @matlab_global_get_f64(id) — the
   * generic call-builtin-to-llvm path lowers it to an opaque runtime
   * call. The slot ID is name-keyed so every function declaring the
   * same `global x` shares storage; `persistent y` inside function f
   * is keyed as "f.y" so it stays distinct from a like-named
   * persistent in another function. */
  if (Bnd->Kind == BindingKind::Global ||
      Bnd->Kind == BindingKind::Persistent) {
    int32_t Id = globalSlotId(Bnd);
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto I32 = mlir::IntegerType::get(&MCtx, 32);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Value IdV = mlir::arith::ConstantOp::create(
        B, L, I32, mlir::IntegerAttr::get(I32, (int64_t)Id));
    /* Pick the typed-pointer table for persistent fi-array bindings
     * (and any other heap-backed type whose Sema type is a non-scalar
     * array). The default scalar f64 table covers everything else —
     * persistent counters, persistent flags, etc. */
    bool UsePtr = false;
    if (Bnd->Kind == BindingKind::Persistent && ValTy &&
        ValTy->K == Type::Kind::Array) {
      auto &VA = static_cast<const ArrayType &>(*ValTy);
      if (VA.Elt == Dtype::Fixed && VA.S.K != Shape::Rank::Scalar)
        UsePtr = true;
    }
    llvm::SmallVector<mlir::NamedAttribute, 3> Attrs;
    Attrs.push_back(mlir::NamedAttribute(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx,
            UsePtr ? "matlab_persistent_get_ptr" : "matlab_global_get_f64")));
    if (Bnd->Kind == BindingKind::Persistent) {
      Attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(&MCtx, "persistent_name"),
          mlir::StringAttr::get(&MCtx, std::string(Bnd->Name))));
      Attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(&MCtx, "persistent_fn"),
          mlir::StringAttr::get(&MCtx, CurFnName)));
    }
    return emitUnreg("matlab.call_builtin", {IdV},
                     UsePtr ? (mlir::Type)PtrTy : (mlir::Type)F64, L, Attrs);
  }
  /* Numeric constants: MATLAB exposes pi / e / Inf / NaN / eps as
   * zero-arg builtins that evaluate to compile-time constants when
   * used as bare names (and to calls when invoked with args — not
   * supported here; the common `pi` / `Inf` case is the only one
   * most programs hit). Emit a direct arith.constant so downstream
   * arithmetic sees an f64 it can fold or lower cheaply. Numeric
   * values match MATLAB's definitions (eps = 2^-52, realmin /
   * realmax are the smallest / largest normal f64). */
  if (Bnd->Kind == BindingKind::Builtin) {
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto emitF = [&](double V) -> mlir::Value {
      return mlir::arith::ConstantOp::create(
          B, L, F64, mlir::FloatAttr::get(F64, V));
    };
    if (Bnd->Name == "pi")      return emitF(3.14159265358979323846);
    if (Bnd->Name == "e")       return emitF(2.71828182845904523536);
    if (Bnd->Name == "Inf")     return emitF(std::numeric_limits<double>::infinity());
    if (Bnd->Name == "NaN")     return emitF(std::numeric_limits<double>::quiet_NaN());
    if (Bnd->Name == "eps")     return emitF(2.2204460492503131e-16);
    if (Bnd->Name == "realmin") return emitF(2.2250738585072014e-308);
    if (Bnd->Name == "realmax") return emitF(1.7976931348623157e+308);
  }
  /* nargin / nargout: emit placeholder matlab.nargin / matlab.nargout
   * ops. A late pass rewrites them to arith.constant per-function AFTER
   * the monomorphiser has produced per-arity clones, so each clone
   * gets its own correct nargin/nargout value. */
  if (Bnd->Kind == BindingKind::Builtin &&
      (Bnd->Name == "nargin" || Bnd->Name == "nargout")) {
    auto F64 = mlir::Float64Type::get(&MCtx);
    llvm::StringRef OpName =
        (Bnd->Name == "nargin") ? "matlab.nargin" : "matlab.nargout";
    return emitUnreg(OpName, {}, F64, L);
  }
  /* MATLAB's implicit-call rule: a bare-name reference to a zero-arg
   * value-returning builtin on the RHS of an assignment or as a sub-
   * expression IS a call, not a function-handle.  Example:
   *   gpuTime = toc;   ← calls toc(), assigns the f64 elapsed time
   * The bare-name STATEMENT form (`toc;` on a line by itself) prints
   * "Elapsed time is ..." and is handled in the ExprStmt arm above.
   * Here we only handle the value-returning expression form. */
  if (Bnd->Kind == BindingKind::Builtin) {
    /* MATLAB's implicit-call rule: a bare-name reference to a zero-arg
     * value-returning builtin on the RHS of an assignment or as a
     * sub-expression IS a call, not a function-handle.  Example:
     *   gpuTime = toc;        ← calls toc(), returns f64
     *   h = gpuDevice;        ← returns the device handle (ptr)
     *   wait(gpuDevice);       ← same — gpuDevice is the call result, not @
     * The bare-name STATEMENT form (`toc;` on a line by itself) prints
     * "Elapsed time is …" and is handled in the ExprStmt arm above.
     * Here we emit a 0-arg matlab.call_builtin with the user name so
     * LowerTensorOps's pde_table dispatches to the right runtime
     * entry (matlab_toc / matlab_gpuDeviceCount / matlab_gpuDevice_handle). */
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Type RetTy;
    if (Bnd->Name == "toc") RetTy = F64;
    else if (Bnd->Name == "gpuDeviceCount") RetTy = F64;
    else if (Bnd->Name == "gpuDevice") RetTy = PtrTy;
    else if (Bnd->Name == "pwd") RetTy = PtrTy;  /* matlab_string* */
    if (RetTy) {
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, std::string(Bnd->Name)));
      return emitUnreg("matlab.call_builtin", {}, RetTy, L, {Cal});
    }
  }
  if (Bnd->Kind == BindingKind::Function ||
      Bnd->Kind == BindingKind::Builtin) {
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, std::string(Bnd->Name)));
    return emitUnreg("matlab.make_handle", {}, mirTy(ValTy), L, {Cal});
  }
  auto It = Slots.find(Bnd);
  if (It == Slots.end()) {
    // Declared but never stored — materialize a slot lazily.
    mlir::Value S = getOrCreateSlot(Bnd, ValTy, Bnd->Name, L);
    return emitLoad(S, mirTy(ValTy), L);
  }
  // Prefer the slot's own type when it's more concrete than what Sema
  // inferred (Sema falls back to `any`/NoneType for values it can't
  // specialize; the slot may have been created with a concrete scalar
  // type, e.g. the f64 spill slot of an anon function's block arg).
  mlir::Type LoadTy = mirTy(ValTy);
  mlir::Type SlotTy = It->second.getType();
  if (mlir::isa<mlir::NoneType>(LoadTy) &&
      !mlir::isa<mlir::NoneType>(SlotTy))
    LoadTy = SlotTy;
  return emitLoad(It->second, LoadTy, L);
}

//===----------------------------------------------------------------------===//
// Top-level
//===----------------------------------------------------------------------===//

mlir::ModuleOp Lowerer::lower(const TranslationUnit &TU) {
  mlir::ModuleOp M = mlir::ModuleOp::create(mlir::UnknownLoc::get(&MCtx));
  B.setInsertionPointToEnd(M.getBody());

  /* Stash the TU pointer so lowerScript can iterate Classes when
   * emitting the per-class debug-name registration (DebugMode only).
   * Cleared at the end of lower() so it never outlives this call. */
  CurTU = &TU;
  if (TU.ScriptNode) lowerScript(*TU.ScriptNode, M);
  for (const Function *F : TU.Functions) if (F) lowerFunction(*F, M);
  for (const ClassDef *C : TU.Classes) if (C) lowerClass(*C, M);
  CurTU = nullptr;

  return M;
}

void Lowerer::lowerClass(const ClassDef &C, mlir::ModuleOp M) {
  /* Each method is emitted as a flat free function with a mangled name
   * `ClassName__method`; the constructor uses the same form
   * `ClassName__ClassName`. Static methods follow the same convention
   * with `ClassName__` prefix but receive no implicit `obj` param.
   * Dispatch happens statically at call sites from a Sema-pinned
   * class — no v-table, no runtime method lookup. */
  for (const Function *Mth : C.Methods) if (Mth) lowerFunction(*Mth, M, &C);
  for (const Function *Mth : C.StaticMethods)
    if (Mth) lowerFunction(*Mth, M, &C, /*IsStatic=*/true);
}

void Lowerer::lowerScript(const Script &S, mlir::ModuleOp M) {
  mlir::OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToEnd(M.getBody());

  auto FnTy = mlir::FunctionType::get(&MCtx, {}, {});
  auto Fn = mlir::func::FuncOp::create(loc(S.Range), "script", FnTy);
  B.insert(Fn);

  auto *Entry = Fn.addEntryBlock();
  B.setInsertionPointToEnd(Entry);
  Slots.clear();
  CurFnName = "script";

  /* Register every classdef's name with the runtime so:
   *   - the DAP server can resolve a matlab_obj's class_id back to a
   *     printable class name (Debug/DAP path);
   *   - the REPL prelude-loader can scan workspace kind=2 bindings on
   *     a fresh turn and discover which classdefs to re-load, even
   *     when the user's input doesn't textually mention the class
   *     name (see `buildReplPrelude` in `tools/matlabc/main.cpp`).
   * Emitted as the first thing in the script body so the table is
   * populated before any constructor runs.  Cheap (one hash insert
   * per class per script entry); always-on. */
  if (CurTU) {
    auto I32 = mlir::IntegerType::get(&MCtx, 32);
    for (const ClassDef *C : CurTU->Classes) {
      if (!C || C->ClassId <= 0) continue;
      mlir::Value ClsId = mlir::arith::ConstantOp::create(
          B, loc(S.Range), I32,
          mlir::IntegerAttr::get(I32, (int64_t)C->ClassId));
      mlir::Value NameV = emitFieldNameChar(C->Name, loc(S.Range));
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_dbg_register_class"));
      emitUnregOp("matlab.call_builtin", {ClsId, NameV},
                  {mlir::NoneType::get(&MCtx)}, loc(S.Range), {Cal});
    }
  }

  bool SavedInScript = InScriptBody;
  InScriptBody = true;
  if (S.Body) lowerBlock(*S.Body);
  InScriptBody = SavedInScript;

  mlir::func::ReturnOp::create(B, loc(S.Range));
}

void Lowerer::lowerFunction(const Function &F, mlir::ModuleOp M,
                             const ClassDef *Owner, bool IsStatic) {
  mlir::OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToEnd(M.getBody());

  /* Save + reset GPU-pragma state.  Each function starts with a clean
   * slate; the `coder.gpu.kernelfun()` marker inside its body re-enables
   * the kernel lane.  Nested functions / methods don't inherit. */
  bool SavedInGpuKernelfun = InGpuKernelfun;
  bool SavedNextForIsGpuKernel = NextForIsGpuKernel;
  InGpuKernelfun = false;
  NextForIsGpuKernel = false;
  auto RestoreGpuState = llvm::make_scope_exit([&]() {
    InGpuKernelfun = SavedInGpuKernelfun;
    NextForIsGpuKernel = SavedNextForIsGpuKernel;
  });

  // Build parameter / result type vectors from Sema-inferred types.
  /* If the function's last input is `varargin`, that parameter
   * receives a matlab_cell pointer packed by the call site, so type
   * it as !llvm.ptr up front. */
  bool IsVariadic = !F.Inputs.empty() && F.Inputs.back() == "varargin";
  auto PtrTyArg = mlir::LLVM::LLVMPointerType::get(&MCtx);
  /* A class method's first input and a class constructor's first
   * output are both the object pointer (matlab_obj*). Tag each one
   * up-front so its slot is allocated ptr-typed and the binding is
   * recognised as a class instance by property / method dispatch. */
  bool IsCtor = Owner && !IsStatic && F.Name == Owner->Name;
  bool IsMethod = Owner && !IsStatic && !IsCtor;
  llvm::SmallVector<mlir::Type, 4> InTys, OutTys;
  for (size_t i = 0; i < F.ParamRefs.size(); ++i) {
    Binding *P = F.ParamRefs[i];
    if (IsVariadic && i + 1 == F.ParamRefs.size()) {
      InTys.push_back(PtrTyArg);
    } else if (IsMethod && i == 0) {
      InTys.push_back(PtrTyArg);
    } else if (P && P->PinnedClass) {
      /* Operator-overload methods may pin additional params to the
       * same class — type them as ptr so property access routes
       * through matlab_obj_get instead of the struct path. */
      InTys.push_back(PtrTyArg);
    } else {
      InTys.push_back(mirTy(P && P->InferredType ? P->InferredType : TC.any()));
    }
  }
  bool FnHasVarargout = !F.Outputs.empty() && F.Outputs.back() == "varargout";
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *O = F.OutputRefs[i];
    if (IsCtor && i == 0) {
      OutTys.push_back(PtrTyArg);
    } else if (FnHasVarargout && i + 1 == F.OutputRefs.size()) {
      /* Phase 1.2: varargout output is a matlab_cell* (ptr). The body
       * holds it in a ptr-typed slot and the implicit-return loads
       * the cell pointer; the call site unpacks per-LHS. */
      OutTys.push_back(PtrTyArg);
    } else {
      OutTys.push_back(mirTy(O && O->InferredType ? O->InferredType : TC.any()));
    }
  }

  auto FnTy = mlir::FunctionType::get(&MCtx, InTys, OutTys);
  std::string FnName;
  if (Owner)
    FnName = std::string(Owner->Name) + "__" + std::string(F.Name);
  else
    FnName = std::string(F.Name);
  /* Replace dots in method names (`get.Prop`, `set.Prop`) with an
   * underscore at the emitted-symbol level so the mangled name stays
   * a valid identifier in C / C++ output. */
  for (char &ch : FnName) if (ch == '.') ch = '_';
  auto Fn = mlir::func::FuncOp::create(loc(F.Range), FnName, FnTy);
  /* #40 Class 2/3: a per-arity / per-nargout clone produced by the
   * Sema-time monomorphizer carries an override so `runLowerNarginNargout`
   * folds the body's nargin / nargout to this call site's value (matching
   * what the late MLIR mono used to stamp). 0 means "unset". */
  if (F.NarginOverride > 0) {
    auto I64 = mlir::IntegerType::get(&MCtx, 64);
    Fn->setAttr("matlab.nargin_value",
                mlir::IntegerAttr::get(I64, (int64_t)F.NarginOverride));
  }
  if (F.NargoutOverride > 0) {
    auto I64 = mlir::IntegerType::get(&MCtx, 64);
    Fn->setAttr("matlab.nargout_value",
                mlir::IntegerAttr::get(I64, (int64_t)F.NargoutOverride));
  }
  /* Attach class-method metadata so the C++ emitter can reconstruct
   * idiomatic class{...}; blocks. These attributes are discardable
   * from a verifier perspective and ignored by the plain C backend. */
  if (Owner) {
    Fn->setAttr("matlab.class_name",
                mlir::StringAttr::get(&MCtx, std::string(Owner->Name)));
    llvm::StringRef Kind = IsCtor ? "ctor"
                          : IsStatic ? "static"
                          : "method";
    Fn->setAttr("matlab.method_kind",
                mlir::StringAttr::get(&MCtx, Kind));
    Fn->setAttr("matlab.method_name",
                mlir::StringAttr::get(&MCtx, std::string(F.Name)));
    if (Owner->Super)
      Fn->setAttr("matlab.class_super",
                  mlir::StringAttr::get(&MCtx,
                                         std::string(Owner->Super->Name)));
  }
  // Attach the MATLAB parameter name to each func arg as a discardable
  // attribute so downstream backends (EmitC) can print readable
  // signatures like `fact(double n)` instead of `fact(double v15)`.
  for (size_t i = 0; i < F.ParamRefs.size(); ++i) {
    Binding *Bnd = F.ParamRefs[i];
    if (!Bnd || Bnd->Name.empty()) continue;
    Fn.setArgAttr(i, mlir::StringAttr::get(&MCtx, "matlab.name"),
                  mlir::StringAttr::get(&MCtx, Bnd->Name));
    // Phase 5.6 Stage B: attach static shape + fi metadata for
    // vector-typed parameters so the downstream pipeline
    // (LowerStaticFiArrays / SV port emission) can recognize the
    // arg as an inferable static array. The TypeMapper has
    // already mapped the param type to `!llvm.ptr`, which loses
    // the shape and element-width info — these attrs reattach
    // it. Only set when the inferred type carries enough info
    // (Vector with known length + Fixed element with FxSpec).
    if (Bnd->InferredType &&
        Bnd->InferredType->K == Type::Kind::Array) {
      auto &AT = static_cast<const ArrayType &>(*Bnd->InferredType);
      if (AT.S.K == Shape::Rank::Vector && !AT.S.Dims.empty() &&
          AT.S.Dims[0] >= 1 && AT.Elt == Dtype::Fixed && AT.FxSpec) {
        auto I64 = mlir::IntegerType::get(&MCtx, 64);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        auto I1 = mlir::IntegerType::get(&MCtx, 1);
        Fn.setArgAttr(i,
            mlir::StringAttr::get(&MCtx, "matlab.array_n"),
            mlir::IntegerAttr::get(I64, AT.S.Dims[0]));
        Fn.setArgAttr(i,
            mlir::StringAttr::get(&MCtx, "matlab.fi_wl"),
            mlir::IntegerAttr::get(I32, (int64_t)AT.FxSpec->WordLength));
        Fn.setArgAttr(i,
            mlir::StringAttr::get(&MCtx, "matlab.fi_fl"),
            mlir::IntegerAttr::get(I32, (int64_t)AT.FxSpec->FractionLength));
        Fn.setArgAttr(i,
            mlir::StringAttr::get(&MCtx, "matlab.fi_signed"),
            mlir::IntegerAttr::get(I1, AT.FxSpec->Signed ? 1 : 0));
      }
    }
  }
  // Phase 5.6.2a: same for return-variable names. The SV emitter uses
  // these to give output ports human-readable names (`output ...
  // data_out, output ... overflow`) instead of the synthesized
  // `y, y1, y2, ...` fallback.
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bnd = F.OutputRefs[i];
    if (!Bnd || Bnd->Name.empty()) continue;
    if (IsCtor && i == 0) continue;
    Fn.setResultAttr(i, mlir::StringAttr::get(&MCtx, "matlab.name"),
                     mlir::StringAttr::get(&MCtx, Bnd->Name));
  }
  B.insert(Fn);

  auto *Entry = Fn.addEntryBlock();
  B.setInsertionPointToEnd(Entry);

  Slots.clear();
  CurFnName = std::string(F.Name);
  CurFnNargin = F.ParamRefs.size();
  CurFnNargout = F.OutputRefs.size();

  /* Push the debug frame BEFORE the parameter spills so the mirror
   * calls emitStore injects (in DebugMode) for each parameter store
   * land in this function's mini-workspace, not the caller's. The
   * displayed name is the bare function name for free functions and
   * "Class.method" for class methods (constructors print as
   * "Class.Class"). No-op outside DebugMode. */
  {
    std::string FrameName = std::string(F.Name);
    if (Owner) FrameName = std::string(Owner->Name) + "." + FrameName;
    emitDbgEnterFrame(FrameName, loc(F.Range));
  }

  // Spill parameters into slots. For the varargin tail, emit a
  // ptr-typed slot and register the binding as a cell so numel /
  // length / iscell(varargin) dispatch to the cell runtime.
  for (size_t i = 0; i < F.ParamRefs.size(); ++i) {
    Binding *Bnd = F.ParamRefs[i];
    if (!Bnd) continue;
    bool IsVarArg = IsVariadic && i + 1 == F.ParamRefs.size();
    bool IsSelfParam = IsMethod && i == 0;
    bool IsClassParam = Bnd->PinnedClass != nullptr;
    mlir::Value Slot;
    if (IsVarArg) {
      mlir::NamedAttribute NA(
          mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Bnd->Name)));
      Slot = emitUnreg("matlab.alloc", {}, PtrTyArg,
                       loc(F.Range), {NA});
      CellBindings.insert(Bnd);
    } else if (IsSelfParam || IsClassParam) {
      /* `obj` parameter of an ordinary method — or any other param
       * pinned to a user class by the resolver (e.g. the second
       * operand of an operator overload). Slot is ptr-typed so
       * property / method dispatch routes through matlab_obj_*. */
      mlir::NamedAttribute NA(
          mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Bnd->Name)));
      Slot = emitUnreg("matlab.alloc", {}, PtrTyArg,
                       loc(F.Range), {NA});
      if (IsSelfParam && !Bnd->PinnedClass)
        Bnd->PinnedClass = const_cast<ClassDef *>(Owner);
      /* Tag the alloc with the pinned class id so the DAP store-mirror
       * in emitStore knows to route through matlab_dbg_frame_set_obj
       * (which preserves class identity in the LOCALS panel) rather
       * than the generic matrix path. */
      if (Bnd->PinnedClass && Bnd->PinnedClass->ClassId > 0) {
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        Slot.getDefiningOp()->setAttr(
            "matlab.class_id",
            mlir::IntegerAttr::get(I32,
                                    (int64_t)Bnd->PinnedClass->ClassId));
      }
    } else {
      const Type *T = Bnd->InferredType ? Bnd->InferredType : TC.any();
      Slot = emitAlloc(T, Bnd->Name, loc(F.Range));
    }
    Slots[Bnd] = Slot;
    /* Phase 3 — note: cloning at the parameter spill is conceptually
     * the right thing for MATLAB value-class semantics, but the
     * existing in-tree class test corpus (class_basic.m,
     * class_dependent.m, class_operators.m, scn_class_instance_*
     * DAP scenarios) assumes handle-style method dispatch — they
     * call `acc.deposit(25)` without rebinding and expect the
     * receiver to mutate. Forcing a clone here breaks that path.
     *
     * We instead emit the clone at the AssignStmt level (`b = a`)
     * so user-level copy-on-assign works while method calls
     * preserve the existing reference-semantics behaviour the
     * corpus depends on. Full method-side value semantics is a
     * follow-up — needs a corpus update plus deeper rebind /
     * implicit-return wiring at every method call site. */
    emitStore(Entry->getArgument(i), Slot, loc(F.Range));
  }
  /* Phase 1.2: a function declared with `varargout` in its outputs
   * holds a matlab_cell* in the varargout slot. The body writes via
   * `varargout{k} = ...` (cell-store) and the call site unpacks. */
  bool HasVarargout = !F.Outputs.empty() && F.Outputs.back() == "varargout";
  // Pre-allocate output slots.
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bnd = F.OutputRefs[i];
    if (!Bnd) continue;
    bool IsCtorObj = IsCtor && i == 0;
    bool IsVarargoutSlot = HasVarargout && i + 1 == F.OutputRefs.size();
    mlir::Value Slot;
    if (IsVarargoutSlot) {
      /* Phase 1.2: varargout slot — cell pointer, initialised to an
       * empty cell so the first `varargout{k} = ...` has somewhere to
       * write. Tagged in CellBindings so numel/length/iscell route
       * through the cell runtime, and so the implicit-return packs
       * its contents into the call site's result tuple. */
      mlir::NamedAttribute NA(
          mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Bnd->Name)));
      Slot = emitUnreg("matlab.alloc", {}, PtrTyArg,
                       loc(F.Range), {NA});
      auto F64 = mlir::Float64Type::get(&MCtx);
      mlir::Value Zero = mlir::arith::ConstantOp::create(
          B, loc(F.Range), F64, mlir::FloatAttr::get(F64, 0.0));
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_cell_new"));
      mlir::Value EmptyCell = emitUnreg("matlab.call_builtin", {Zero},
                                         PtrTyArg, loc(F.Range), {Cal});
      emitStore(EmptyCell, Slot, loc(F.Range));
      CellBindings.insert(Bnd);
      Slots[Bnd] = Slot;
      continue;
    }
    if (IsCtorObj) {
      /* The constructor's first output is the newly-built object. Emit
       * a ptr-typed slot, then initialise it with matlab_obj_new(class_id)
       * before the user body runs so `obj.Prop = ...` has somewhere to
       * write. */
      mlir::NamedAttribute NA(
          mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Bnd->Name)));
      Slot = emitUnreg("matlab.alloc", {}, PtrTyArg,
                       loc(F.Range), {NA});
      Bnd->PinnedClass = const_cast<ClassDef *>(Owner);
      auto I32 = mlir::IntegerType::get(&MCtx, 32);
      /* Tag the alloc so emitStore picks up the class identity for
       * the DAP mirror (see analogous comment in the IsClassParam
       * branch above). */
      Slot.getDefiningOp()->setAttr(
          "matlab.class_id",
          mlir::IntegerAttr::get(I32, (int64_t)Owner->ClassId));
      mlir::Value ClsId = mlir::arith::ConstantOp::create(
          B, loc(F.Range), I32,
          mlir::IntegerAttr::get(I32, (int64_t)Owner->ClassId));
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_obj_new"));
      mlir::Value Obj = emitUnreg("matlab.call_builtin", {ClsId},
                                   PtrTyArg, loc(F.Range), {Cal});
      emitStore(Obj, Slot, loc(F.Range));
      /* Apply default property values, if any, by emitting the literal
       * and storing to the field via matlab_obj_set_f64 / _set_mat. */
      for (const auto &P : Owner->Props) {
        if (!P.Default) continue;
        mlir::Value DV = lowerExpr(*P.Default);
        mlir::Value ObjPtr = emitLoad(Slot, PtrTyArg, loc(F.Range));
        mlir::Value NameV = emitFieldNameChar(P.Name, loc(F.Range));
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        bool IsMat = DV && (DV.getType() == PtrTy ||
                            mlir::isa<mlir::RankedTensorType,
                                      mlir::UnrankedTensorType>(DV.getType()));
        llvm::StringRef Callee = IsMat ? "matlab_obj_set_mat"
                                       : "matlab_obj_set_f64";
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        emitUnregOp("matlab.call_builtin", {ObjPtr, NameV, DV},
                    {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal2});
      }
    } else {
      const Type *T = Bnd->InferredType ? Bnd->InferredType : TC.any();
      Slot = emitAlloc(T, Bnd->Name, loc(F.Range));
    }
    Slots[Bnd] = Slot;
  }
  // Pre-allocate local var slots so allocas stay at the function prologue.
  if (F.FnScope) {
    std::vector<std::pair<std::string, Binding *>> Locals;
    for (auto &[K, Bnd] : F.FnScope->locals())
      if (Bnd && Bnd->Kind == BindingKind::Var) Locals.emplace_back(K, Bnd);
    std::sort(Locals.begin(), Locals.end(),
              [](const auto &A, const auto &B) { return A.first < B.first; });
    for (auto &[N, Bnd] : Locals) {
      if (Slots.count(Bnd)) continue;
      const Type *T = Bnd->InferredType ? Bnd->InferredType : TC.any();
      mlir::Value Slot = emitAlloc(T, N, loc(F.Range));
      /* Tag class-instance locals so the DAP mirror routes them
       * through matlab_dbg_frame_set_obj. */
      if (Bnd->PinnedClass && Bnd->PinnedClass->ClassId > 0) {
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        Slot.getDefiningOp()->setAttr(
            "matlab.class_id",
            mlir::IntegerAttr::get(I32, (int64_t)Bnd->PinnedClass->ClassId));
      }
      Slots[Bnd] = Slot;
    }
  }

  if (F.Body) lowerBlock(*F.Body);

  // Implicit return: load each output slot and return.
  llvm::SmallVector<mlir::Value, 4> Rets;
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bnd = F.OutputRefs[i];
    if (!Bnd) continue;
    mlir::Value Slot = Slots[Bnd];
    Rets.push_back(emitLoad(Slot, OutTys[i], loc(F.Range)));
  }
  emitDbgLeaveFrame(loc(F.Range));
  mlir::func::ReturnOp::create(B, loc(F.Range), Rets);

  // Nested functions: emit at module level.
  for (const Function *N : F.Nested) if (N) lowerFunction(*N, M);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void Lowerer::lowerBlock(const ::matlab::Block &Blk) {
  for (const Stmt *S : Blk.Stmts) if (S) lowerStmt(*S);
}

bool Lowerer::stmtContainsBreakOrContinue(const Stmt &St) {
  switch (St.Kind) {
  case NodeKind::BreakStmt:
  case NodeKind::ContinueStmt:
    return true;
  case NodeKind::IfStmt: {
    auto &I = static_cast<const IfStmt &>(St);
    if (I.Then && blockContainsBreakOrContinue(*I.Then)) return true;
    for (auto &EI : I.Elseifs)
      if (EI.Body && blockContainsBreakOrContinue(*EI.Body)) return true;
    if (I.Else && blockContainsBreakOrContinue(*I.Else)) return true;
    return false;
  }
  case NodeKind::SwitchStmt: {
    auto &S = static_cast<const SwitchStmt &>(St);
    for (auto &C : S.Cases)
      if (C.Body && blockContainsBreakOrContinue(*C.Body)) return true;
    return false;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<const TryStmt &>(St);
    if (T.TryBody && blockContainsBreakOrContinue(*T.TryBody)) return true;
    if (T.CatchBody && blockContainsBreakOrContinue(*T.CatchBody)) return true;
    return false;
  }
  /* for/while establish their OWN break/continue scope — nested break
   * inside a sub-loop binds to that sub-loop, not the outer one. So we
   * don't recurse into their bodies. */
  case NodeKind::ForStmt:
  case NodeKind::WhileStmt:
    return false;
  default:
    return false;
  }
}

bool Lowerer::blockContainsBreakOrContinue(const ::matlab::Block &Blk) {
  for (const Stmt *S : Blk.Stmts)
    if (S && stmtContainsBreakOrContinue(*S)) return true;
  return false;
}

mlir::Value Lowerer::fixupIfCond(mlir::OpBuilder &B, mlir::Value Cond,
                                  mlir::Location LC) {
  mlir::Type CT = Cond.getType();
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(CT)) {
    if (IT.getWidth() == 1) return Cond;
    mlir::Value Zero = mlir::arith::ConstantOp::create(
        B, LC, IT, mlir::IntegerAttr::get(IT, 0));
    return mlir::arith::CmpIOp::create(
        B, LC, mlir::arith::CmpIPredicate::ne, Cond, Zero);
  }
  if (mlir::isa<mlir::Float64Type, mlir::Float32Type>(CT)) {
    auto FT = mlir::cast<mlir::FloatType>(CT);
    mlir::Value Zero = mlir::arith::ConstantOp::create(
        B, LC, FT, mlir::FloatAttr::get(FT, 0.0));
    return mlir::arith::CmpFOp::create(
        B, LC, mlir::arith::CmpFPredicate::ONE, Cond, Zero);
  }
  if (mlir::isa<mlir::NoneType>(CT)) {
    auto I1 = mlir::IntegerType::get(&MCtx, 1);
    return mlir::UnrealizedConversionCastOp::create(
               B, LC, mlir::TypeRange{I1}, mlir::ValueRange{Cond})
        .getResult(0);
  }
  /* Matrix-pointer conditions appear in DAP/REPL mode whenever a
   * scalar slot is fetched via matlab_ws_get_mat: the runtime returns
   * a 1x1 matlab_mat*, and any matlab.lt/gt/etc. on it propagates the
   * ptr type. Route through matlab_mat_truth(ptr) -> i8 (1 iff
   * MATLAB's `if M` is true: non-empty AND every element non-zero),
   * then compare against zero to materialise an i1.
   *
   * A matrix-VALUED comparison whose operand keeps the Sema tensor type
   * (`if abs(M) < c` — abs returns the Sema array type tensor<*xf64>, so
   * matlab.lt yields tensor<*xi1>) reaches here as a tensor, not a ptr
   * (#120). scf.if rejects a tensor operand, so wrap it the same way:
   * matlab_mat_truth tolerates the tensor operand (it's lowered by
   * LowerTensorOps::rewriteMatTruth only once the producing comparison
   * is rewritten to a matlab_mat* ptr — see that pass), giving MATLAB's
   * "if every element is true" reduction. */
  if (mlir::isa<mlir::LLVM::LLVMPointerType, mlir::RankedTensorType,
                mlir::UnrankedTensorType>(CT)) {
    auto I8 = mlir::IntegerType::get(&MCtx, 8);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_mat_truth"));
    mlir::Value I8V = emitUnreg("matlab.call_builtin", {Cond}, I8, LC, {Cal});
    mlir::Value Zero = mlir::arith::ConstantOp::create(
        B, LC, I8, mlir::IntegerAttr::get(I8, 0));
    return mlir::arith::CmpIOp::create(
        B, LC, mlir::arith::CmpIPredicate::ne, I8V, Zero);
  }
  // Pass through anything else (e.g. already-i1) unchanged.
  return Cond;
}

void Lowerer::lowerLoopBody(const ::matlab::Block &Blk) {
  /* Walk statements. After any stmt that might have broken/continued,
   * wrap the remainder in scf.if(!did_break && !did_continue) { ... }.
   * The flags are stored in the top-of-stack LoopCtx. */
  auto I1 = mlir::IntegerType::get(&MCtx, 1);
  auto wrap = [&](size_t Start) {
    if (LoopStack.empty()) {
      for (size_t j = Start; j < Blk.Stmts.size(); ++j)
        if (Blk.Stmts[j]) lowerStmt(*Blk.Stmts[j]);
      return;
    }
    auto &Ctx = LoopStack.back();
    mlir::Location L = loc(Blk.Range);
    mlir::Value BV = emitLoad(Ctx.BreakSlot, I1, L);
    mlir::Value CV = emitLoad(Ctx.ContinueSlot, I1, L);
    mlir::Value True = mlir::arith::ConstantOp::create(
        B, L, I1, mlir::IntegerAttr::get(I1, 1));
    mlir::Value NotBr = mlir::arith::XOrIOp::create(B, L, BV, True);
    mlir::Value NotCt = mlir::arith::XOrIOp::create(B, L, CV, True);
    mlir::Value Cond = mlir::arith::AndIOp::create(B, L, NotBr, NotCt);
    auto IfOp = mlir::scf::IfOp::create(B, L, mlir::TypeRange{}, Cond,
                                         /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard G(B);
    /* Insert before scf.yield so cloned ops don't land after the
     * terminator. IfOp auto-creates an empty then block with a
     * scf.yield terminator. */
    B.setInsertionPoint(IfOp.thenBlock()->getTerminator());
    /* Recurse so nested risky stmts in the tail get the same treatment. */
    matlab::Block Sub;
    Sub.Range = Blk.Range;
    for (size_t j = Start; j < Blk.Stmts.size(); ++j)
      Sub.Stmts.push_back(Blk.Stmts[j]);
    lowerLoopBody(Sub);
  };
  for (size_t i = 0; i < Blk.Stmts.size(); ++i) {
    const Stmt *S = Blk.Stmts[i];
    if (!S) continue;
    lowerStmt(*S);
    if (stmtContainsBreakOrContinue(*S) && i + 1 < Blk.Stmts.size()) {
      wrap(i + 1);
      return;
    }
  }
}

void Lowerer::lowerStmt(const Stmt &St) {
  /* Debug-mode hook: emit matlab_dbg_hook(file_id, line) at the
   * start of every statement. The compiled-in runtime sees
   * file_id + line, checks its breakpoint / step state, and blocks
   * if it should pause. Skipped for Blocks (the block's children
   * each get their own hook) and for a few no-value control
   * structures where a hook on the keyword is redundant with the
   * one on the first inner statement. */
  if (DebugMode && SM && St.Range.Begin.isValid() &&
      St.Kind != NodeKind::Block) {
    auto LC = SM->getLineColumn(St.Range.Begin);
    uint32_t HookLine = LC.Line;
    /* Normalize the hook line to the first non-blank, non-comment-only
     * source line within [Begin.Line, End.Line]. For well-formed
     * statements Begin already points at code and the loop exits on
     * its first iteration, so this is a no-op. The slide-forward path
     * matters when a parse path anchors Begin to a position that ends
     * up on a blank/comment line — without it, stepping would land on
     * a row with no executable code, which is confusing in the IDE.
     * The walk is bounded by End.Line so it can never cross into the
     * next statement and steal its line. */
    if (St.Range.End.isValid()) {
      uint32_t EndLine = SM->getLineColumn(St.Range.End).Line;
      while (HookLine < EndLine) {
        auto Text = SM->getLineText(St.Range.Begin.File, HookLine);
        size_t I = 0;
        while (I < Text.size() && (Text[I] == ' ' || Text[I] == '\t'))
          ++I;
        bool Blank = (I == Text.size());
        bool CommentOnly = (I < Text.size() &&
                            (Text[I] == '%' || Text[I] == '#'));
        if (!Blank && !CommentOnly) break;
        ++HookLine;
      }
    }
    auto I32 = mlir::IntegerType::get(&MCtx, 32);
    mlir::Value FileV = mlir::arith::ConstantOp::create(
        B, loc(St.Range), I32,
        mlir::IntegerAttr::get(I32, (int64_t)St.Range.Begin.File));
    mlir::Value LineV = mlir::arith::ConstantOp::create(
        B, loc(St.Range), I32,
        mlir::IntegerAttr::get(I32, (int64_t)HookLine));
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_dbg_hook"));
    emitUnregOp("matlab.call_builtin", {FileV, LineV},
                {mlir::NoneType::get(&MCtx)}, loc(St.Range), {Cal});
  }
  switch (St.Kind) {
  case NodeKind::ExprStmt: {
    auto &E = static_cast<const ExprStmt &>(St);
    if (!E.E) return;
    /* GPU Coder pragma intercept — `coder.gpu.kernelfun()` and
     * `coder.gpu.kernel` are folded by Parser into NameExprs
     * `coder_gpu_kernelfun` / `coder_gpu_kernel`.  They have no
     * runtime semantics: the kernelfun-form flags every for-loop in
     * the enclosing function as a GPU kernel, the kernel-form flags
     * only the next for-loop.  Drop the marker call without emitting
     * IR. */
    if (auto *CI = dynamic_cast<const CallOrIndex *>(E.E)) {
      if (auto *NE = dynamic_cast<const NameExpr *>(CI->Callee)) {
        if (NE->Name == "coder_gpu_kernelfun") {
          InGpuKernelfun = true;
          return;
        }
        if (NE->Name == "coder_gpu_kernel") {
          NextForIsGpuKernel = true;
          return;
        }
      }
    }
    /* Bare-name forms (no parens) `coder.gpu.kernelfun` / `coder.gpu.kernel`
     * also fold to NameExpr and reach here as ExprStmt(NameExpr). */
    if (auto *NE = dynamic_cast<const NameExpr *>(E.E)) {
      if (NE->Name == "coder_gpu_kernelfun") {
        InGpuKernelfun = true;
        return;
      }
      if (NE->Name == "coder_gpu_kernel") {
        NextForIsGpuKernel = true;
        return;
      }
    }
    /* Bare-name invocation of side-effect-only builtins like `who` /
     * `whos` / `clear`: MATLAB allows `who` at the prompt as a
     * command, parsed here as `ExprStmt(NameExpr("who"))`. Without
     * this special case, lowerExpr would emit a matlab.make_handle
     * and the implicit-display path would try to print a function
     * handle. Treat them as zero-arg calls to the runtime entry. */
    if (auto *NE = dynamic_cast<const NameExpr *>(E.E)) {
      if (NE->Ref && NE->Ref->Kind == BindingKind::Builtin) {
        llvm::StringRef RN;
        if (NE->Name == "who")  RN = "matlab_ws_who";
        else if (NE->Name == "whos") RN = "matlab_ws_whos";
        else if (NE->Name == "clear") RN = "matlab_ws_clear";
        /* `keyboard` drops the worker into a paused state at the
         * next hook firing — same machinery as a breakpoint, but
         * triggered by the program itself rather than a DAP-set
         * bp. The IDE's REPL panel (already wired via
         * evaluate context=repl) takes over from there.
         *
         * Outside DebugMode the runtime function is a no-op (the
         * matlab_dbg.enabled flag is 0), so a `keyboard` call in
         * a release-mode binary does nothing — same posture as
         * the breakpoint hook itself. */
        else if (NE->Name == "keyboard") RN = "matlab_dbg_keyboard_hook";
        /* `tic` / `toc` / `pause` typed bare (no parens) — MATLAB
         * command-syntax form. Bare `pause` blocks until a keypress
         * (matched by matlab_pause_keypress); bare `toc` prints the
         * formatted "Elapsed time is X seconds." line that MATLAB
         * emits when toc is used as a statement. */
        else if (NE->Name == "tic")   RN = "matlab_tic";
        else if (NE->Name == "toc")   RN = "matlab_toc_print";
        else if (NE->Name == "pause") RN = "matlab_pause_keypress";
        /* Bare `cd` with no argument → change to $HOME, matching the
         * common interactive shortcut. `cd <dir>` is a CommandStmt and
         * `cd('dir')` a call — both handled separately. */
        else if (NE->Name == "cd")    RN = "matlab_cd_home";
        if (!RN.empty()) {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, RN));
          emitUnregOp("matlab.call_builtin", {},
                      {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
          return;
        }
      }
    }
    /* #286: mark the bare top-level call so its open user-function result
     * type gets nudged to concrete (see CallOrIndex lowering) — but only
     * when this statement will actually display (non-suppressed). */
    const void *SavedBareDisplay = BareDisplayCall;
    if (!E.Suppressed && dynamic_cast<const CallOrIndex *>(E.E))
      BareDisplayCall = E.E;
    mlir::Value V = lowerExpr(*E.E);
    BareDisplayCall = SavedBareDisplay;
    /* Implicit display on a non-suppressed bare expression: MATLAB
     * prints `name =\n<value>` for a NameExpr and `ans =\n<value>`
     * for any other expression, and additionally binds the value to
     * `ans` in the workspace (REPL mode). Skip when the expression
     * has no value (void call result like `disp(x)`) or the
     * statement ends in `;`. */
    if (E.Suppressed) return;
    if (!V || mlir::isa<mlir::NoneType>(V.getType())) return;
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    /* Function-handle / anon values aren't meaningfully displayable. */
    if (auto *NE = dynamic_cast<const NameExpr *>(E.E))
      if (NE->Ref && HandleBindings.count(NE->Ref)) return;
    std::string Label;
    if (auto *NE = dynamic_cast<const NameExpr *>(E.E))
      Label = std::string(NE->Name) + " =";
    else
      Label = "ans =";
    mlir::NamedAttribute LV(
        mlir::StringAttr::get(&MCtx, "value"),
        mlir::StringAttr::get(&MCtx, Label));
    mlir::Value LabelV = emitUnreg("matlab.const_char", {},
                                    mlir::NoneType::get(&MCtx),
                                    loc(E.Range), {LV});
    mlir::NamedAttribute DispCal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "disp"));
    emitUnregOp("matlab.call_builtin", {LabelV},
                {mlir::NoneType::get(&MCtx)}, loc(E.Range), {DispCal});
    /* For string-typed bare expressions (`"Test"`, `upper(s)`, ...)
     * route the value-disp through matlab_string_disp directly. The
     * generic "disp" callee otherwise survives into LowerTensorOps's
     * matrix-disp lowering and prints the matlab_string descriptor's
     * bytes as a matrix. Detect via Sema's StringArray type or a bare
     * StringLiteral. */
    {
      bool DispIsString = (E.E->Ty &&
                           E.E->Ty->K == Type::Kind::StringArray) ||
                          E.E->Kind == NodeKind::StringLiteral;
      if (auto *NE = dynamic_cast<const NameExpr *>(E.E)) {
        if (NE->Ref && StringBindings.count(NE->Ref))
          DispIsString = true;
        /* Cross-REPL-input case: a fresh translation unit that just
         * mentions `t` doesn't repopulate StringBindings; rely on the
         * binding's persisted InferredType. */
        else if (NE->Ref && NE->Ref->InferredType &&
                 NE->Ref->InferredType->K == Type::Kind::StringArray)
          DispIsString = true;
      }
      /* Phase 5.1: datetime / duration disp dispatch.
       * Phase 5.2: categorical disp dispatch. */
      bool DispIsDatetime = false, DispIsDuration = false;
      bool DispIsDatetimeVec = false, DispIsDurationVec = false;
      bool DispIsCategorical = false, DispIsTable = false;
      bool DispIsTimetable = false;
      bool DispIsSym = exprIsSym(E.E);
      bool DispIsSymmat = exprIsSymmat(E.E);
      if (auto *NE = dynamic_cast<const NameExpr *>(E.E)) {
        if (NE->Ref && DatetimeBindings.count(NE->Ref)) DispIsDatetime = true;
        if (NE->Ref && DurationBindings.count(NE->Ref)) DispIsDuration = true;
        if (NE->Ref && DatetimeVecBindings.count(NE->Ref)) DispIsDatetimeVec = true;
        if (NE->Ref && DurationVecBindings.count(NE->Ref)) DispIsDurationVec = true;
        if (NE->Ref && CategoricalBindings.count(NE->Ref)) DispIsCategorical = true;
        if (NE->Ref && isTableBinding(NE->Ref)) DispIsTable = true;
        if (NE->Ref && isTimetableBinding(NE->Ref)) DispIsTimetable = true;
      }
      llvm::StringRef IntSuf = Lowerer::intDtypeSuffixOf(E.E);
      if (DispIsString) {
        mlir::NamedAttribute SCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_string_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {SCal});
      } else if (DispIsDatetimeVec) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_datetime_vec_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsDurationVec) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_duration_vec_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsDatetime) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_datetime_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsDuration) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_duration_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsCategorical) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_categorical_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsTimetable) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_timetable_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsTable) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_table_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsSym) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_sym_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (DispIsSymmat) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_symmat_disp"));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {Cal});
      } else if (!IntSuf.empty()) {
        /* Phase 1.1.C — typed int matrix disp. Sema marks the expression
         * as Int32 / UInt8 array so we can emit the typed callee directly
         * and avoid the polymorphic matlab_disp_mat path (which expects
         * the f64 layout). */
        std::string TyCallee = ("matlab_mat_" + IntSuf + "_disp").str();
        mlir::NamedAttribute TCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, TyCallee));
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {TCal});
      } else {
        emitUnregOp("matlab.call_builtin", {V},
                    {mlir::NoneType::get(&MCtx)}, loc(E.Range), {DispCal});
      }
    }
    /* In REPL mode, bind non-named-expression results to `ans` in the
     * workspace so subsequent inputs can reference them. A bare
     * NameExpr already has its value stored under its own name, so
     * no extra write is needed for that case. */
    if (ReplMode && InScriptBody && E.E->Kind != NodeKind::NameExpr) {
      mlir::Value AnsName = emitFieldNameChar("ans", loc(E.Range));
      bool IsMat = V.getType() == PtrTy ||
                    mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(V.getType());
      /* Same string-aware guard as the AssignStmt path: a bare
       * `"Test"` at the REPL produces a matlab_string* (LLVM ptr)
       * which would otherwise route through matlab_ws_set_mat and
       * have its descriptor reinterpreted as matlab_mat. Detect via
       * Sema's inferred type on the expression. */
      bool IsString = (E.E->Ty &&
                       E.E->Ty->K == Type::Kind::StringArray) ||
                      E.E->Kind == NodeKind::StringLiteral;
      llvm::StringRef Callee = IsString ? "matlab_ws_set_string"
                                        : (IsMat ? "matlab_ws_set_mat"
                                                 : "matlab_ws_set_f64");
      mlir::NamedAttribute WsCal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      emitUnregOp("matlab.call_builtin", {AnsName, V},
                  {mlir::NoneType::get(&MCtx)}, loc(E.Range), {WsCal});
    }
    return;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<const AssignStmt &>(St);
    /* If RHS is an anonymous function or a function handle, tag the LHS
     * binding so later reads through it call the handle rather than
     * trying to subscript a matrix. */
    bool RhsIsHandle = A.RHS && (A.RHS->Kind == NodeKind::AnonFunction ||
                                 A.RHS->Kind == NodeKind::FuncHandle);
    /* Track cell-typed bindings so downstream numel/length/iscell
     * calls can route to the matlab_cell_* runtime. Both bare
     * CellLiteral and calls to known cell-producing builtins qualify;
     * for v1 we cover the literal case. */
    /* Phase 1.3: a MatrixLiteral whose elements are all cell-bound names
     * or CellLiterals is a cell concatenation expression — treat the LHS
     * as cell-bound just like a direct CellLiteral RHS. Without this,
     * `R = [A, B]` (A, B cells) would not flag R in CellBindings, so the
     * subsequent `size(R, 1)` would route through the matrix runtime
     * and read garbage from the cell layout. */
    auto isCellExprForAssign = [&](const Expr *X) -> bool {
      if (!X) return false;
      if (X->Kind == NodeKind::CellLiteral) return true;
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        if (NE->Ref && CellBindings.count(NE->Ref)) return true;
      return false;
    };
    /* Phase 5.1: datetime / duration RHS detection. Either a direct
     * builtin call (`datetime(...)`, `seconds(n)`, etc.) or a
     * NameExpr that's already tagged. Binop results land in their
     * respective sets via the BinaryOp emission below. */
    bool RhsIsDatetime = false;
    bool RhsIsDuration = false;
    bool RhsIsDatetimeVec = false;
    bool RhsIsDurationVec = false;
    bool RhsIsCategorical = false;
    bool RhsIsTable = false;
    bool RhsIsTimetable = false;
    bool RhsIsTimerange = false;
    /* RHS is a VideoWriter handle when it's a direct `VideoWriter(...)`
     * call or a re-assignment from an existing VideoWriter binding. */
    bool RhsIsVideoWriter = false;
    /* Plain matlab_struct* RHS — needs a dedicated workspace setter
     * (matlab_ws_set_struct, kind=9) so field-access dispatch sees
     * `s` as struct rather than mat on subsequent REPL turns.  RHS
     * is struct when it's a direct call to a known struct-returning
     * builtin (struct(...) itself, plus the PROP linkBudget) or a
     * NameExpr referencing a previously-tagged struct binding. */
    bool RhsIsStruct = false;
    /* Bioinformatics fastaread returns a struct array (matlab_struct_arr*);
     * tag the LHS into StructArrayBindings so element field reads route
     * through the struct-array path. */
    bool RhsIsStructArray = false;
    /* Phase 6 — Symbolic Math Toolbox. RHS is sym-typed when:
     *  - direct call to a sym-producing builtin
     *  - NameExpr already in SymBindings (re-assignment)
     *  - BinaryOp where either operand is sym (handled below)
     *  - UnaryOp with sym operand */
    bool RhsIsSym = exprIsSym(A.RHS);
    bool RhsIsSymmat = exprIsSymmat(A.RHS);
    if (A.RHS && A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *Cx = static_cast<const CallOrIndex *>(A.RHS);
      if (auto *NE = dynamic_cast<const NameExpr *>(Cx->Callee)) {
        if (NE->Name == "categorical") RhsIsCategorical = true;
        /* readtable returns a matlab_table*, so its result binding
         * gets the same dispatch as `T = table(...)`. */
        if (NE->Name == "table" || NE->Name == "readtable")
          RhsIsTable = true;
        /* Phase 5.4 (cont.): timetable + table2timetable produce
         * matlab_timetable*. The LHS slot gets tagged so disp /
         * height / etc. route to matlab_timetable_*. */
        if (NE->Name == "timetable" || NE->Name == "table2timetable" ||
            NE->Name == "retime" || NE->Name == "synchronize" ||
            NE->Name == "fillmissing" ||
            NE->Name == "movavg" || NE->Name == "macd")
          RhsIsTimetable = true;
        /* TT(rowIdx, :) and TT(:, 'colName') both return a new
         * timetable, so the LHS slot inherits the tag. */
        if (NE->Ref && isTimetableBinding(NE->Ref))
          RhsIsTimetable = true;
        if (NE->Name == "timerange") RhsIsTimerange = true;
        if (NE->Name == "VideoWriter") RhsIsVideoWriter = true;
        if (isVideoWriterBinding(NE->Ref))
          RhsIsVideoWriter = true;
        /* Known struct-returning builtins. struct() is the textbook
         * literal; linkBudget is the PROP-Tier-2b struct return; stepinfo
         * returns the CST step-response-metrics struct. Adding more is a
         * one-liner per future entry.
         *
         * #77: the PDE surface/mesh loaders return a matlab_struct* and
         * can legitimately yield NULL (e.g. pde_load_glb on a missing
         * file).  Sema lumps them into the matrix-returning block, so
         * without this tag the REPL/JIT workspace round-trip stores the
         * struct (or NULL) through matlab_ws_set_mat (kind=1, matrix) and
         * reads it back mis-typed — a NULL becomes a fresh mat_alloc(0,0)
         * that the downstream struct accessor dereferences as a
         * matlab_struct*, a heap-dependent wild crash.  Tagging them
         * struct routes the store through matlab_ws_set_struct (kind=12),
         * which round-trips NULL faithfully (matlab_struct_get_mat). */
        if (NE->Name == "struct" || NE->Name == "linkBudget" ||
            NE->Name == "stepinfo" ||
            NE->Name == "bleLLDataChannelPDUDecode" ||
            NE->Name == "bleL2CAPFrameDecode" ||
            NE->Name == "basecount" || NE->Name == "aacount" ||
            NE->Name == "atomiccomp" ||
            NE->Name == "pde_load_glb" || NE->Name == "pde_load_stl" ||
            NE->Name == "pde_voxelize_surface")
          RhsIsStruct = true;
        /* Bioinformatics fastaread returns a matlab_struct_arr* — tag the
         * LHS as a struct array so `s(i).Header` / `length(s)` route through
         * the struct-array read path (same set the `s(i).x=v` store fills). */
        if (NE->Name == "fastaread") RhsIsStructArray = true;
      }
    } else if (A.RHS && A.RHS->Kind == NodeKind::NameExpr) {
      auto *NE = static_cast<const NameExpr *>(A.RHS);
      if (NE->Ref && CategoricalBindings.count(NE->Ref))
        RhsIsCategorical = true;
      if (NE->Ref && isTableBinding(NE->Ref))
        RhsIsTable = true;
      if (NE->Ref && isTimetableBinding(NE->Ref))
        RhsIsTimetable = true;
      if (NE->Ref &&
          (StructInitialised.count(NE->Ref) || NE->Ref->IsStruct))
        RhsIsStruct = true;
      /* #258: a struct-array COPY `t = s` must propagate struct-array-ness so
       * `t(i).Field` / `length(t)` route through the struct-array path and the
       * workspace store persists `t` as kind=14.  Without this the copy lost
       * the tag and `t(i).Field` fell to resolveStructBase -> undef. */
      if (NE->Ref && isStructArrayBinding(NE->Ref))
        RhsIsStructArray = true;
    } else if (A.RHS && A.RHS->Kind == NodeKind::FieldAccess) {
      /* Phase 5.4: TT.Time produces a matlab_datetime_vec *. TT.<col>
       * is a plain matlab_mat — the default lane handles it. */
      auto *FA = static_cast<const FieldAccess *>(A.RHS);
      if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
        if (BN->Ref && isTimetableBinding(BN->Ref) &&
            FA->Field == "Time")
          RhsIsDatetimeVec = true;
    }
    if (A.RHS && A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *Cx = static_cast<const CallOrIndex *>(A.RHS);
      if (auto *NE = dynamic_cast<const NameExpr *>(Cx->Callee)) {
        if (NE->Name == "datetime") RhsIsDatetime = true;
        /* Financial Toolbox Tier-1: date-arithmetic helpers that
         * return matlab_datetime *. Tag the LHS so disp routes
         * to matlab_datetime_disp. */
        else if (NE->Name == "daysadd"  || NE->Name == "busdate"   ||
                 NE->Name == "eomdate"  || NE->Name == "lweekdate" ||
                 NE->Name == "fweekdate")
          RhsIsDatetime = true;
        else if (NE->Name == "seconds" || NE->Name == "minutes" ||
                 NE->Name == "hours"   || NE->Name == "days"    ||
                 NE->Name == "years"   || NE->Name == "duration") {
          /* Phase 5.4: `days(0:251)` etc. → duration_vec; scalar
           * f64 arg → scalar duration. Detect the colon-range form
           * syntactically so we don't depend on the lowered Value's
           * runtime type. */
          if (!Cx->Args.empty() && Cx->Args[0] &&
              (dynamic_cast<const RangeExpr *>(Cx->Args[0]) ||
               dynamic_cast<const ColonExpr *>(Cx->Args[0]) ||
               dynamic_cast<const MatrixLiteral *>(Cx->Args[0])))
            RhsIsDurationVec = true;
          else
            RhsIsDuration = true;
        }
      }
    } else if (A.RHS && A.RHS->Kind == NodeKind::NameExpr) {
      auto *NE = static_cast<const NameExpr *>(A.RHS);
      if (NE->Ref && DatetimeBindings.count(NE->Ref))    RhsIsDatetime    = true;
      if (NE->Ref && DurationBindings.count(NE->Ref))    RhsIsDuration    = true;
      if (NE->Ref && DatetimeVecBindings.count(NE->Ref)) RhsIsDatetimeVec = true;
      if (NE->Ref && DurationVecBindings.count(NE->Ref)) RhsIsDurationVec = true;
    } else if (A.RHS && A.RHS->Kind == NodeKind::BinaryOp) {
      auto *BX = static_cast<const BinaryOpExpr *>(A.RHS);
      auto isDt = [&](const Expr *X) -> bool {
        if (auto *NE = dynamic_cast<const NameExpr *>(X))
          return NE->Ref && DatetimeBindings.count(NE->Ref);
        if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
          if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee))
            if (NE->Name == "datetime") return true;
        return false;
      };
      auto argIsRange = [](const CallOrIndex *CX) -> bool {
        return !CX->Args.empty() && CX->Args[0] &&
               (dynamic_cast<const RangeExpr *>(CX->Args[0]) ||
                dynamic_cast<const ColonExpr *>(CX->Args[0]) ||
                dynamic_cast<const MatrixLiteral *>(CX->Args[0]));
      };
      auto isDur = [&](const Expr *X) -> bool {
        if (auto *NE = dynamic_cast<const NameExpr *>(X))
          return NE->Ref && DurationBindings.count(NE->Ref);
        if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
          if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee))
            if (NE->Name == "seconds" || NE->Name == "minutes" ||
                NE->Name == "hours"   || NE->Name == "days"    ||
                NE->Name == "years"   || NE->Name == "duration") {
              if (argIsRange(CX))
                return false;       /* duration_vec, handled below */
              return true;
            }
        return false;
      };
      auto isDtVec = [&](const Expr *X) -> bool {
        if (auto *NE = dynamic_cast<const NameExpr *>(X))
          return NE->Ref && DatetimeVecBindings.count(NE->Ref);
        return false;
      };
      auto isDurVec = [&](const Expr *X) -> bool {
        if (auto *NE = dynamic_cast<const NameExpr *>(X))
          return NE->Ref && DurationVecBindings.count(NE->Ref);
        if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
          if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee))
            if (NE->Name == "seconds" || NE->Name == "minutes" ||
                NE->Name == "hours"   || NE->Name == "days"    ||
                NE->Name == "years")
              if (argIsRange(CX))
                return true;
        return false;
      };
      /* Vec-producing forms first. */
      if (BX->Op == BinOp::Add &&
          ((isDt(BX->LHS) && isDurVec(BX->RHS)) ||
           (isDurVec(BX->LHS) && isDt(BX->RHS))))
        RhsIsDatetimeVec = true;
      else if (BX->Op == BinOp::Add &&
               ((isDtVec(BX->LHS) && isDur(BX->RHS)) ||
                (isDur(BX->LHS) && isDtVec(BX->RHS))))
        RhsIsDatetimeVec = true;
      else if (BX->Op == BinOp::Sub && isDtVec(BX->LHS) && isDur(BX->RHS))
        RhsIsDatetimeVec = true;
      else if (BX->Op == BinOp::Add && isDtVec(BX->LHS) && isDurVec(BX->RHS))
        RhsIsDatetimeVec = true;
      else if (BX->Op == BinOp::Sub &&
               ((isDtVec(BX->LHS) && isDtVec(BX->RHS)) ||
                (isDtVec(BX->LHS) && isDt(BX->RHS))))
        RhsIsDurationVec = true;
      /* Scalar forms (unchanged). */
      else if (BX->Op == BinOp::Sub && isDt(BX->LHS) && isDt(BX->RHS))
        RhsIsDuration = true;
      else if (BX->Op == BinOp::Add &&
               ((isDt(BX->LHS) && isDur(BX->RHS)) ||
                (isDur(BX->LHS) && isDt(BX->RHS))))
        RhsIsDatetime = true;
      else if (BX->Op == BinOp::Sub && isDt(BX->LHS) && isDur(BX->RHS))
        RhsIsDatetime = true;
      else if ((BX->Op == BinOp::Add || BX->Op == BinOp::Sub) &&
               isDur(BX->LHS) && isDur(BX->RHS))
        RhsIsDuration = true;
    }
    /* Phase 4: dict-producing RHS forms — `containers.Map()` /
     * `dictionary(...)`. Tag the LHS so subsequent `m(k)` reads /
     * writes route through the matlab_dict_* runtime. */
    bool RhsIsDict = false;
    if (A.RHS && A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *Cx = static_cast<const CallOrIndex *>(A.RHS);
      if (auto *FA = dynamic_cast<const FieldAccess *>(Cx->Callee))
        if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
          if (BN->Name == "containers" && FA->Field == "Map")
            RhsIsDict = true;
      if (!RhsIsDict)
        if (auto *NE = dynamic_cast<const NameExpr *>(Cx->Callee))
          if (NE->Name == "dictionary")
            RhsIsDict = true;
    }
    bool RhsIsCellLit = A.RHS && A.RHS->Kind == NodeKind::CellLiteral;
    if (!RhsIsCellLit && A.RHS && A.RHS->Kind == NodeKind::MatrixLiteral) {
      auto *MM = static_cast<const MatrixLiteral *>(A.RHS);
      bool All = !MM->Rows.empty();
      for (auto &R : MM->Rows) {
        if (R.empty()) { All = false; break; }
        for (const Expr *X : R)
          if (!isCellExprForAssign(X)) { All = false; break; }
        if (!All) break;
      }
      if (All) RhsIsCellLit = true;
    }
    /* Phase 5.4 (cont.): [TT1 TT2 ...] over timetable bindings -> the
     * matlab_timetable_horzcat chain returns a fresh timetable. */
    if (A.RHS && A.RHS->Kind == NodeKind::MatrixLiteral) {
      auto *MM = static_cast<const MatrixLiteral *>(A.RHS);
      if (MM->Rows.size() == 1 && !MM->Rows[0].empty()) {
        bool AllTT = true;
        for (const Expr *X : MM->Rows[0]) {
          auto *NE = dynamic_cast<const NameExpr *>(X);
          if (!NE || !NE->Ref || !isTimetableBinding(NE->Ref)) {
            AllTT = false; break;
          }
        }
        if (AllTT) RhsIsTimetable = true;
      }
    }
    /* Track string-typed bindings (from "..." literals, string-
     * returning builtins, or `+` chains where either operand is a
     * string) so `+` / disp / strlen / isstring can dispatch
     * correctly. See Lowerer::isStringExpr. */
    bool RhsIsString = isStringExpr(A.RHS);
    /* #233: RHS is a cell-of-strings-returning builtin (strsplit) — tag the
     * LHS so brace reads route to matlab_cell_get_str. */
    bool RhsIsCellOfStr = false;
    if (auto *RC = dynamic_cast<const CallOrIndex *>(A.RHS))
      if (auto *RN = dynamic_cast<const NameExpr *>(RC->Callee))
        if (RN->Ref && RN->Ref->Kind == BindingKind::Builtin &&
            RN->Name == "strsplit")
          RhsIsCellOfStr = true;
    /* A single-quoted char literal assigned wholesale to a variable
     * (`c = 'abc'`) lowers to a bare matlab.const_char, which the REPL/DAP
     * workspace store can't round-trip (matlab_ws_set_mat has no const_char
     * coercion → "unsupported call shape").  In ReplMode treat it like a
     * double-quoted string: tag the binding string + materialize the value to
     * a matlab_string below (kind=3), so the store routes through
     * matlab_ws_set_string and reloads as text.  AOT (ReplMode=false) keeps
     * the const_char lane untouched. */
    bool RhsIsCharLiteralStore =
        ReplMode && A.RHS && A.RHS->Kind == NodeKind::CharLiteral &&
        A.LHS.size() == 1 && A.LHS[0] &&
        dynamic_cast<const NameExpr *>(A.LHS[0]) != nullptr;
    if (RhsIsCharLiteralStore) RhsIsString = true;

    /* Track 3-D bindings: RHS produces a matlab_mat3 — zeros/ones with 3
     * args, cat(3, …), or a builtin that always returns truecolor
     * (colour-space conversions, label2rgb).  These let A(i,j,k) /
     * A(:,:,k) / size(A,3) route to the matlab_mat3 runtime. */
    /* Track whether the RHS produces a 3-D (matlab_mat3) value so the LHS
     * binding is registered in ThreeDBindings and its later subscripts /
     * size / numel / ndims route through the *3 runtime helpers.  Expression-
     * aware (arithmetic, unary, aliasing, the 3-D-producing builtins, and —
     * via funcReturns3D — user-function returns); see Lowerer::exprIsThreeD. */
    bool RhsIsThreeD = exprIsThreeD(A.RHS, ThreeDBindings);

    /* Multi-return call: [V, D] = eig(A). If the LHS arity is > 1 and
     * the RHS is a call to a builtin that has a multi-return variant,
     * emit a matlab.call_builtin with N result types and a nargout
     * attribute so LowerTensorOps can dispatch to the right runtime
     * entry. Each LHS then gets its own result.
     *
     * Phase 1.2: same shape applies to user functions declared with
     * multiple outputs (`function [a, b] = swap(x, y)`). Without this
     * branch the call returned a single none-typed value that got
     * stored into every LHS slot — `[p, q] = swap(10, 20)` gave p == q.
     * varargout-using callees go through the cell-unpack path further
     * down. */
    if (A.LHS.size() > 1 && A.RHS &&
        A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *C = static_cast<const CallOrIndex *>(A.RHS);
      auto *Callee = dynamic_cast<const NameExpr *>(C->Callee);
      bool IsBuiltin = Callee && Callee->Ref &&
                       Callee->Ref->Kind == BindingKind::Builtin;
      bool IsUserFn = Callee && Callee->Ref &&
                      Callee->Ref->Kind == BindingKind::Function &&
                      Callee->Ref->FuncDef;
      bool HasVarargout = IsUserFn && !Callee->Ref->FuncDef->Outputs.empty() &&
                          Callee->Ref->FuncDef->Outputs.back() == "varargout";
      /* #80: multi-return through a function handle — `[a,b] = h(3)` or
       * `[o,r,d] = env.StepFcn(act)`.  A named handle (variable, via
       * HandleTargetRef; or struct field / property, via
       * FieldHandleBindings) resolves to its target function; emit a
       * direct multi-return call (callee `<name>`, nargout = LHS arity)
       * exactly like a syntactic multi-output user-function call.
       * Without this the indirect call defaulted to nargout=1 and every
       * LHS past the first read a duplicated first output. */
      const Function *HandleFn = nullptr;
      std::string HandleFnName;
      if (Callee && Callee->Ref && !IsBuiltin && !IsUserFn) {
        auto HIt = HandleTargetRef.find(Callee->Ref);
        if (HIt != HandleTargetRef.end() && HIt->second && HIt->second->FuncDef) {
          HandleFn = HIt->second->FuncDef;
          HandleFnName = std::string(HIt->second->Name);
        }
      } else if (auto *FA = dynamic_cast<const FieldAccess *>(C->Callee)) {
        if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
          if (BN->Ref) {
            auto HIt = FieldHandleBindings.find({BN->Ref, std::string(FA->Field)});
            if (HIt != FieldHandleBindings.end() && HIt->second &&
                HIt->second->FuncDef) {
              HandleFn = HIt->second->FuncDef;
              HandleFnName = std::string(HIt->second->Name);
            }
          }
      }
      if (HandleFn && !HandleFnName.empty()) {
        size_t DeclOuts = HandleFn->Outputs.size();
        size_t N = std::min(A.LHS.size(),
                            DeclOuts ? DeclOuts : A.LHS.size());
        if (N >= 2) {
          llvm::SmallVector<mlir::Value, 4> Args;
          for (const Expr *Arg : C->Args)
            if (Arg) Args.push_back(lowerExpr(*Arg));
          auto F64 = mlir::Float64Type::get(&MCtx);
          llvm::SmallVector<mlir::Type, 4> Rtys;
          Rtys.reserve(N);
          for (size_t i = 0; i < N; ++i) {
            const Type *OT = (i < HandleFn->OutputRefs.size() &&
                              HandleFn->OutputRefs[i])
                                 ? HandleFn->OutputRefs[i]->InferredType
                                 : nullptr;
            mlir::Type RT0 = OT ? mirTy(OT)
                                : (mlir::Type)mlir::NoneType::get(&MCtx);
            if (mlir::isa<mlir::NoneType>(RT0)) RT0 = F64;
            Rtys.push_back(RT0);
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, HandleFnName));
          mlir::NamedAttribute NO(
              mlir::StringAttr::get(&MCtx, "nargout"),
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(&MCtx, 64), (int64_t)N));
          mlir::Operation *Op = emitUnregOp("matlab.call", Args, Rtys,
                                             loc(A.Range), {Cal, NO});
          for (size_t i = 0; i < N; ++i)
            if (A.LHS[i]) lowerLValueStore(*A.LHS[i], Op->getResult(i));
          return;
        }
      }
      /* Function-style method dispatch for a multi-return call:
       * `[a, b, …] = meth(obj, …)` where `meth` names a method of the first
       * argument's class (e.g. `[A,B,C,D] = ssdata(sys)` -> ss.ssdata).
       * The single-return path already does this; mirror it here so the
       * method is invoked as a user-function multi-return (callee
       * `ClassName__meth`, obj passed as the first parameter). */
      if (IsBuiltin && Callee && !C->Args.empty()) {
        const ClassDef *Cls = nullptr;
        if (auto *AN = dynamic_cast<const NameExpr *>(C->Args[0]))
          if (AN->Ref) Cls = AN->Ref->PinnedClass;
        const Function *MethodFn = nullptr;
        std::string MethodCallee;
        for (const ClassDef *CC = Cls; CC && !MethodFn; CC = CC->Super)
          for (const Function *Mm : CC->Methods)
            if (Mm && Mm->Name == Callee->Name) {
              MethodFn = Mm;
              MethodCallee = std::string(CC->Name) + "__" + std::string(Callee->Name);
              break;
            }
        if (MethodFn && !MethodFn->OutputRefs.empty()) {
          llvm::SmallVector<mlir::Value, 4> Args;
          for (const Expr *Arg : C->Args) if (Arg) Args.push_back(lowerExpr(*Arg));
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          size_t N = std::min(A.LHS.size(), MethodFn->OutputRefs.size());
          llvm::SmallVector<mlir::Type, 4> Rtys;
          for (size_t i = 0; i < N; ++i) {
            const Type *OT = MethodFn->OutputRefs[i]
                                 ? MethodFn->OutputRefs[i]->InferredType : nullptr;
            mlir::Type RT0 = OT ? mirTy(OT) : (mlir::Type)PtrTy;
            if (mlir::isa<mlir::NoneType>(RT0) || mlir::isa<mlir::Float64Type>(RT0))
              RT0 = mlir::isa<mlir::Float64Type>(RT0) ? (mlir::Type)F64 : (mlir::Type)PtrTy;
            Rtys.push_back(RT0);
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, MethodCallee));
          mlir::NamedAttribute NO(
              mlir::StringAttr::get(&MCtx, "nargout"),
              mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 64), (int64_t)N));
          mlir::Operation *Op = emitUnregOp("matlab.call", Args, Rtys,
                                             loc(A.Range), {Cal, NO});
          for (size_t i = 0; i < N; ++i)
            if (A.LHS[i]) lowerLValueStore(*A.LHS[i], Op->getResult(i));
          return;
        }
      }
      /* Model-object CST multi-return splitters. A few CST functions take a
       * model object and return several values; the single-return forms are
       * dispatched in lowerExpr (step_ss / lsim_ss / …). Here we handle the
       * multi-return forms by extracting the object's matrices and calling
       * the per-output runtime entries. */
      if (IsBuiltin && Callee && !C->Args.empty()) {
        const ClassDef *Cls0 = nullptr;
        if (auto *AN = dynamic_cast<const NameExpr *>(C->Args[0]))
          if (AN->Ref) Cls0 = AN->Ref->PinnedClass;
        llvm::StringRef Cn0 = Cls0 ? llvm::StringRef(Cls0->Name) : llvm::StringRef();
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto Lc = loc(A.Range);
        auto loadObjP = [&](const Expr *X) -> mlir::Value {
          mlir::Value V = lowerExpr(*X);
          if (V.getType() != PtrTy) V.setType(PtrTy);
          return V;
        };
        auto getPropP = [&](mlir::Value Obj, llvm::StringRef F) -> mlir::Value {
          mlir::Value FN = emitFieldNameChar(F, Lc);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                   mlir::StringAttr::get(&MCtx, "matlab_obj_get_mat"));
          return emitUnreg("matlab.call_builtin", {Obj, FN}, PtrTy, Lc, {Cal});
        };
        auto boxP = [&](mlir::Value V) -> mlir::Value {
          if (V.getType() == PtrTy) return V;
          if (mlir::isa<mlir::Float64Type>(V.getType())) {
            mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                     mlir::StringAttr::get(&MCtx, "matlab_mat_from_scalar"));
            return emitUnreg("matlab.call_builtin", {V}, PtrTy, Lc, {Cal});
          }
          V.setType(PtrTy);
          return V;
        };
        auto callRT = [&](const char *Fn, llvm::ArrayRef<mlir::Value> Av) -> mlir::Value {
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                   mlir::StringAttr::get(&MCtx, Fn));
          return emitUnreg("matlab.call_builtin", Av, PtrTy, Lc, {Cal});
        };
        /* Curve Fitting Tier-1 — [f, gof] / [f, gof, output] = fit(x,y,'polyN').
         * Alloc a cfit, populate it, then read back the goodness-of-fit and
         * output structs from the populated object (the reader pattern,
         * mirroring the stats [h,p,ci,stats] split — but object-sourced). */
        if (Callee->Name == "fit" && C->Args.size() >= 3) {
          /* Surface fit ([sf,gof] = fit([x y], z, 'polyNM')) → sfit shell. */
          std::string surfTag;
          if (auto *CL = dynamic_cast<const CharLiteral *>(C->Args[2])) surfTag = CL->Value;
          else if (auto *SL = dynamic_cast<const StringLiteral *>(C->Args[2])) surfTag = SL->Value;
          bool isSurf = (surfTag.size() == 6 && surfTag.compare(0, 4, "poly") == 0 &&
                         isdigit(static_cast<unsigned char>(surfTag[4])) &&
                         isdigit(static_cast<unsigned char>(surfTag[5])));
          const char *ctorSym = isSurf ? "sfit__sfit" : "cfit__cfit";
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, ctorSym));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, Lc, {CtorCal});
          mlir::Value Xd = lowerExpr(*C->Args[0]);
          mlir::Value Yd = lowerExpr(*C->Args[1]);
          if (isSurf) {
            mlir::Value Tg = lowerExpr(*C->Args[2]);
            callRT("matlab_curvefit_fit_surface", {Obj, Xd, Yd, Tg});
            mlir::Value Gof2  = callRT("matlab_curvefit_gof", {Obj});
            mlir::Value Outp2 = callRT("matlab_curvefit_output", {Obj});
            mlir::Value SOuts[3] = {Obj, Gof2, Outp2};
            for (size_t i = 0; i < A.LHS.size() && i < 3; ++i)
              if (A.LHS[i]) lowerLValueStore(*A.LHS[i], SOuts[i]);
            return;
          }
          const ClassDef *MdlCls = nullptr;
          if (auto *MN = dynamic_cast<const NameExpr *>(C->Args[2]))
            if (MN->Ref) MdlCls = MN->Ref->PinnedClass;
          if (MdlCls && llvm::StringRef(MdlCls->Name) == "fittype") {
            mlir::Value Ft = loadObjP(C->Args[2]);
            callRT("matlab_curvefit_fit_custom", {Obj, Xd, Yd, Ft});
          } else {
            mlir::Value Md = lowerExpr(*C->Args[2]);
            if (C->Args.size() >= 4) {             /* [f,gof,…] = fit(x,y,model,opts) */
              mlir::Value Op = loadObjP(C->Args[3]);
              callRT("matlab_curvefit_fit_opts", {Obj, Xd, Yd, Md, Op});
            } else {
              callRT("matlab_curvefit_fit", {Obj, Xd, Yd, Md});   /* populate */
            }
          }
          mlir::Value Gof  = callRT("matlab_curvefit_gof", {Obj});
          mlir::Value Outp = callRT("matlab_curvefit_output", {Obj});
          mlir::Value Outs[3] = {Obj, Gof, Outp};
          for (size_t i = 0; i < A.LHS.size() && i < 3; ++i)
            if (A.LHS[i]) lowerLValueStore(*A.LHS[i], Outs[i]);
          return;
        }
        /* [kest, L, P] = kalman(sys, Qn, Rn) — steady-state Kalman filter.
         * L = kalman gain, P = error covariance; kest (the estimator ss
         * object) is returned as the source object as a placeholder. */
        if (Callee->Name == "kalman" && Cn0 == "ss" && C->Args.size() >= 3) {
          mlir::Value Obj = loadObjP(C->Args[0]);
          mlir::Value Av = getPropP(Obj, "A"), Bv = getPropP(Obj, "B"),
                      Cv = getPropP(Obj, "C");
          mlir::Value Qn = boxP(lowerExpr(*C->Args[1]));
          mlir::Value Rn = boxP(lowerExpr(*C->Args[2]));
          mlir::Value Lv = callRT("kalman_L", {Av, Bv, Cv, Qn, Rn});
          mlir::Value Pv = callRT("kalman_P", {Av, Bv, Cv, Qn, Rn});
          mlir::Value Outs[3] = {Obj, Lv, Pv};
          for (size_t i = 0; i < A.LHS.size() && i < 3; ++i)
            if (A.LHS[i]) lowerLValueStore(*A.LHS[i], Outs[i]);
          return;
        }
        /* [Gm, Pm, Wcg, Wcp] = margin(sys) — gain/phase margins and their
         * crossover frequencies. allmargin_ss returns the 1×4 row; split it
         * into the (scalar) outputs. */
        if (Callee->Name == "margin" && (Cn0 == "ss" || Cn0 == "tf")) {
          mlir::Value Obj = loadObjP(C->Args[0]);
          mlir::Value Row;
          if (Cn0 == "tf") {
            Row = callRT("margin_tf_auto",
                         {getPropP(Obj, "Numerator"),
                          getPropP(Obj, "Denominator")});
          } else {
            mlir::Value Av = getPropP(Obj, "A"), Bv = getPropP(Obj, "B"),
                        Cv = getPropP(Obj, "C"), Dv = getPropP(Obj, "D");
            Row = callRT("margin_ss_auto", {Av, Bv, Cv, Dv});
          }
          auto F64 = mlir::Float64Type::get(&MCtx);
          for (size_t i = 0; i < A.LHS.size() && i < 4; ++i) {
            if (!A.LHS[i]) continue;
            mlir::Value Idx = mlir::arith::ConstantOp::create(
                B, Lc, F64, mlir::FloatAttr::get(F64, (double)(i + 1))).getResult();
            mlir::NamedAttribute NI(mlir::StringAttr::get(&MCtx, "nindices"),
                mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 64), 1));
            mlir::Value S = emitUnreg("matlab.subscript", {Row, Idx}, F64, Lc, {NI});
            lowerLValueStore(*A.LHS[i], S);
          }
          return;
        }
        /* [y, tout] = step(model, t): y is the response over the supplied
         * time grid t (ss or tf), tout echoes t. */
        if (Callee->Name == "step" && (Cn0 == "ss" || Cn0 == "tf") &&
            C->Args.size() == 2 && C->Args[1]) {
          mlir::Value Obj = loadObjP(C->Args[0]);
          mlir::Value T = lowerExpr(*C->Args[1]);
          if (T.getType() != PtrTy) T.setType(PtrTy);
          mlir::Value Y = (Cn0 == "tf")
              ? callRT("step_tf_t",
                       {getPropP(Obj, "Numerator"), getPropP(Obj, "Denominator"), T})
              : callRT("step_ss_t",
                       {getPropP(Obj, "A"), getPropP(Obj, "B"),
                        getPropP(Obj, "C"), getPropP(Obj, "D"), T});
          mlir::Value Outs[2] = {Y, T};        /* y, tout = t */
          for (size_t i = 0; i < A.LHS.size() && i < 2; ++i)
            if (A.LHS[i]) lowerLValueStore(*A.LHS[i], Outs[i]);
          return;
        }
        /* [mag, phase, wout] = bode(model, w): magnitude + phase over the
         * frequency grid w (ss or tf), with wout echoing w. Also serves the
         * 2-output [mag, phase] form. */
        if (Callee->Name == "bode" && (Cn0 == "ss" || Cn0 == "tf") &&
            C->Args.size() == 2 && C->Args[1]) {
          mlir::Value Obj = loadObjP(C->Args[0]);
          mlir::Value W = lowerExpr(*C->Args[1]);
          if (W.getType() != PtrTy) W.setType(PtrTy);
          mlir::Value Mag, Phase;
          if (Cn0 == "tf") {
            mlir::Value Num = getPropP(Obj, "Numerator");
            mlir::Value Den = getPropP(Obj, "Denominator");
            Mag   = callRT("bode_tf_mag",   {Num, Den, W});
            Phase = callRT("bode_tf_phase", {Num, Den, W});
          } else {
            mlir::Value Av = getPropP(Obj, "A"), Bv = getPropP(Obj, "B"),
                        Cv = getPropP(Obj, "C"), Dv = getPropP(Obj, "D");
            Mag   = callRT("bode_ss_mag",   {Av, Bv, Cv, Dv, W});
            Phase = callRT("bode_ss_phase", {Av, Bv, Cv, Dv, W});
          }
          mlir::Value Outs[3] = {Mag, Phase, W};   /* mag, phase, wout = w */
          for (size_t i = 0; i < A.LHS.size() && i < 3; ++i)
            if (A.LHS[i]) lowerLValueStore(*A.LHS[i], Outs[i]);
          return;
        }
      }
      if (IsBuiltin) {
        llvm::SmallVector<mlir::Value, 4> Args;
        for (const Expr *Arg : C->Args)
          if (Arg) Args.push_back(lowerExpr(*Arg));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, std::string(Callee->Name)));
        mlir::NamedAttribute NO(
            mlir::StringAttr::get(&MCtx, "nargout"),
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(&MCtx, 64),
                (int64_t)A.LHS.size()));
        llvm::SmallVector<mlir::Type, 4> Rtys(
            A.LHS.size(), mlir::NoneType::get(&MCtx));
        /* Result-type refinement for builtins whose multi-return
         * results have a well-known kind. Keeps the receiving slots
         * from being allocated as matrix (tensor<?xf64>) when the
         * callee actually yields scalar f64s. Same role as the
         * single-return overrides just above the generic emit: we
         * refine when Sema left the arity-split vague. */
        {
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          llvm::StringRef CN = Callee->Name;
          /* [r, c] = size(A) — both f64. */
          if (CN == "size" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), F64);
          /* [h,p,ci,stats] = ttest/ttest2/… — out0/out1 f64 (h/p), out2
           * ci (ptr 1x2), out3 stats (ptr struct). Rank tests use the
           * [p,h,…] order but the slot types are identical. */
          else if ((CN == "ttest" || CN == "ttest2" || CN == "vartest2" ||
                    CN == "ztest" || CN == "kstest" || CN == "ranksum" ||
                    CN == "signrank" || CN == "signtest") && A.LHS.size() >= 2) {
            for (size_t i = 0; i < A.LHS.size(); ++i)
              Rtys[i] = (i < 2) ? mlir::Type(F64) : mlir::Type(PtrTy);
          }
          /* [V, D] = eig / [Q, R] = qr / [L, U] = lu / [U, S, V] = svd /
           * [H, P] = hess — all ptr (matrix) results. */
          else if ((CN == "eig" || CN == "qr" || CN == "lu" ||
                    CN == "svd" || CN == "hess") && A.LHS.size() >= 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [AA, BB, Q, Z] = qz(A, B) — all four ptr (matrix). */
          else if (CN == "qz" && A.LHS.size() == 4)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [coeff,score,latent,…]=pca / [idx,C,sumd,D]=kmeans — all ptr. */
          else if ((CN == "pca" || CN == "kmeans") && A.LHS.size() >= 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [seq,states]=hmmgenerate / [TRANS,EMIS]=hmmtrain — both ptr. */
          else if ((CN == "hmmgenerate" || CN == "hmmtrain") && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [pstates,logpseq]=hmmdecode — ptr + f64. */
          else if (CN == "hmmdecode" && A.LHS.size() == 2) {
            Rtys[0] = PtrTy; Rtys[1] = F64;
          }
          /* [t, y] = ode45(@f, tspan, y0) / ode23 / ode23s — all column
           * matrices. The 3-return form `[t, y, stats]` adds a struct
           * (also ptr). */
          else if ((CN == "ode45" || CN == "ode23" || CN == "ode23s") &&
                   (A.LHS.size() == 2 || A.LHS.size() == 3))
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [t, y, te, ye, ie] = ode_events(@f, tspan, y0, @evt). */
          else if (CN == "ode_events" && A.LHS.size() == 5)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [r, p, k] = residue(b, a) — all ptr (complex column for
           * r and p, real row for k; uniform ptr at the MLIR level). */
          else if (CN == "residue" && A.LHS.size() == 3)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [pks, locs] = findpeaks(x) — both ptr (column matrices). */
          else if (CN == "findpeaks" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [b, a] = butter(n, Wn) / cheby1(n, Rp, Wn) / cheby2(n, Rs, Wn)
           * — both real row vectors (ptr at MLIR level). */
          else if ((CN == "butter" || CN == "cheby1" || CN == "cheby2" ||
                    CN == "besself") &&
                   A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [b, a] = iirnotch(w0, bw) / iirpeak(w0, bw) — both real row
           * vectors (DSP Tier-2 second-order designers). */
          else if ((CN == "iirnotch" || CN == "iirpeak") &&
                   A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [n, Wn] = buttord/cheb1ord(Wp, Ws, Rp, Rs) — both scalar
           * f64 (n is integer-valued but stored as double, matching
           * MATLAB's idiom). */
          else if ((CN == "buttord" || CN == "cheb1ord" ||
                    CN == "cheb2ord") &&
                   A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), F64);
          /* [bd, ad] = bilinear(b, a, fs) — both ptr. */
          else if (CN == "bilinear" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [z, p, k] = tf2zp(b, a) — z, p ptr (complex columns), k scalar. */
          else if (CN == "tf2zp" && A.LHS.size() == 3) {
            Rtys[0] = PtrTy; Rtys[1] = PtrTy; Rtys[2] = F64;
          }
          /* [b, a] = zp2tf(z, p, k) — both ptr. */
          else if (CN == "zp2tf" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [b, a] = sos2tf(sos) — both ptr. */
          else if (CN == "sos2tf" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [H, w] = freqz(b, a, N) — H complex column, w real column;
           * uniform ptr. */
          else if (CN == "freqz" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [row, col] = ind2sub(sz, i) — scalar f64s. */
          else if (CN == "ind2sub" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), F64);
          /* [X, Y] = meshgrid(x, y) / [X, Y, Z] = meshgrid(x, y, z) and
           * the corresponding ndgrid forms — all ptr (matrix) results.
           * Without this, downstream `exp(X)` etc. saw a `none`-typed
           * input, fell back to f64, and an arith.mulf(f64, ptr) op
           * snuck through to LLVM lowering and crashed. */
          else if ((CN == "meshgrid" || CN == "ndgrid") &&
                   (A.LHS.size() == 2 || A.LHS.size() == 3))
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [X, Y, Z] = peaks(N) — three N×N grids on [-3, 3]² with the
           * canonical 3-D demo function. Scalar f64 in, 3 ptr out. */
          else if (CN == "peaks" && A.LHS.size() == 3)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [K, S, e] = lqr(A, B, Q, R) — K is the gain (m × n ptr),
           * S is the Riccati solution (n × n ptr), e is the closed-loop
           * spectrum (n × 1 ptr, possibly complex). Same for dlqr. */
          else if ((CN == "lqr" || CN == "dlqr") &&
                   (A.LHS.size() == 2 || A.LHS.size() == 3))
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [X, K, L] = care(A, B, Q, R) — Riccati X (n × n), gain K
           * (m × n), closed-loop poles L (n × 1, possibly complex).
           * Same shape for dare. The 2-return [X, K] form drops L. */
          else if ((CN == "care" || CN == "dare") &&
                   (A.LHS.size() == 2 || A.LHS.size() == 3))
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [Ar, Br, Cr] = balred(A, B, C, k) — k-state truncated
           * balanced realisation. All three results are matrix ptrs. */
          else if (CN == "balred" && A.LHS.size() == 3)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [L, P] = kalman(A, G, C, Qn, Rn) — gain (n × p ptr) +
           * Riccati covariance (n × n ptr). Same for kalmd. */
          else if ((CN == "kalman" || CN == "kalmd") && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [A, B] = d2c_tustin(Ad, Bd, Ts) — inverse Tustin reverse
           * mapping. Same 2-ptr shape as c2d_tustin. */
          else if (CN == "d2c_tustin" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [Acl, Bcl, Ccl] = feedback_ss(A1, B1, C1, A2, B2, C2) —
           * negative-feedback closed-loop assembly. All three results
           * are matrix ptrs. Same shape for series_ss / parallel_ss. */
          else if ((CN == "feedback_ss" || CN == "series_ss" ||
                    CN == "parallel_ss" || CN == "append_ss") &&
                   A.LHS.size() == 3)
            Rtys.assign(A.LHS.size(), PtrTy);
        }
        mlir::Operation *Op = emitUnregOp("matlab.call_builtin", Args,
                                           Rtys, loc(A.Range), {Cal, NO});
        for (size_t i = 0;
             i < A.LHS.size() && i < (size_t)Op->getNumResults(); ++i) {
          if (!A.LHS[i]) continue;
          /* Slot-type shortcut: when the result is a concrete f64 or
           * ptr (we refined Rtys above) and the LHS is a plain
           * NameExpr, pre-allocate the slot with that type so Sema's
           * possibly-incorrect tensor type doesn't leak through to
           * the alloc. Without this, `[r, c] = size(A)` would get
           * tensor<?xf64> slots that never lower to llvm.alloca of
           * f64, leaving stray matlab.alloc ops in the final IR. */
          auto ResTy = Op->getResult(i).getType();
          if (auto *NE = dynamic_cast<const NameExpr *>(A.LHS[i]);
              NE && NE->Ref &&
              !(ReplMode && InScriptBody &&
                NE->Ref->Kind == BindingKind::Var) &&
              (mlir::isa<mlir::Float64Type>(ResTy) ||
               ResTy == mlir::LLVM::LLVMPointerType::get(&MCtx))) {
            if (Slots.find(NE->Ref) == Slots.end()) {
              mlir::OpBuilder::InsertionGuard G(B);
              auto *InsBlock = B.getInsertionBlock();
              mlir::Operation *P = InsBlock ? InsBlock->getParentOp() : nullptr;
              while (P && !mlir::isa<mlir::func::FuncOp>(P)) {
                auto *PB = P->getBlock();
                P = PB ? PB->getParentOp() : nullptr;
              }
              mlir::Block *Entry = P
                  ? &mlir::cast<mlir::func::FuncOp>(P).getBody().front()
                  : InsBlock;
              B.setInsertionPointToStart(Entry);
              mlir::NamedAttribute NameA(
                  mlir::StringAttr::get(&MCtx, "name"),
                  mlir::StringAttr::get(&MCtx, std::string(NE->Name)));
              mlir::Value Slot = emitUnreg(
                  "matlab.alloc", {}, ResTy, loc(NE->Range), {NameA});
              Slots[NE->Ref] = Slot;
            }
            emitUnregOp("matlab.store",
                        {Op->getResult(i), Slots[NE->Ref]}, {},
                        loc(A.Range));
            continue;
          }
          lowerLValueStore(*A.LHS[i], Op->getResult(i));
        }
        return;
      }
      /* Phase 1.2: user-function multi-return. The declared output
       * arity (Outputs.size()) tells us how many values the function
       * actually returns; emit matlab.call with that many result
       * slots and unpack into each LHS. The varargout case packs
       * everything into a single matlab_cell* and is handled below. */
      if (IsUserFn && !HasVarargout) {
        size_t DeclOuts = Callee->Ref->FuncDef->Outputs.size();
        size_t N = std::min(A.LHS.size(), DeclOuts);
        if (N >= 2) {
          llvm::SmallVector<mlir::Value, 4> Args;
          for (const Expr *Arg : C->Args)
            if (Arg) Args.push_back(lowerExpr(*Arg));
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          /* Result type per declared output, taken from Sema's
           * inferred type for the OutputRef binding when set, else
           * fall back to ptr (the matrix lane is the safer default
           * for unrefined results — f64 scalars are still happy to
           * be stored into a ptr slot via auto-boxing later). */
          llvm::SmallVector<mlir::Type, 4> Rtys;
          Rtys.reserve(N);
          for (size_t i = 0; i < N; ++i) {
            const Type *OT = nullptr;
            if (i < Callee->Ref->FuncDef->OutputRefs.size())
              OT = Callee->Ref->FuncDef->OutputRefs[i]
                      ? Callee->Ref->FuncDef->OutputRefs[i]->InferredType
                      : nullptr;
            mlir::Type RT0 = OT ? mirTy(OT)
                                : (mlir::Type)mlir::NoneType::get(&MCtx);
            if (mlir::isa<mlir::NoneType>(RT0)) RT0 = F64;
            Rtys.push_back(RT0);
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, std::string(Callee->Name)));
          mlir::NamedAttribute NO(
              mlir::StringAttr::get(&MCtx, "nargout"),
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(&MCtx, 64), (int64_t)N));
          mlir::Operation *Op = emitUnregOp("matlab.call", Args, Rtys,
                                             loc(A.Range), {Cal, NO});
          for (size_t i = 0; i < N; ++i) {
            if (!A.LHS[i]) continue;
            auto ResTy = Op->getResult(i).getType();
            if (auto *NE = dynamic_cast<const NameExpr *>(A.LHS[i]);
                NE && NE->Ref &&
                !(ReplMode && InScriptBody &&
                  NE->Ref->Kind == BindingKind::Var) &&
                (mlir::isa<mlir::Float64Type>(ResTy) || ResTy == PtrTy)) {
              if (Slots.find(NE->Ref) == Slots.end()) {
                mlir::OpBuilder::InsertionGuard G(B);
                auto *InsBlock = B.getInsertionBlock();
                mlir::Operation *P = InsBlock ? InsBlock->getParentOp() : nullptr;
                while (P && !mlir::isa<mlir::func::FuncOp>(P)) {
                  auto *PB = P->getBlock();
                  P = PB ? PB->getParentOp() : nullptr;
                }
                mlir::Block *Entry = P
                    ? &mlir::cast<mlir::func::FuncOp>(P).getBody().front()
                    : InsBlock;
                B.setInsertionPointToStart(Entry);
                mlir::NamedAttribute NameA(
                    mlir::StringAttr::get(&MCtx, "name"),
                    mlir::StringAttr::get(&MCtx, std::string(NE->Name)));
                mlir::Value Slot = emitUnreg(
                    "matlab.alloc", {}, ResTy, loc(NE->Range), {NameA});
                Slots[NE->Ref] = Slot;
              }
              emitUnregOp("matlab.store",
                          {Op->getResult(i), Slots[NE->Ref]}, {},
                          loc(A.Range));
              continue;
            }
            lowerLValueStore(*A.LHS[i], Op->getResult(i));
          }
          return;
        }
      }
      /* Phase 1.2: varargout unpack at call site. Function declared as
       *   `function [a, ..., varargout] = f(...)`
       * returns (DeclOuts - 1) declared outputs followed by a
       * matlab_cell* holding the varargout entries. The call site
       * receives DeclOuts result values (the cell at the tail) and
       * unpacks any LHS beyond the declared boundary from the cell
       * via matlab_cell_get_<f64|mat>. */
      if (IsUserFn && HasVarargout) {
        size_t DeclOuts = Callee->Ref->FuncDef->Outputs.size();
        if (DeclOuts >= 1 && A.LHS.size() >= 1) {
          llvm::SmallVector<mlir::Value, 4> Args;
          for (const Expr *Arg : C->Args)
            if (Arg) Args.push_back(lowerExpr(*Arg));
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          /* Result types: one per declared output. Last is ptr (the
           * varargout cell); earlier ones come from Sema's inferred
           * type, falling back to f64 if unrefined. */
          llvm::SmallVector<mlir::Type, 4> Rtys;
          Rtys.reserve(DeclOuts);
          for (size_t i = 0; i + 1 < DeclOuts; ++i) {
            const Type *OT = nullptr;
            if (i < Callee->Ref->FuncDef->OutputRefs.size())
              OT = Callee->Ref->FuncDef->OutputRefs[i]
                      ? Callee->Ref->FuncDef->OutputRefs[i]->InferredType
                      : nullptr;
            mlir::Type RT0 = OT ? mirTy(OT)
                                : (mlir::Type)mlir::NoneType::get(&MCtx);
            if (mlir::isa<mlir::NoneType>(RT0)) RT0 = F64;
            Rtys.push_back(RT0);
          }
          Rtys.push_back(PtrTy); /* varargout cell */
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, std::string(Callee->Name)));
          mlir::NamedAttribute NO(
              mlir::StringAttr::get(&MCtx, "nargout"),
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(&MCtx, 64),
                  (int64_t)A.LHS.size()));
          mlir::Operation *Op = emitUnregOp("matlab.call", Args, Rtys,
                                             loc(A.Range), {Cal, NO});
          mlir::Value VarargoutCell = Op->getResult(DeclOuts - 1);
          /* Store each LHS:
           *   - i in [0, DeclOuts - 1): use Op->getResult(i).
           *   - i in [DeclOuts - 1, A.LHS.size()): unpack from the cell
           *     at index (i - (DeclOuts - 1) + 1). */
          for (size_t i = 0; i < A.LHS.size(); ++i) {
            if (!A.LHS[i]) continue;
            mlir::Value Val;
            if (i + 1 < DeclOuts) {
              Val = Op->getResult(i);
            } else {
              /* Cell unpack: matlab_cell_get_mat returns ptr; for f64
               * cells matlab_cell_get_f64 returns f64. We default to
               * the matrix entry — scalar f64s ride through ptr-typed
               * slots fine because LowerScalarsToArith / disp paths
               * handle the unbox at use time. */
              size_t CellIdx = i - (DeclOuts - 1) + 1;
              mlir::Value Idx = mlir::arith::ConstantOp::create(
                  B, loc(A.Range), F64,
                  mlir::FloatAttr::get(F64, (double)CellIdx));
              mlir::NamedAttribute GetCal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_cell_get_mat"));
              Val = emitUnreg("matlab.call_builtin",
                              {VarargoutCell, Idx}, PtrTy,
                              loc(A.Range), {GetCal});
            }
            lowerLValueStore(*A.LHS[i], Val);
          }
          return;
        }
      }
    }

    mlir::Value Rhs = A.RHS ? lowerExpr(*A.RHS) : mlir::Value{};
    /* Materialize a wholesale char-literal RHS into a matlab_string so the
     * ReplMode workspace store (matlab_ws_set_string) gets the right
     * descriptor — see RhsIsCharLiteralStore above. */
    if (RhsIsCharLiteralStore && Rhs) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
      Rhs = emitUnreg("matlab.call_builtin", {Rhs}, PtrTy, loc(A.Range), {Cal});
    }
    if (RhsIsHandle) {
      /* Pick up capture spill slots left by the AnonFunction lowering
       * (empty vector for @name and capture-free anons). */
      std::vector<mlir::Value> Caps;
      if (A.RHS->Kind == NodeKind::AnonFunction) {
        auto *AF = static_cast<const AnonFunction *>(A.RHS);
        auto It = PendingCaptures.find(AF);
        if (It != PendingCaptures.end()) {
          Caps = std::move(It->second);
          PendingCaptures.erase(It);
        }
      }
      /* Resolve the target binding for a *named* handle RHS: `@inc`
       * directly (FuncHandle->Ref), or a copy of another tracked named
       * handle (`h2 = h` where `h = @inc`).  Used for the field-stored-
       * handle call path (#81) and multi-return handle calls (#80). */
      Binding *HandleTgt = nullptr;
      if (A.RHS->Kind == NodeKind::FuncHandle)
        HandleTgt = static_cast<const FuncHandle *>(A.RHS)->Ref;
      else if (auto *RN = dynamic_cast<const NameExpr *>(A.RHS)) {
        if (RN->Ref) {
          auto It = HandleTargetRef.find(RN->Ref);
          if (It != HandleTargetRef.end()) HandleTgt = It->second;
        }
      }
      for (const Expr *L : A.LHS) {
        if (auto *N = dynamic_cast<const NameExpr *>(L)) {
          if (N->Ref) {
            HandleBindings[N->Ref] = Caps;
            if (HandleTgt) HandleTargetRef[N->Ref] = HandleTgt;
            else HandleTargetRef.erase(N->Ref);
          }
        }
      }
      /* #116: also PERSIST a capture-free anonymous handle to the ReplMode
       * workspace (kind=13) so a LATER REPL turn can recover it. The binding
       * stays on the local-slot lane above for SAME-turn calls (isLocalHandle
       * keeps `f(vec)` / `fminunc(f,..)` lowering correct in this unit, #77),
       * and that lane emits no workspace store — so without this extra store
       * `f = @(x) ..` is lost at end of turn: the next turn reads `f` back as
       * an empty matrix and a solver that invokes it jumps through a bogus
       * pointer (SIGBUS). Named handles already persist via the lowerLValueStore
       * kind=13 path; anon handles were diverted to the slot lane by #77 and
       * dropped their cross-turn store. Only capture-free anons round-trip — the
       * stored value is just the function pointer (matlab_ws_set_handle's ABI,
       * runtime_debug.cpp); the per-session g_ReplEngines vector keeps this
       * turn's JIT'd anon code resident so the pointer stays valid across turns.
       * Captured closures (`@(s) M*s`) need their environment serialized too —
       * a documented follow-up (#116), so they stay matrix-path here. */
      if (ReplMode && InScriptBody && Rhs && Caps.empty() &&
          A.RHS->Kind == NodeKind::AnonFunction) {
        /* #119: classify the anon's return-kind so a cross-turn `f(vec)`
         * with a matrix argument can pick the right matrix trampoline /
         * result type: 1 = matrix, 0 = scalar.  Sema reliably types a
         * non-scalar Array body (a matrix-literal residual `@(x) [..;..]`,
         * a `M*x` product) as matrix; a scalar-arithmetic body over indexed
         * params (`@(x) x(1)+x(2)`) is often left `any` because indexing a
         * param array isn't scalarised — so anything NOT proven matrix is
         * treated as scalar, which matches the dominant objective shape.
         * (A matrix-returning anon whose body Sema can't type — e.g.
         * `@(x) reshape(x,2,2)` — would misdispatch; that residual corner is
         * noted in #119.) */
        int32_t RetKind = 0;
        if (auto *AF = static_cast<const AnonFunction *>(A.RHS)) {
          const Type *BT = AF->Body ? AF->Body->Ty : nullptr;
          if (BT && BT->K == Type::Kind::Array) {
            auto &AT = static_cast<const ArrayType &>(*BT);
            if (AT.S.K != Shape::Rank::Scalar) RetKind = 1;
          }
        }
        for (const Expr *L : A.LHS) {
          auto *N = dynamic_cast<const NameExpr *>(L);
          if (!N || !N->Ref) continue;
          mlir::Value NameV = emitFieldNameChar(N->Ref->Name, loc(A.Range));
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_set_handle"));
          emitUnregOp("matlab.call_builtin", {NameV, Rhs},
                      {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
          /* Record the return-kind side-channel (#119). */
          auto I32 = mlir::IntegerType::get(&MCtx, 32);
          mlir::Value RkV = mlir::arith::ConstantOp::create(
              B, loc(A.Range), I32,
              mlir::IntegerAttr::get(I32, (int64_t)RetKind));
          mlir::NamedAttribute SCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_set_handle_sig"));
          emitUnregOp("matlab.call_builtin", {NameV, RkV},
                      {mlir::NoneType::get(&MCtx)}, loc(A.Range), {SCal});
        }
      }
    }
    /* #81: a named function handle stored into a struct field / classdef
     * property — `s.h = @inc` (FuncHandle RHS) or `h = @inc; s.h = h`
     * (a tracked handle variable).  Resolve the target name and record
     * the (base, field) pair so a later `s.h(args)` lowers as a direct
     * call.  Runs for any RHS shape (RhsIsHandle is false for the
     * NameExpr form, so this can't live in the block above). */
    {
      Binding *FHTgt = nullptr;
      if (A.RHS && A.RHS->Kind == NodeKind::FuncHandle)
        FHTgt = static_cast<const FuncHandle *>(A.RHS)->Ref;
      else if (auto *RN = dynamic_cast<const NameExpr *>(A.RHS)) {
        if (RN->Ref) {
          auto NIt = HandleTargetRef.find(RN->Ref);
          if (NIt != HandleTargetRef.end()) FHTgt = NIt->second;
        }
      }
      for (const Expr *L : A.LHS)
        if (auto *F = dynamic_cast<const FieldAccess *>(L))
          if (auto *BN = dynamic_cast<const NameExpr *>(F->Base))
            if (BN->Ref) {
              auto Key = std::make_pair(BN->Ref, std::string(F->Field));
              if (FHTgt) FieldHandleBindings[Key] = FHTgt;
              else FieldHandleBindings.erase(Key);
            }
    }
    if (RhsIsCellLit) {
      /* Classify each element's storage kind so a later constant-index
       * brace read picks matlab_cell_get_mat for matrix/string slots.
       * Conservative: only flag elements we can prove are stored as a
       * ptr (matrix literal, range, char/string, nested cell, or a
       * known string expr) — a misclassified scalar would wrongly route
       * a scalar slot through get_mat, so we never flag the uncertain
       * cases. Linear (row-major) indexing matches the 1-D storage loop. */
      auto elemIsPtrStored = [&](const Expr *El) -> bool {
        if (!El) return false;
        switch (El->Kind) {
        case NodeKind::MatrixLiteral:
        case NodeKind::RangeExpr:
        case NodeKind::CharLiteral:
        case NodeKind::StringLiteral:
        case NodeKind::CellLiteral:
          return true;
        default:
          return isStringExpr(El);
        }
      };
      /* A string-typed element (char/string literal or a string-valued
       * expr) — subset of the ptr-stored set, tracked so `c{i}` recovers
       * string-ness for disp / assignment propagation (#206). */
      auto elemIsStr = [&](const Expr *El) -> bool {
        if (!El) return false;
        return El->Kind == NodeKind::CharLiteral ||
               El->Kind == NodeKind::StringLiteral || isStringExpr(El);
      };
      std::set<int64_t> MatIdx, StrIdx;
      int64_t elemCount = 0;
      if (auto *CL = dynamic_cast<const CellLiteral *>(A.RHS)) {
        int64_t lin = 1;
        for (const auto &Row : CL->Rows)
          for (const Expr *El : Row) {
            if (elemIsPtrStored(El)) MatIdx.insert(lin);
            if (elemIsStr(El)) StrIdx.insert(lin);
            ++lin;
          }
        elemCount = lin - 1;
      }
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) {
            CellBindings.insert(N->Ref);
            if (!MatIdx.empty()) CellMatElems[N->Ref] = MatIdx;
            else CellMatElems.erase(N->Ref);
            if (!StrIdx.empty()) CellStrElems[N->Ref] = StrIdx;
            else CellStrElems.erase(N->Ref);
            if (elemCount > 0) CellElemCount[N->Ref] = elemCount;
          }
    }
    if (RhsIsDict) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) DictBindings.insert(N->Ref);
    }
    if (RhsIsDatetime) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) DatetimeBindings.insert(N->Ref);
    }
    if (RhsIsDuration) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) DurationBindings.insert(N->Ref);
    }
    if (RhsIsDatetimeVec) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) DatetimeVecBindings.insert(N->Ref);
    }
    if (RhsIsDurationVec) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) DurationVecBindings.insert(N->Ref);
    }
    if (RhsIsCategorical) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) CategoricalBindings.insert(N->Ref);
    }
    if (RhsIsTable) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) TableBindings.insert(N->Ref);
    }
    if (RhsIsVideoWriter) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L)) {
          if (N->Ref) VideoWriterBindings.insert(N->Ref);
          /* #236: in the REPL, record the name in the runtime VideoWriter
           * registry so a later submission's `v.FrameRate = ...` re-stamps
           * the binding (kind=15 hook) and routes to the setter instead of
           * the struct-field path that corrupted the handle. */
          if (ReplMode && InScriptBody && N->Ref &&
              N->Ref->Kind == BindingKind::Var) {
            mlir::Value NameV = emitFieldNameChar(N->Ref->Name, loc(A.Range));
            auto I32 = mlir::IntegerType::get(&MCtx, 32);
            mlir::Value OnV = mlir::arith::ConstantOp::create(
                B, loc(A.Range), I32, mlir::IntegerAttr::get(I32, 1));
            mlir::NamedAttribute MCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_ws_mark_videowriter"));
            emitUnregOp("matlab.call_builtin", {NameV, OnV},
                        {mlir::NoneType::get(&MCtx)}, loc(A.Range), {MCal});
          }
        }
    }
    if (RhsIsTimetable) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) {
            TimetableBindings.insert(N->Ref);
            /* #259: in the REPL, record the name in the runtime timetable
             * registry so a later submission re-stamps the binding
             * (IsTimetable via the kind hook) and `summary(TT)` / `TT.col`
             * route to the timetable path instead of the plain-matrix path
             * (which fails the call-shape detector).  Mirrors the
             * VideoWriter (#236) mark above. */
            if (ReplMode && InScriptBody &&
                N->Ref->Kind == BindingKind::Var) {
              mlir::Value NameV =
                  emitFieldNameChar(N->Ref->Name, loc(A.Range));
              auto I32 = mlir::IntegerType::get(&MCtx, 32);
              mlir::Value OnV = mlir::arith::ConstantOp::create(
                  B, loc(A.Range), I32, mlir::IntegerAttr::get(I32, 1));
              mlir::NamedAttribute MCal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_ws_mark_timetable"));
              emitUnregOp("matlab.call_builtin", {NameV, OnV},
                          {mlir::NoneType::get(&MCtx)}, loc(A.Range), {MCal});
            }
          }
    }
    if (RhsIsTimerange) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) TimerangeBindings.insert(N->Ref);
    }
    if (RhsIsStructArray) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) {
            StructArrayBindings.insert(N->Ref);
            /* fastaread populates each element with char-string Header /
             * Sequence fields; record them as matrix-valued (read via
             * matlab_struct_get_mat) so `s(i).Sequence` returns the
             * matlab_string* — disp/fprintf %s recognise it via the string
             * registry, and bioinfo functions read it as a sequence ptr. */
            MatStructFields.insert({N->Ref, "Header"});
            MatStructFields.insert({N->Ref, "Sequence"});
          }
    }
    if (RhsIsStruct) {
      /* Tag for workspace-setter routing.  We use a separate set
       * from `StructInitialised` because the latter ALSO suppresses
       * the matlab_struct_new auto-init inside ensureStructSlot —
       * we want the same-TU struct lowering to keep doing its init,
       * just have the cross-input workspace-store route via
       * matlab_ws_set_struct rather than _set_mat. */
      /* Decode builtins return a struct with a matrix-valued `Payload` field;
       * record it so `d.Payload` reads via matlab_struct_get_mat instead of
       * defaulting to get_f64 (the Bioinformatics struct-field-typing fix). */
      bool rhsHasPayload = false;
      if (A.RHS && A.RHS->Kind == NodeKind::CallOrIndex)
        if (auto *Cx = dynamic_cast<const CallOrIndex *>(A.RHS))
          if (auto *NE = dynamic_cast<const NameExpr *>(Cx->Callee))
            rhsHasPayload = (NE->Name == "bleLLDataChannelPDUDecode" ||
                             NE->Name == "bleL2CAPFrameDecode");
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) {
            StructBindings.insert(N->Ref);
            if (rhsHasPayload) MatStructFields.insert({N->Ref, "Payload"});
          }
    }
    if (RhsIsSym) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) SymBindings.insert(N->Ref);
    }
    if (RhsIsSymmat) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) SymmatBindings.insert(N->Ref);
    }
    if (RhsIsString) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) StringBindings.insert(N->Ref);
    }
    if (RhsIsCellOfStr) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) CellAllStrBindings.insert(N->Ref);
    }
    if (RhsIsThreeD) {
      for (const Expr *L : A.LHS) {
        if (auto *N = dynamic_cast<const NameExpr *>(L)) {
          if (N->Ref) ThreeDBindings.insert(N->Ref);
        } else if (auto *F = dynamic_cast<const FieldAccess *>(L)) {
          /* #78: `s.T = zeros(3,3,2)` — remember the field is 3-D so
           * later `s.T(i,j,k)=v` / reads route through subscript3. */
          if (auto *BN = dynamic_cast<const NameExpr *>(F->Base))
            if (BN->Ref)
              ThreeDStructFields.insert({BN->Ref, std::string(F->Field)});
        }
      }
    }
    /* #189: 2D row/column deletion — `A(i,:) = []` / `A(:,j) = []`. When the
     * RHS is an empty matrix literal and the LHS is a 2-subscript with a
     * ColonExpr in exactly one position, erase the indexed rows/cols via the
     * runtime helper and store the shrunk result back into the base binding.
     * (matlab_erase_rows / matlab_erase_cols and the shim equivalents already
     * exist; they were simply never wired from lowering.) */
    /* Element deletion by empty-assignment — `A(i,:)=[]` / `A(:,j)=[]` (2D,
     * #189) and `x(idx)=[]` (vector linear, #188). Detect an empty matrix
     * literal RHS with an indexed LHS and route to the erase/delete runtime
     * helper, storing the shrunk result back into the base binding. Scoped to
     * a scalar / `end` index (the dominant case): a vector or range index
     * lowers to a matlab.range / matlab.concat_row operand that this call's
     * operand match doesn't yet resolve here, so those forms fall through
     * untouched (no regression) — tracked as the #188/#189 follow-up. */
    if (A.LHS.size() == 1 && A.LHS[0]) {
      auto *EmptyM = dynamic_cast<const MatrixLiteral *>(A.RHS);
      auto *CI = dynamic_cast<const CallOrIndex *>(A.LHS[0]);
      auto isScalarIdxExpr = [](const Expr *E) -> bool {
        return E && (dynamic_cast<const IntegerLiteral *>(E) ||
                     dynamic_cast<const NameExpr *>(E) ||
                     dynamic_cast<const EndExpr *>(E) ||
                     dynamic_cast<const BinaryOpExpr *>(E) ||
                     dynamic_cast<const UnaryOpExpr *>(E)) &&
               !dynamic_cast<const MatrixLiteral *>(E) &&
               !dynamic_cast<const RangeExpr *>(E);
      };
      if (EmptyM && EmptyM->Rows.empty() && CI && CI->Callee) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto F64 = mlir::Float64Type::get(&MCtx);
        /* Helper: box a scalar f64 index to a 1×1 matrix, run `fn(base, idx)`,
         * and store the shrunk result back into the base binding. */
        auto emitDelete = [&](mlir::Value Base, mlir::Value Idx,
                              const char *fn) {
          if (Base.getType() != PtrTy) Base.setType(PtrTy);
          mlir::NamedAttribute BoxCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mat_from_scalar"));
          Idx = emitUnreg("matlab.call_builtin", {Idx}, PtrTy, loc(A.Range),
                          {BoxCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, fn));
          mlir::Value Res = emitUnreg("matlab.call_builtin", {Base, Idx},
                                      PtrTy, loc(A.Range), {Cal});
          lowerLValueStore(*CI->Callee, Res);
        };
        /* Vector form: x(idx) = []  (#188).  Accept any single non-colon
         * index: a scalar (`x(2)`, `x(end)`, `x(k)`) is boxed to a 1×1
         * matrix; a range (`x(2:3)`), a vector binding (`idx=[2 3]; x(idx)`),
         * or a logical mask (`x(x>3)`) already lowers to a matrix ptr and is
         * passed straight through — matlab_delete_lin resolves a same-shape
         * 0/1 mask to positions, otherwise treats the values as 1-based
         * linear positions. */
        if (CI->Args.size() == 1 && CI->Args[0] &&
            !dynamic_cast<const ColonExpr *>(CI->Args[0])) {
          mlir::Value Base = lowerExpr(*CI->Callee);
          if (Base.getType() != PtrTy) Base.setType(PtrTy);
          SubscriptCtx.push_back({Base, 0}); // dim 0 → numel for `end`
          mlir::Value Idx = lowerExpr(*CI->Args[0]);
          SubscriptCtx.pop_back();
          if (Idx.getType() == F64) {
            emitDelete(Base, Idx, "matlab_delete_lin");
            return;
          }
          /* Matrix-typed index (range / vector / mask): pass directly, no
           * scalar boxing.  A `matlab.range` / matrix-literal index is
           * tensor-typed here — leave it so LowerTensorOps materialises it
           * to a matlab_range / concat call (a ptr); relabelling it to ptr
           * now would make rewriteRange treat it as already lowered and
           * skip it.  Only coerce an untyped (None) value. */
          if (mlir::isa<mlir::NoneType>(Idx.getType())) Idx.setType(PtrTy);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_delete_lin"));
          mlir::Value Res = emitUnreg("matlab.call_builtin", {Base, Idx},
                                      PtrTy, loc(A.Range), {Cal});
          lowerLValueStore(*CI->Callee, Res);
          return;
        }
        /* 2D row/column form: A(i,:) = [] / A(:,j) = []  (#189) */
        if (CI->Args.size() == 2 && CI->Args[0] && CI->Args[1]) {
          bool col0 = dynamic_cast<const ColonExpr *>(CI->Args[0]) != nullptr;
          bool col1 = dynamic_cast<const ColonExpr *>(CI->Args[1]) != nullptr;
          const Expr *IdxE = col1 ? CI->Args[0] : CI->Args[1];
          if (col0 != col1 && isScalarIdxExpr(IdxE)) {
            mlir::Value Base = lowerExpr(*CI->Callee);
            if (Base.getType() != PtrTy) Base.setType(PtrTy);
            int64_t EndDim = col1 ? 1 : 2; // 1=rows, 2=cols (for `end`)
            SubscriptCtx.push_back({Base, EndDim});
            mlir::Value Idx = lowerExpr(*IdxE);
            SubscriptCtx.pop_back();
            if (Idx.getType() == F64) {
              emitDelete(Base, Idx,
                         col1 ? "matlab_erase_rows" : "matlab_erase_cols");
              return;
            }
          }
        }
      }
    }
    /* Phase 3: when the RHS is a value-class binding (`b = a`), clone
     * the underlying matlab_obj before the store so b owns its own
     * fields. Method-call returns are already fresh, so we only clone
     * for NameExpr-shaped RHS. The helper is a no-op for handle
     * classes and non-class RHS. */
    mlir::Value StoreRhs = maybeCloneObjForAssign(Rhs, A.RHS, loc(A.Range));
    for (const Expr *L : A.LHS) if (L) lowerLValueStore(*L, StoreRhs);
    /* #131: persist a struct mutated via field-assignment (`s.x = v`,
     * `s.a.b = v`) to the ReplMode workspace.  The field store writes into
     * s's local struct slot but never round-trips s, so a later REPL turn
     * reads an empty s and `s.x` comes back 0.  Mirror the explicit
     * `s = struct(...)` path (matlab_ws_set_struct, kind=12).  Plain structs
     * only — class-pinned objs / tables / videowriters route through their
     * own setters in lowerLValueStore above and must not be re-stored here. */
    if (ReplMode && InScriptBody) {
      for (const Expr *L : A.LHS) {
        auto *F = dynamic_cast<const FieldAccess *>(L);
        if (!F) continue;
        const Expr *Base = F->Base;
        while (auto *FB = dynamic_cast<const FieldAccess *>(Base))
          Base = FB->Base;   /* walk to the root of an s.a.b chain */
        auto *BN = dynamic_cast<const NameExpr *>(Base);
        if (!BN || !BN->Ref) continue;
        Binding *SB = BN->Ref;
        if (!StructInitialised.count(SB)) continue;   /* plain struct only */
        if (SB->PinnedClass) continue;                /* not a classdef obj */
        if (TableBindings.count(SB) || isVideoWriterBinding(SB)) continue;
        auto SlotIt = Slots.find(SB);
        if (SlotIt == Slots.end()) continue;
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value SVal = emitLoad(SlotIt->second, PtrTy, loc(A.Range));
        mlir::Value NameV = emitFieldNameChar(SB->Name, loc(A.Range));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_ws_set_struct"));
        emitUnregOp("matlab.call_builtin", {NameV, SVal},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
      }
    }
    /* Implicit display: MATLAB prints the result of a statement that
     * doesn't end in a semicolon. We handle the common case of a single
     * named LHS (x = expr). Skip when: the rhs is a handle (we've
     * spilled it; disp would try to matrix-print a function pointer),
     * the rhs's type is NoneType (void call result), or the LHS isn't
     * a single NameExpr.
     *
     * Formatted as two disp calls — "x =" then the value — so it lines
     * up with MATLAB's '%NAME =\n<value>' layout without needing a new
     * runtime entry. */
    if (!A.Suppressed && Rhs && !RhsIsHandle &&
        !mlir::isa<mlir::NoneType>(Rhs.getType()) &&
        A.LHS.size() == 1 && A.LHS[0] &&
        A.LHS[0]->Kind == NodeKind::NameExpr) {
      auto *N = static_cast<const NameExpr *>(A.LHS[0]);
      std::string Label = std::string(N->Name) + " =";
      mlir::NamedAttribute LV(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::StringAttr::get(&MCtx, Label));
      mlir::Value LabelV = emitUnreg("matlab.const_char", {},
                                      mlir::NoneType::get(&MCtx),
                                      loc(A.Range), {LV});
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "disp"));
      emitUnregOp("matlab.call_builtin", {LabelV},
                  {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
      /* The implicit-display path bypasses the explicit disp(...) call
       * dispatcher below (which sees the AST and routes string args to
       * matlab_string_disp). At lowering time we already know whether
       * the RHS is a string binding — if it is, emit the string-disp
       * call directly so the value renders as text instead of being
       * pushed through matlab_disp_mat (which would matrix-print the
       * string descriptor's bytes). */
      llvm::StringRef IntSuf = Lowerer::intDtypeSuffixOf(A.RHS);
      if (RhsIsString) {
        mlir::NamedAttribute SCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_string_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {SCal});
      } else if (RhsIsDatetimeVec) {
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_datetime_vec_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal2});
      } else if (RhsIsDurationVec) {
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_duration_vec_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal2});
      } else if (RhsIsDatetime) {
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_datetime_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal2});
      } else if (RhsIsDuration) {
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_duration_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal2});
      } else if (RhsIsTimetable) {
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_timetable_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal2});
      } else if (RhsIsTable) {
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_table_disp"));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal2});
      } else if (!IntSuf.empty()) {
        /* Phase 1.1.C — typed int matrix disp on `A = int32(...)` style
         * implicit display. Skip the matlab_disp_mat polymorphic path and
         * call the typed disp directly. */
        std::string TyCallee = ("matlab_mat_" + IntSuf + "_disp").str();
        mlir::NamedAttribute TCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, TyCallee));
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {TCal});
      } else {
        emitUnregOp("matlab.call_builtin", {Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
      }
    }
    return;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<const IfStmt &>(St);
    mlir::OpBuilder::InsertionGuard G(B);
    mlir::Value Cond = I.Cond
        ? lowerExpr(*I.Cond)
        : emitUnreg("matlab.const_logical", {},
                    mlir::IntegerType::get(&MCtx, 1), loc(I.Range));
    /* scf.if requires i1; if the cond came back as something else
     * (f64 from a runtime call, integer wider than i1, or `none`
     * from a still-unrefined slot / function param), insert the
     * appropriate truthy comparison. `none` cases get an
     * unrealized_conversion_cast as a verifier placeholder that the
     * RefineIfConds fixup replaces with a real cmpi/cmpf once
     * type-flow propagates. */
    Cond = fixupIfCond(B, Cond, loc(I.Range));

    // scf.if with withElseRegion=true auto-inserts scf.yield terminators; we
    // insert BEFORE those when emitting our body.
    auto IfOp = mlir::scf::IfOp::create(B, loc(I.Range),
                                        /*resultTypes=*/mlir::TypeRange{},
                                        Cond, /*withElseRegion=*/true);
    mlir::Block *ThenB = &IfOp.getThenRegion().front();
    mlir::Block *ElseB = &IfOp.getElseRegion().front();

    B.setInsertionPoint(ThenB->getTerminator());
    if (I.Then) lowerBlock(*I.Then);

    // Chain elseifs into nested scf.ifs in the else region.
    mlir::Block *ElseCursor = ElseB;
    for (auto &EI : I.Elseifs) {
      B.setInsertionPoint(ElseCursor->getTerminator());
      mlir::Value Cond2 = EI.Cond
          ? lowerExpr(*EI.Cond)
          : emitUnreg("matlab.const_logical", {},
                      mlir::IntegerType::get(&MCtx, 1), loc(I.Range));
      Cond2 = fixupIfCond(B, Cond2, loc(I.Range));
      auto Inner = mlir::scf::IfOp::create(
          B, loc(I.Range), mlir::TypeRange{}, Cond2, /*withElseRegion=*/true);
      B.setInsertionPoint(Inner.getThenRegion().front().getTerminator());
      if (EI.Body) lowerBlock(*EI.Body);
      ElseCursor = &Inner.getElseRegion().front();
    }

    B.setInsertionPoint(ElseCursor->getTerminator());
    if (I.Else) lowerBlock(*I.Else);
    return;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<const ForStmt &>(St);
    mlir::Value Iter = F.Iter
        ? lowerExpr(*F.Iter)
        : emitUnreg("matlab.undef", {}, mirTy(TC.any()), loc(F.Range));

    // Loop-var element type: if iter is ranked, take its element type.
    mlir::Type ElemTy = mlir::NoneType::get(&MCtx);
    if (Iter) {
      auto IterTy = Iter.getType();
      if (auto RT = mlir::dyn_cast<mlir::RankedTensorType>(IterTy))
        ElemTy = RT.getElementType();
      else if (auto UT = mlir::dyn_cast<mlir::UnrankedTensorType>(IterTy))
        ElemTy = UT.getElementType();
    }

    // Build a matlab.for with one region, one block with an i-var argument.
    mlir::NamedAttribute VarAttr(
        mlir::StringAttr::get(&MCtx, "var"),
        mlir::StringAttr::get(&MCtx, std::string(F.Var)));
    // Save the outer insertion point *before* createBlock moves it.
    mlir::OpBuilder::InsertionGuard G(B);

    /* If the body has break/continue, pre-allocate the flag slots here
     * so they're visible to LowerSeqLoops as matlab.for's second
     * operand (did_break). We emit them BEFORE the matlab.for op. */
    bool ForHasBC = !F.IsParfor && F.Body &&
                    blockContainsBreakOrContinue(*F.Body);
    mlir::Value BSlotF, CSlotF;
    if (ForHasBC) {
      auto I1 = mlir::IntegerType::get(&MCtx, 1);
      mlir::NamedAttribute NB(mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, "__did_break"));
      BSlotF = emitUnreg("matlab.alloc", {}, I1, loc(F.Range), {NB});
      mlir::NamedAttribute NC(mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, "__did_continue"));
      CSlotF = emitUnreg("matlab.alloc", {}, I1, loc(F.Range), {NC});
      mlir::Value FalseV = mlir::arith::ConstantOp::create(
          B, loc(F.Range), I1, mlir::IntegerAttr::get(I1, 0));
      emitStore(FalseV, BSlotF, loc(F.Range));
      emitStore(FalseV, CSlotF, loc(F.Range));
    }
    llvm::SmallVector<mlir::Value, 2> ForOperands;
    ForOperands.push_back(Iter);
    if (ForHasBC) ForOperands.push_back(BSlotF);
    /* Pick between `matlab.for`, `matlab.parfor`, and `matlab.gpu.kernel`.
     * The GPU kernel form is chosen when a `coder.gpu.kernelfun()` is
     * active on the enclosing function, or when an immediately preceding
     * `coder.gpu.kernel` pragma flagged this loop.  Per-loop flag is
     * one-shot: consumed here so a later for-loop in the same function
     * (without the pragma) goes back to the CPU lane unless kernelfun
     * is on. */
    bool ThisIsGpuKernel = NextForIsGpuKernel || InGpuKernelfun;
    NextForIsGpuKernel = false;  /* one-shot */
    llvm::StringRef ForOpName;
    if (ThisIsGpuKernel) ForOpName = "matlab.gpu.kernel";
    else if (F.IsParfor) ForOpName = "matlab.parfor";
    else ForOpName = "matlab.for";
    mlir::Operation *ForOp = emitUnregOp(
        ForOpName, ForOperands, {}, loc(F.Range), {VarAttr}, /*NumRegions=*/1);

    /* Issue #33 Phase 3b — `% matlab_llvm: write-disjoint(A, j)` escape hatch.
     * For a parfor whose body element-writes a captured matrix through an
     * index the structural disjointness check (OutlineParfor) can't prove
     * injective in the loop variable — e.g. `A(perm(j)) = ...` where `perm`
     * is a permutation — the user asserts the writes don't alias across
     * iterations.  We scan the source lines the loop spans (plus the line
     * just above its header, the conventional pragma placement) for the
     * directive and record each named matrix as a discardable
     * `matlab.write_disjoint` string-array attr on the parfor op.  The
     * outliner reads it and trusts the assertion for those slots.
     *
     * Scanning the source text here — where the Lowerer already holds the
     * SourceManager and the loop's SourceRange — attaches the attr at parfor
     * creation, so it is uniformly visible to every lowering path (AOT, JIT,
     * REPL, -dap) without threading the SourceManager into the shared
     * software-lowering core. */
    if (F.IsParfor && SM && F.Range.Begin.isValid()) {
      matlab::FileID FID = F.Range.Begin.File;
      uint32_t BeginLine = SM->getLineColumn(F.Range.Begin).Line;
      uint32_t EndLine = F.Range.End.isValid()
          ? SM->getLineColumn(F.Range.End).Line : BeginLine;
      uint32_t ScanFrom = BeginLine > 1 ? BeginLine - 1 : 1;
      llvm::SmallVector<mlir::Attribute, 2> Disjoint;
      for (uint32_t Ln = ScanFrom; Ln <= EndLine; ++Ln) {
        std::string_view Line = SM->getLineText(FID, Ln);
        llvm::StringRef LR(Line.data(), Line.size());
        // Locate `% matlab_llvm:` (or `%matlab_llvm:`), tolerating spaces.
        size_t Pct = LR.find('%');
        if (Pct == llvm::StringRef::npos) continue;
        llvm::StringRef Tail = LR.drop_front(Pct + 1).ltrim();
        if (!Tail.consume_front("matlab_llvm:")) continue;
        Tail = Tail.ltrim();
        if (!Tail.consume_front("write-disjoint")) continue;
        Tail = Tail.ltrim();
        if (!Tail.consume_front("(")) continue;
        size_t RP = Tail.find(')');
        if (RP == llvm::StringRef::npos) continue;
        // First comma-separated argument is the matrix variable name.
        llvm::StringRef Args = Tail.take_front(RP);
        llvm::StringRef Name = Args.split(',').first.trim();
        if (Name.empty()) continue;
        auto NameAttr = mlir::StringAttr::get(&MCtx, Name);
        if (!llvm::is_contained(Disjoint, mlir::Attribute(NameAttr)))
          Disjoint.push_back(NameAttr);
      }
      if (!Disjoint.empty())
        ForOp->setAttr("matlab.write_disjoint",
                       mlir::ArrayAttr::get(&MCtx, Disjoint));
    }
    auto &Region = ForOp->getRegion(0);
    mlir::Block *Body = B.createBlock(&Region, Region.end(), {ElemTy}, {loc(F.Range)});

    /* Find Sema binding for F.Var. First-time references to the loop
     * variable happen INSIDE the body (via NameExpr) and would otherwise
     * allocate the slot lazily at read time — too late for our induction-
     * store. Resolve F.VarRef (populated by the Resolver) and
     * pre-allocate the slot before emitting the store. */
    Binding *VarBind = F.VarRef;
    if (!VarBind)
      for (auto &[Bnd, _] : Slots)
        if (Bnd->Name == F.Var) { VarBind = Bnd; break; }

    B.setInsertionPointToEnd(Body);
    if (VarBind) {
      mlir::Value Slot = getOrCreateSlot(VarBind, TC.scalar(Dtype::Double),
                                         VarBind->Name, loc(F.Range));
      B.setInsertionPointToEnd(Body);
      emitStore(Body->getArgument(0), Slot, loc(F.Range));
    }
    if (ForHasBC) {
      auto I1 = mlir::IntegerType::get(&MCtx, 1);
      B.setInsertionPointToEnd(Body);
      LoopStack.push_back({BSlotF, CSlotF});
      lowerLoopBody(*F.Body);
      /* Reset did_continue at the end of each iteration. */
      mlir::Value FalseR = mlir::arith::ConstantOp::create(
          B, loc(F.Range), I1, mlir::IntegerAttr::get(I1, 0));
      emitStore(FalseR, CSlotF, loc(F.Range));
      LoopStack.pop_back();
    } else if (F.Body) {
      lowerBlock(*F.Body);
    }
    emitUnregOp("matlab.yield", {}, {}, loc(F.Range));
    return;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<const WhileStmt &>(St);
    mlir::OpBuilder::InsertionGuard G(B);

    bool HasBC = W.Body && blockContainsBreakOrContinue(*W.Body);
    auto I1 = mlir::IntegerType::get(&MCtx, 1);
    mlir::Value BSlot, CSlot;
    if (HasBC) {
      /* Allocate the flags in the surrounding scope, before the while. */
      mlir::NamedAttribute NB(mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, "__did_break"));
      BSlot = emitUnreg("matlab.alloc", {}, I1, loc(W.Range), {NB});
      mlir::NamedAttribute NC(mlir::StringAttr::get(&MCtx, "name"),
          mlir::FlatSymbolRefAttr::get(&MCtx, "__did_continue"));
      CSlot = emitUnreg("matlab.alloc", {}, I1, loc(W.Range), {NC});
      mlir::Value FalseV = mlir::arith::ConstantOp::create(
          B, loc(W.Range), I1, mlir::IntegerAttr::get(I1, 0));
      emitStore(FalseV, BSlot, loc(W.Range));
      emitStore(FalseV, CSlot, loc(W.Range));
    }

    // matlab.while has two regions: cond (yields i1) and body.
    mlir::Operation *Op = emitUnregOp("matlab.while", {}, {}, loc(W.Range), {},
                                      /*NumRegions=*/2);
    mlir::Block *Cond = B.createBlock(&Op->getRegion(0), Op->getRegion(0).end(),
                                      {}, {});
    mlir::Block *Body = B.createBlock(&Op->getRegion(1), Op->getRegion(1).end(),
                                      {}, {});

    B.setInsertionPointToEnd(Cond);
    mlir::Value C = W.Cond
        ? lowerExpr(*W.Cond)
        : emitUnreg("matlab.const_logical", {},
                    mlir::IntegerType::get(&MCtx, 1), loc(W.Range));
    /* Match the if-stmt path: in DAP/REPL mode the cond may come back
     * as ptr (matlab_mat *) or f64 / wider int. Coerce to i1 before
     * the optional break-fold below combines it with the i1 break flag. */
    C = fixupIfCond(B, C, loc(W.Range));
    if (HasBC) {
      /* cond = orig && !did_break */
      mlir::Value BV = emitLoad(BSlot, I1, loc(W.Range));
      mlir::Value True = mlir::arith::ConstantOp::create(
          B, loc(W.Range), I1, mlir::IntegerAttr::get(I1, 1));
      mlir::Value NotBr = mlir::arith::XOrIOp::create(B, loc(W.Range), BV, True);
      C = mlir::arith::AndIOp::create(B, loc(W.Range), C, NotBr);
    }
    emitUnregOp("matlab.yield", {C}, {}, loc(W.Range));

    B.setInsertionPointToEnd(Body);
    if (HasBC) {
      LoopStack.push_back({BSlot, CSlot});
      if (W.Body) lowerLoopBody(*W.Body);
      /* Reset did_continue for the next iteration. */
      mlir::Value FalseR = mlir::arith::ConstantOp::create(
          B, loc(W.Range), I1, mlir::IntegerAttr::get(I1, 0));
      emitStore(FalseR, CSlot, loc(W.Range));
      LoopStack.pop_back();
    } else if (W.Body) {
      lowerBlock(*W.Body);
    }
    emitUnregOp("matlab.yield", {}, {}, loc(W.Range));
    return;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<const SwitchStmt &>(St);
    mlir::Value Disc = Sw.Discriminant
        ? lowerExpr(*Sw.Discriminant)
        : emitUnreg("matlab.undef", {}, mirTy(TC.any()), loc(Sw.Range));
    // Lower as a chain of nested scf.if:
    //   if (disc == v1) { body1 }
    //   else if (disc == v2) { body2 }
    //   else { otherwise_body }
    // Each subsequent case goes into the ELSE region of the previous if.
    // Without this nesting cases run independently (so case 2 fires even
    // after case 1 matched, and `otherwise` runs unconditionally).
    const ::matlab::Block *OtherwiseBody = nullptr;
    llvm::SmallVector<const ::matlab::SwitchCase *, 8> ValueCases;
    for (auto &C : Sw.Cases) {
      if (!C.Value) OtherwiseBody = C.Body;
      else ValueCases.push_back(&C);
    }
    mlir::OpBuilder::InsertionGuard OuterGuard(B);
    for (auto *C : ValueCases) {
      mlir::Value V = lowerExpr(*C->Value);
      mlir::Value Cond = emitUnreg("matlab.eq", {Disc, V},
                                   mlir::IntegerType::get(&MCtx, 1),
                                   loc(Sw.Range));
      auto IfOp = mlir::scf::IfOp::create(B, loc(Sw.Range), mlir::TypeRange{},
                                          Cond, /*withElseRegion=*/true);
      B.setInsertionPoint(IfOp.getThenRegion().front().getTerminator());
      if (C->Body) lowerBlock(*C->Body);
      // Descend into the else region for any remaining cases / otherwise.
      B.setInsertionPoint(IfOp.getElseRegion().front().getTerminator());
    }
    if (OtherwiseBody) lowerBlock(*OtherwiseBody);
    (void)Disc;
    return;
  }
  case NodeKind::TryStmt: {
    /* try/catch without real stack unwinding: the try body runs
     * normally; after it, we check the runtime error flag. If set, we
     * clear it and run the catch body. The frontend doesn't yet wrap
     * individual try-body statements in error-flag guards, so calls
     * that explicitly error() will only trigger the catch if the
     * error() call is the last thing evaluated before leaving try —
     * good enough for the common 'try; error_if_bad; catch; fallback'
     * idiom. */
    auto &T = static_cast<const TryStmt &>(St);
    if (T.TryBody) lowerBlock(*T.TryBody);
    if (T.CatchBody) {
      mlir::Location L = loc(T.Range);
      auto I32 = mlir::IntegerType::get(&MCtx, 32);
      auto I1 = mlir::IntegerType::get(&MCtx, 1);
      /* matlab_check_error() -> i32 ; !=0 -> i1 */
      mlir::NamedAttribute Chk(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_check_error"));
      mlir::Value Flag = emitUnreg("matlab.call_builtin", {}, I32, L, {Chk});
      mlir::Value Zero = mlir::arith::ConstantOp::create(
          B, L, I32, mlir::IntegerAttr::get(I32, 0));
      mlir::Value Cond = mlir::arith::CmpIOp::create(
          B, L, mlir::arith::CmpIPredicate::ne, Flag, Zero);
      (void)I1;
      auto IfOp = mlir::scf::IfOp::create(B, L, mlir::TypeRange{}, Cond,
                                           /*withElseRegion=*/false);
      mlir::OpBuilder::InsertionGuard G(B);
      B.setInsertionPoint(IfOp.thenBlock()->getTerminator());
      mlir::NamedAttribute Clr(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_clear_error"));
      emitUnregOp("matlab.call_builtin", {},
                  {mlir::NoneType::get(&MCtx)}, L, {Clr});
      if (T.CatchVarRef) CatchBindings.insert(T.CatchVarRef);
      lowerBlock(*T.CatchBody);
      if (T.CatchVarRef) CatchBindings.erase(T.CatchVarRef);
    }
    return;
  }
  case NodeKind::ReturnStmt:
    /* Pop the debug frame before the early return so the DAP frame
     * stack stays balanced with the implicit-return path above. The
     * helper is a no-op when DebugMode is off and when InScriptBody
     * is set (the script frame is owned by matlab_dbg_enable, never
     * pushed by the lowerer). */
    if (!InScriptBody) emitDbgLeaveFrame(loc(St.Range));
    mlir::func::ReturnOp::create(B, loc(St.Range));
    return;
  case NodeKind::BreakStmt:
    if (!LoopStack.empty()) {
      auto I1 = mlir::IntegerType::get(&MCtx, 1);
      mlir::Value True = mlir::arith::ConstantOp::create(
          B, loc(St.Range), I1, mlir::IntegerAttr::get(I1, 1));
      emitStore(True, LoopStack.back().BreakSlot, loc(St.Range));
    } else {
      emitUnregOp("matlab.break", {}, {}, loc(St.Range));
    }
    return;
  case NodeKind::ContinueStmt:
    if (!LoopStack.empty()) {
      auto I1 = mlir::IntegerType::get(&MCtx, 1);
      mlir::Value True = mlir::arith::ConstantOp::create(
          B, loc(St.Range), I1, mlir::IntegerAttr::get(I1, 1));
      emitStore(True, LoopStack.back().ContinueSlot, loc(St.Range));
    } else {
      emitUnregOp("matlab.continue", {}, {}, loc(St.Range));
    }
    return;
  case NodeKind::GlobalDecl:
  case NodeKind::PersistentDecl:
  case NodeKind::ImportStmt:
    /* ID allocation is lazy: the first load/store against a Global or
     * Persistent binding consults GlobalIdByName keyed by the name
     * (globals) or <fnname>.<name> (persistents). See loadBinding and
     * the Global/Persistent handling in lowerLValueStore. */
    return;
  case NodeKind::CommandStmt: {
    auto &C = static_cast<const CommandStmt &>(St);
    /* `clear A B C` maps each named variable's slot to an empty matrix.
     * We resolve each arg to a binding by name inside the current scope
     * (walking SlotMap for an entry with that name). Unmatched args are
     * silently ignored, matching MATLAB's behavior when clearing an
     * undefined name.
     *
     * `clear` with no args clears all variables in the current scope. */
    if (C.Name == "clear") {
      mlir::NamedAttribute NameAttr(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_empty_mat"));
      auto emitClearSlot = [&](mlir::Value Slot) {
        /* Call matlab_empty_mat(), store its result into the slot. We
         * emit as matlab.call_builtin so the tensor-ops pass picks it
         * up and converts it to a real llvm.call in due course. */
        auto PtrT = mlir::NoneType::get(&MCtx);  /* will be retyped */
        mlir::Value Empty = emitUnreg("matlab.call_builtin", {},
                                       PtrT, loc(C.Range), {NameAttr});
        emitStore(Empty, Slot, loc(C.Range));
      };
      if (C.Args.empty()) {
        for (auto &P : Slots) emitClearSlot(P.second);
        /* REPL: also wipe the workspace. */
        if (ReplMode && InScriptBody) {
          mlir::NamedAttribute WCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_clear"));
          emitUnregOp("matlab.call_builtin", {},
                      {mlir::NoneType::get(&MCtx)}, loc(C.Range), {WCal});
        }
      } else {
        for (auto &A : C.Args) {
          for (auto &P : Slots) {
            if (P.first->Name == A) { emitClearSlot(P.second); break; }
          }
          /* REPL: also remove this name from the workspace. */
          if (ReplMode && InScriptBody) {
            mlir::Value NameV = emitFieldNameChar(A, loc(C.Range));
            mlir::NamedAttribute WCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_ws_clear_one"));
            emitUnregOp("matlab.call_builtin", {NameV},
                        {mlir::NoneType::get(&MCtx)}, loc(C.Range), {WCal});
          }
        }
      }
      return;
    }

    /* `cd <dir>` / `cd ..` command syntax — chdir the interpreter process.
     * The parser collects the (possibly space-free) path as a single bare
     * word argument. In the in-process JIT/REPL the chdir persists across
     * turns, so current-folder function resolution follows it. A bare `cd`
     * (no args) is parsed as an expression statement and handled there. */
    if (C.Name == "cd") {
      if (!C.Args.empty()) {
        mlir::Value PathV = emitFieldNameChar(C.Args[0], loc(C.Range));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_cd"));
        emitUnregOp("matlab.call_builtin", {PathV},
                    {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
      } else {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_cd_home"));
        emitUnregOp("matlab.call_builtin", {},
                    {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
      }
      return;
    }

    /* `syms x y z` (Phase 6 — Symbolic Math Toolbox) declares each
     * identifier as a fresh matlab_sym in the current scope. The
     * Resolver already pre-declared the names; here we (a) build a
     * matlab_sym_named for each and (b) bind it via the appropriate
     * write path — workspace setter at REPL/script body, local slot
     * inside a function. SymBindings is populated so subsequent reads
     * route through the sym dispatch. */
    if (C.Name == "syms") {
      auto isIdent = [](const std::string &s) {
        if (s.empty()) return false;
        char c = s.front();
        return c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
      };
      auto PtrT = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::NamedAttribute NamedCal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_sym_named"));
      mlir::NamedAttribute WsCal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_ws_set_sym"));
      auto findBindingByName = [&](std::string_view Nm) -> Binding * {
        /* The script scope binding for each `syms` arg lives in the
         * Resolver's <script> scope — which isn't directly reachable
         * from the lowering. Walk every NameExpr in CurTU's script
         * body and grab the .Ref of the first one matching this name.
         * The Resolver populated NameExpr.Ref before lowering ran,
         * so this is reliable. Skip when there are no functions /
         * script (impossible at this stack frame, but defensive). */
        if (!CurTU || !CurTU->ScriptNode || !CurTU->ScriptNode->Body)
          return nullptr;
        Binding *Found = nullptr;
        std::function<void(const Block &)> walkBlock;
        std::function<void(const Expr &)> walkExpr;
        std::function<void(const Stmt &)> walkStmt;
        walkExpr = [&](const Expr &E) {
          if (Found) return;
          if (auto *N = dynamic_cast<const NameExpr *>(&E)) {
            if (N->Name == Nm && N->Ref) Found = N->Ref;
            return;
          }
          if (auto *Ci = dynamic_cast<const CallOrIndex *>(&E)) {
            if (Ci->Callee) walkExpr(*Ci->Callee);
            for (Expr *A2 : Ci->Args) if (A2) walkExpr(*A2);
            return;
          }
          if (auto *B2 = dynamic_cast<const BinaryOpExpr *>(&E)) {
            if (B2->LHS) walkExpr(*B2->LHS);
            if (B2->RHS) walkExpr(*B2->RHS);
            return;
          }
          if (auto *U = dynamic_cast<const UnaryOpExpr *>(&E)) {
            if (U->Operand) walkExpr(*U->Operand);
            return;
          }
          /* Phase 6.2 — recurse into MatrixLiteral so the AST walk
           * finds NameExprs inside `[u^2 + v^2 - w, ...]` literals.
           * Without this, sym_solve_sys-style array args left the
           * `syms u v w` slot unstored, surfacing as "unsupported op"
           * in the C++ emitter. */
          if (auto *M = dynamic_cast<const MatrixLiteral *>(&E)) {
            for (auto &Row : M->Rows)
              for (Expr *Cx : Row) if (Cx) walkExpr(*Cx);
            return;
          }
        };
        walkStmt = [&](const Stmt &St) {
          if (Found) return;
          if (auto *Es = dynamic_cast<const ExprStmt *>(&St)) {
            if (Es->E) walkExpr(*Es->E);
            return;
          }
          if (auto *As = dynamic_cast<const AssignStmt *>(&St)) {
            for (Expr *Lh : As->LHS) if (Lh) walkExpr(*Lh);
            if (As->RHS) walkExpr(*As->RHS);
            return;
          }
        };
        walkBlock = [&](const Block &Bk) {
          for (Stmt *St : Bk.Stmts) if (St) walkStmt(*St);
        };
        walkBlock(*CurTU->ScriptNode->Body);
        return Found;
      };

      for (auto &Arg : C.Args) {
        if (!isIdent(Arg)) continue;  /* skip 'real', 'positive', etc. */
        mlir::Value NameV = emitFieldNameChar(Arg, loc(C.Range));
        mlir::Value SymV = emitUnreg("matlab.call_builtin", {NameV},
                                       PtrT, loc(C.Range), {NamedCal});
        Binding *Bnd = findBindingByName(Arg);
        if (Bnd) {
          SymBindings.insert(Bnd);
          Bnd->IsSym = true;
          if (!(ReplMode && InScriptBody)) {
            mlir::Value Slot =
                getOrCreateSlot(Bnd, TC.any(), Arg, loc(C.Range));
            emitStore(SymV, Slot, loc(C.Range));
          }
        }
        if (ReplMode && InScriptBody) {
          mlir::Value WName = emitFieldNameChar(Arg, loc(C.Range));
          emitUnregOp("matlab.call_builtin", {WName, SymV},
                      {mlir::NoneType::get(&MCtx)}, loc(C.Range), {WsCal});
        }
      }
      return;
    }

    llvm::SmallVector<mlir::Value, 4> Args;
    for (auto &A : C.Args) {
      mlir::NamedAttribute VA(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::StringAttr::get(&MCtx, A));
      Args.push_back(emitUnreg("matlab.const_str", {},
                               mlir::NoneType::get(&MCtx),
                               loc(C.Range), {VA}));
    }
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, std::string(C.Name)));
    emitUnregOp("matlab.call_builtin", Args,
                {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// LValue store
//===----------------------------------------------------------------------===//

void Lowerer::lowerLValueStore(const Expr &LHS, mlir::Value Rhs) {
  switch (LHS.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(LHS);
    if (!N.Ref) return;
    /* Globals / persistents route through matlab_global_set_f64(id).
     * For persistents we additionally tag the call with the binding's
     * bare name + enclosing function name so the AOT emitters
     * (-emit-c, -emit-cpp, -emit-python, -emit-typescript) can recover
     * a readable identifier and lower to an idiomatic per-language
     * construct (`static double n;`, function-attribute, closure-let)
     * instead of the verbatim runtime call. The LLVM / JIT path
     * ignores the extra attrs, so REPL state-survives-across-
     * invocations semantics are unchanged. */
    if (N.Ref->Kind == BindingKind::Global ||
        N.Ref->Kind == BindingKind::Persistent) {
      if (!Rhs) return;
      int32_t Id = globalSlotId(N.Ref);
      auto I32 = mlir::IntegerType::get(&MCtx, 32);
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::Value IdV = mlir::arith::ConstantOp::create(
          B, loc(N.Range), I32,
          mlir::IntegerAttr::get(I32, (int64_t)Id));
      /* Same UsePtr heuristic as the load path (loadBinding above): a
       * persistent fi-array binding rides the typed-pointer table. */
      bool UsePtr = (N.Ref->Kind == BindingKind::Persistent &&
                     Rhs.getType() == PtrTy);
      llvm::SmallVector<mlir::NamedAttribute, 3> Attrs;
      Attrs.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx,
              UsePtr ? "matlab_persistent_set_ptr" : "matlab_global_set_f64")));
      if (N.Ref->Kind == BindingKind::Persistent) {
        Attrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(&MCtx, "persistent_name"),
            mlir::StringAttr::get(&MCtx, std::string(N.Ref->Name))));
        Attrs.push_back(mlir::NamedAttribute(
            mlir::StringAttr::get(&MCtx, "persistent_fn"),
            mlir::StringAttr::get(&MCtx, CurFnName)));
        // Tag with the binding's user-declared fi spec so the SV
        // backend (`HWStateInfer`) can render the persistent
        // register at the user's `fi(_, signed, WL, FL)` width
        // and signedness. Without this the storage class (i8/i16/
        // i32/i64) and the signless arith integer type's default
        // (signed for multi-bit) would be used, producing too-
        // wide signed regs for `fi(0, 0, 4, 0)`-style declarations.
        // Walk type sources in priority: the NameExpr's own
        // resolved Ty (set by Sema for the LHS) → the Rhs value's
        // type-inference type (when the assignment is `x = fi(...)`
        // the RHS expression's Ty carries the fresh spec) →
        // the binding's InferredType / DeclaredType.
        auto fxFromTy = [](const Type *T) -> const FixedSpec * {
          if (!T || T->K != Type::Kind::Array) return nullptr;
          auto &AT = static_cast<const ArrayType &>(*T);
          if (AT.Elt != Dtype::Fixed || !AT.FxSpec) return nullptr;
          return &(*AT.FxSpec);
        };
        const FixedSpec *Fx = fxFromTy(N.Ty);
        if (!Fx) Fx = fxFromTy(N.Ref->InferredType);
        if (!Fx) Fx = fxFromTy(N.Ref->DeclaredType);
        if (Fx) {
          auto FAttrs = buildFixedAttrs(&MCtx, *Fx);
          for (auto &E0 : FAttrs) Attrs.push_back(E0);
        }
      }
      emitUnregOp("matlab.call_builtin", {IdV, Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(N.Range), Attrs);
      return;
    }
    /* #77: a `x = fi(...)` store carries its FixedSpec only on the
     * assignment LHS type (N.Ty); record the binding so the read side
     * (isFiBinding) keeps it on the local-slot lane too — a workspace
     * round-trip would reload the integer-encoded value as a matrix ptr
     * and a later fi op gets `arith.shrsi(!llvm.ptr, i32)`. */
    if (N.Ref && fixedSpecOf(N.Ty)) FiBindings.insert(N.Ref);
    /* REPL script-level Var writes route through matlab_ws_set_*.
     * Like loadBinding above, skip the workspace path when a local
     * slot already exists (e.g. for-loop induction variable) — the
     * slot is the canonical store site during that loop. */
    if (ReplMode && InScriptBody && N.Ref->Kind == BindingKind::Var && Rhs &&
        Slots.find(N.Ref) == Slots.end() && !isLocalHandle(N.Ref) &&
        !isFiBinding(N.Ref)) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::Value NameV = emitFieldNameChar(N.Name, loc(N.Range));
      bool IsMat = (Rhs.getType() == PtrTy ||
                    mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(Rhs.getType()));
      /* When the Sema-inferred class for this binding is set, the
       * pointer is a matlab_obj* (class instance) rather than a
       * matlab_mat*. Route to matlab_ws_set_obj so the workspace
       * tracks kind=2; the DAP server uses that to render the
       * variable as `1x1 ClassName` and to expose its properties.
       *
       * The IsMat guard is intentionally NOT required: when the RHS
       * is a call to a user-defined factory (e.g. the problem-based
       * `optimvar()` / `optimproblem()` prelude functions) the call
       * result is still `none`-typed at this point — the user-call
       * lowering pass refines it to `!llvm.ptr` later. PinnedClass
       * being set is itself the authoritative signal that the value
       * is a class instance, so route on that alone. */
      bool IsObj = N.Ref->PinnedClass != nullptr;
      /* Strings are LLVM pointer-typed too (matlab_string*), so an
       * unguarded IsMat check would route them to matlab_ws_set_mat
       * — which then aliases matlab_string::data into matlab_mat::data
       * and matlab_string::len into matlab_mat::rows, making the
       * REPL workspace render `text = "Test"` as a 4 x <heap-garbage>
       * double matrix and crash any inspector that dereferences the
       * cooked-up shape. Honour the AssignStmt's StringBindings
       * tracking (and StringArrayType-typed LHSs that arrive here
       * from elsewhere) and route to matlab_ws_set_string so the
       * workspace records kind=3. */
      bool IsString = false;
      if (StringBindings.count(N.Ref)) IsString = true;
      else if (N.Ref->InferredType &&
               N.Ref->InferredType->K == Type::Kind::StringArray)
        IsString = true;
      else if (N.Ty && N.Ty->K == Type::Kind::StringArray)
        IsString = true;
      bool IsSym = SymBindings.count(N.Ref) != 0;
      bool IsSymmat = SymmatBindings.count(N.Ref) != 0;
      /* Phase 5 heterogeneous types — route through their dedicated
       * workspace setters so the DAP Workspace pane renders the row
       * with the right type tag (`table` / `categorical` / `datetime`
       * / `duration`) and the variable-children drill-in walks the
       * native layout instead of casting the pointer to matlab_mat*
       * and reading garbage. The binding sets are populated by the
       * AssignStmt RhsIs* tagging block above (~:2579) — by the time
       * we get here, a `T = readtable(...)` LHS is in TableBindings,
       * `c = categorical(...)` in CategoricalBindings, etc. */
      bool IsTable = isTableBinding(N.Ref) != 0;
      bool IsCategorical = CategoricalBindings.count(N.Ref) != 0;
      bool IsDatetime = DatetimeBindings.count(N.Ref) != 0;
      bool IsDuration = DurationBindings.count(N.Ref) != 0;
      bool IsStruct   = (N.Ref &&
                         (StructBindings.count(N.Ref) || N.Ref->IsStruct));
      /* #258: a struct-array-returning assignment (`s = fastaread(...)`)
       * must persist via matlab_ws_set_struct_arr (kind=14) so a later turn's
       * `s(i).Field` / `length(s)` rehydrates the array (and its elements)
       * instead of reading a generic matrix.  Without this it fell to set_mat
       * and `s(1).Header` came back undef cross-turn. */
      bool IsStructArr = N.Ref && isStructArrayBinding(N.Ref);
      /* Function handle: `f = @sin` / `f = @myFn` / capture-free anon.
       * HandleBindings was populated by the AssignStmt handle-tracking
       * block just above this store.  Only capture-free handles (empty
       * spill list) survive a workspace round-trip — the stored value is
       * a bare function pointer with no closure state — so a captured
       * anon stays on the matrix path (its same-turn slot still works;
       * the cross-turn case is a documented follow-up). */
      bool IsHandle = false;
      if (N.Ref) {
        auto HIt = HandleBindings.find(N.Ref);
        if (HIt != HandleBindings.end() && HIt->second.empty())
          IsHandle = true;
      }
      llvm::StringRef Callee =
          IsHandle       ? "matlab_ws_set_handle"
          : (IsSymmat    ? "matlab_ws_set_symmat"
                         : (IsSym ? "matlab_ws_set_sym"
                              : (IsString ? "matlab_ws_set_string"
                              : (IsObj    ? "matlab_ws_set_obj"
                              : (IsStructArr ? "matlab_ws_set_struct_arr"
                              : (IsStruct ? "matlab_ws_set_struct"
                              : (IsTable  ? "matlab_ws_set_table"
                              : (IsCategorical ? "matlab_ws_set_categorical"
                              : (IsDatetime    ? "matlab_ws_set_datetime"
                              : (IsDuration    ? "matlab_ws_set_duration"
                              : (IsMat ? "matlab_ws_set_mat"
                                       : "matlab_ws_set_f64")))))))))));
      /* #77: remember whether this workspace var currently holds a matrix
       * so an anon capturing it can reload it as a ptr (matlab_ws_get_mat)
       * rather than mis-typing the capture as f64. */
      if (N.Ref) {
        if (Callee == "matlab_ws_set_mat") MatrixWsBindings.insert(N.Ref);
        else if (Callee == "matlab_ws_set_f64") MatrixWsBindings.erase(N.Ref);
      }
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      emitUnregOp("matlab.call_builtin", {NameV, Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(N.Range), {Cal});
      return;
    }
    const Type *T = LHS.Ty ? LHS.Ty : TC.any();
    mlir::Value Slot = getOrCreateSlot(N.Ref, T, N.Name, loc(N.Range));
    if (Rhs) emitStore(Rhs, Slot, loc(N.Range));
    return;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(LHS);
    /* PDE Toolbox (#28): `model.FaceBC(ids) = faceBC(Constraint="fixed")`
     * and `model.FaceLoad(ids) = faceLoad(Pressure=p)` — wire the
     * indexed-property assignment into the solver's flat FixedFaces /
     * PressureFaces tables (the generic __subscript_store below would
     * stash the BC object in an unread property array, leaving the solve
     * unconstrained / unloaded).  v1 enumerates literal scalar / range
     * indices and treats faceBC as a fixed constraint; the pressure
     * value is read off the faceLoad object at runtime. */
    if (auto *FA = dynamic_cast<const FieldAccess *>(C.Callee))
      if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
        if (BN->Ref && BN->Ref->PinnedClass &&
            BN->Ref->PinnedClass->Name == "femodel" && C.Args.size() == 1 &&
            (FA->Field == "FaceBC" || FA->Field == "FaceLoad") && Rhs) {
          /* Enumerate literal scalar / a:b range face ids. */
          llvm::SmallVector<int64_t, 8> Ids;
          const Expr *Ix = C.Args[0];
          if (auto *RE = dynamic_cast<const RangeExpr *>(Ix)) {
            if (RE->Start && RE->End && !RE->Step) {
              int64_t a = foldInt(RE->Start), b = foldInt(RE->End);
              for (int64_t v = a; v <= b; ++v) Ids.push_back(v);
            }
          } else if (Ix) {
            Ids.push_back(foldInt(Ix));
          }
          if (!Ids.empty()) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            auto F64 = mlir::Float64Type::get(&MCtx);
            bool IsLoad = (FA->Field == "FaceLoad");
            mlir::Value Pressure;
            if (IsLoad) {
              mlir::Value NameV = emitFieldNameChar("Pressure", loc(C.Range));
              mlir::NamedAttribute GCal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_obj_get_f64"));
              Pressure = emitUnreg("matlab.call_builtin", {Rhs, NameV}, F64,
                                   loc(C.Range), {GCal});
            }
            for (int64_t Id : Ids) {
              mlir::Value Model = lowerExpr(*FA->Base);
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::FloatAttr::get(F64, (double)Id));
              mlir::Value Fid = emitUnreg("matlab.const_float", {}, F64,
                                          loc(C.Range), {VA});
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(
                      &MCtx, IsLoad ? "pde_set_face_pressure"
                                    : "pde_set_face_fixed"));
              if (IsLoad)
                emitUnreg("matlab.call_builtin", {Model, Fid, Pressure}, PtrTy,
                          loc(C.Range), {Cal});
              else
                emitUnreg("matlab.call_builtin", {Model, Fid}, PtrTy,
                          loc(C.Range), {Cal});
            }
            return;
          }
        }
    /* Phase 4: m(k) = rhs on a dict binding. Detect via DictBindings,
     * dispatch to matlab_dict_set_<str|num>_<f64|mat>. Single key
     * arg only for v1 (multi-dim keying isn't a MATLAB idiom).
     * CharLiteral / StringLiteral keys are coerced to matlab_string*
     * via matlab_string_from_literal before the call. */
    if (C.Args.size() == 1 && C.Args[0]) {
      if (auto *N = dynamic_cast<const NameExpr *>(C.Callee))
        if (N->Ref && DictBindings.count(N->Ref)) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          mlir::Value D = lowerExpr(*C.Callee);
          const Expr *KeyExpr = C.Args[0];
          mlir::Value K;
          bool KeyIsStr = false;
          if (auto *CL = dynamic_cast<const CharLiteral *>(KeyExpr)) {
            mlir::NamedAttribute VA(
                mlir::StringAttr::get(&MCtx, "value"),
                mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
            mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                        mlir::NoneType::get(&MCtx),
                                        loc(C.Range), {VA});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
            K = emitUnreg("matlab.call_builtin", {Ch}, PtrTy,
                          loc(C.Range), {Cal});
            KeyIsStr = true;
          } else {
            K = lowerExpr(*KeyExpr);
            KeyIsStr = K && (K.getType() == PtrTy || isStringExpr(KeyExpr));
          }
          bool ValIsMat = Rhs && (Rhs.getType() == PtrTy ||
                                  mlir::isa<mlir::RankedTensorType,
                                            mlir::UnrankedTensorType>(Rhs.getType()));
          std::string Callee = "matlab_dict_set_";
          Callee += KeyIsStr ? "str_" : "num_";
          Callee += ValIsMat ? "mat" : "f64";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          emitUnregOp("matlab.call_builtin", {D, K, Rhs},
                      {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
          return;
        }
    }
    /* `name(:) = rhs` for an fi-typed scalar `name` is the type-preserving
     * idiom that holds `name`'s FixedSpec across iterations of an
     * accumulator loop (see plan §11). At Sema time we already kept the
     * lhs's type; here we emit a `matlab.fi.cast` to clamp the rhs to
     * the lhs's spec, then store into lhs's slot directly. */
    if (Rhs && C.Args.size() == 1 && C.Args[0] &&
        C.Args[0]->Kind == NodeKind::ColonExpr) {
      auto *N = dynamic_cast<const NameExpr *>(C.Callee);
      if (N && N->Ref && N->Ty && N->Ty->K == Type::Kind::Array) {
        auto &LA = static_cast<const ArrayType &>(*N->Ty);
        if (LA.Elt == Dtype::Fixed && LA.FxSpec) {
          mlir::Type Stor = mirTy(N->Ty);
          mlir::Value RhsVal = Rhs;
          // Tag the cast with both source and destination specs so
          // LowerFixedPoint can emit the shift/sat sequence. The source
          // spec is read off the RHS's defining op (which the binop
          // emission tags with fi_signed/fi_wl/fi_fl) — that's the
          // closest match for the rhs's runtime type.
          auto Attrs = buildFixedAttrs(&MCtx, *LA.FxSpec);
          mlir::NamedAttribute IsClamp(
              mlir::StringAttr::get(&MCtx, "fi_clamp"),
              mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 1), 1));
          llvm::SmallVector<mlir::NamedAttribute, 12> A;
          A.push_back(IsClamp);
          for (auto &E0 : Attrs) A.push_back(E0);
          if (auto *DefOp = RhsVal.getDefiningOp()) {
            auto carry = [&](llvm::StringRef From, llvm::StringRef To) {
              if (auto Atr = DefOp->getAttr(From))
                A.emplace_back(mlir::StringAttr::get(&MCtx, To), Atr);
            };
            carry("fi_signed", "fi_lhs_signed");
            carry("fi_wl",     "fi_lhs_wl");
            carry("fi_fl",     "fi_lhs_fl");
          }
          mlir::Value Cast = emitUnreg("matlab.fi.cast", {RhsVal}, Stor,
                                        loc(C.Range), A);
          mlir::Value Slot =
              getOrCreateSlot(N->Ref, N->Ty, N->Name, loc(N->Range));
          emitStore(Cast, Slot, loc(C.Range));
          return;
        }
      }
    }
    llvm::SmallVector<mlir::Value, 4> Os;
    mlir::Value Base;
    if (C.Callee) {
      Base = lowerExpr(*C.Callee);
      Os.push_back(Base);
    }
    // Push subscript context so any `end` inside an index expression
    // resolves to size(Base, dim).
    for (size_t a = 0; a < C.Args.size(); ++a) {
      const Expr *Arg = C.Args[a];
      if (!Arg) continue;
      // Single-subscript indexing: `end` means numel(Base), not size(,1).
      // Use sentinel dim 0 → matlab_end_of_dim treats it as numel.
      int64_t EndDim = (C.Args.size() == 1) ? 0 : (int64_t)(a + 1);
      if (Base) SubscriptCtx.push_back({Base, EndDim});
      Os.push_back(lowerExpr(*Arg));
      if (Base) SubscriptCtx.pop_back();
    }
    if (Rhs) Os.push_back(Rhs);
    /* 3-D store on a matlab_mat3 binding: A(i,j,k)=v (scalar element) →
     * matlab_subscript3_store; A(:,:,k)=v|M (whole plane) →
     * matlab_subscript3_pstore_{s,m}. */
    if (C.Args.size() == 3 && Rhs) {
      /* 3-D base may be a plain variable (ThreeDBindings) or a struct
       * field / classdef property (ThreeDStructFields, #78).  Os[0] is
       * the already-lowered base — a mat3 ptr in both cases — and
       * matlab_subscript3_store mutates it in place, so the field path
       * needs no write-back. */
      bool Is3D = false;
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        Is3D = isThreeDBinding(NE->Ref);
      else if (auto *F = dynamic_cast<const FieldAccess *>(C.Callee))
        if (auto *BN = dynamic_cast<const NameExpr *>(F->Base))
          Is3D = BN->Ref &&
                 ThreeDStructFields.count({BN->Ref, std::string(F->Field)});
      if (Is3D) {
          bool c0 = dynamic_cast<const ColonExpr *>(C.Args[0]) != nullptr;
          bool c1 = dynamic_cast<const ColonExpr *>(C.Args[1]) != nullptr;
          bool c2 = dynamic_cast<const ColonExpr *>(C.Args[2]) != nullptr;
          if (c0 && c1 && !c2) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            bool rhsMat = (Rhs.getType() == PtrTy ||
                           mlir::isa<mlir::RankedTensorType,
                                     mlir::UnrankedTensorType>(Rhs.getType()));
            mlir::NamedAttribute Cal3(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, rhsMat ? "matlab_subscript3_pstore_m"
                                                    : "matlab_subscript3_pstore_s"));
            emitUnregOp("matlab.call_builtin", {Os[0], Os[3], Os[4]},
                        {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal3});
            return;
          }
          if (!c0 && !c1 && !c2) {
            mlir::NamedAttribute Cal3(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_subscript3_store"));
            emitUnregOp("matlab.call_builtin", Os,
                        {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal3});
            return;
          }
        }
    }
    /* Rank-4 scalar store: A(i,j,k,l) = v on any binding.  Routes through
     * matlab_subscript4_pstore_s, which is N-D-aware (falls back to lower-
     * rank descriptors via mat_is_3d / mat_is_nd). */
    if (C.Args.size() == 4 && Rhs) {
      bool anyColon = false;
      for (size_t a = 0; a < 4; ++a)
        if (dynamic_cast<const ColonExpr *>(C.Args[a])) { anyColon = true; break; }
      if (!anyColon) {
        mlir::NamedAttribute Cal4(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_subscript4_pstore_s"));
        emitUnregOp("matlab.call_builtin",
                    {Os[0], Os[1], Os[2], Os[3], Os[4], Os[5]},
                    {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal4});
        return;
      }
    }
    /* Rank>=5 scalar store: A(i,j,k,l,m[,...]) = v.  Variadic — routes
     * through matlab_subscriptN_pstore_s (runtime generic to 16 dims).  The
     * index-packing into an int64_t[] happens in LowerTensorOps once types
     * settle to ptr/f64.  #93. */
    if (C.Args.size() >= 5 && Rhs) {
      bool anyColon = false;
      for (size_t a = 0; a < C.Args.size(); ++a)
        if (dynamic_cast<const ColonExpr *>(C.Args[a])) { anyColon = true; break; }
      if (!anyColon) {
        mlir::NamedAttribute CalN(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_subscriptN_pstore_s"));
        emitUnregOp("matlab.call_builtin", Os,
                    {mlir::NoneType::get(&MCtx)}, loc(C.Range), {CalN});
        return;
      }
    }
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "__subscript_store"));
    emitUnregOp("matlab.call_builtin", Os,
                {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
    return;
  }
  case NodeKind::CellIndex: {
    /* C{i} = Rhs (1-D) routes to matlab_cell_set_<f64|mat>.
     * C{r, k} = Rhs (2-D, Phase 1.3) routes to matlab_cell_set_<f64|mat>_2d.
     * Kind is picked from Rhs's MLIR type — ptr / tensor -> _mat,
     * everything else -> _f64. */
    auto &C = static_cast<const CellIndex &>(LHS);
    if (C.Args.empty() || C.Args.size() > 2 || !C.Callee) return;
    mlir::Value Cell = lowerExpr(*C.Callee);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    bool IsMat = Rhs && (Rhs.getType() == PtrTy ||
                         mlir::isa<mlir::RankedTensorType,
                                   mlir::UnrankedTensorType>(Rhs.getType()));
    if (C.Args.size() == 1) {
      mlir::Value Idx = lowerExpr(*C.Args[0]);
      llvm::StringRef Callee = IsMat ? "matlab_cell_set_mat"
                                      : "matlab_cell_set_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      emitUnregOp("matlab.call_builtin", {Cell, Idx, Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
      return;
    }
    mlir::Value R = lowerExpr(*C.Args[0]);
    mlir::Value K = lowerExpr(*C.Args[1]);
    llvm::StringRef Callee = IsMat ? "matlab_cell_set_mat_2d"
                                    : "matlab_cell_set_f64_2d";
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, Callee));
    emitUnregOp("matlab.call_builtin", {Cell, R, K, Rhs},
                {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
    return;
  }
  case NodeKind::FieldAccess: {
    /* s.x = Rhs  OR  s.a.b = Rhs. For the nested case the base is
     * itself a FieldAccess; resolveStructBase walks the chain,
     * auto-allocating intermediate struct fields via
     * matlab_struct_get_child_struct so 's.a.b = v' works even when
     * s.a didn't exist yet.
     *
     * If the base is a class-pinned variable, route to matlab_obj_set_*
     * instead so class_id + property table is preserved.
     *
     * Phase 2: s(i).x = Rhs — Base is `CallOrIndex(NameExpr s, [i])`.
     * Auto-promote `s` to a struct array; route to
     * matlab_struct_arr_get_or_create + matlab_struct_set_*. */
    auto &F = static_cast<const FieldAccess &>(LHS);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    /* Char-string field/property store (#79.2): `s.name = 'hello'`.
     * A char literal lowers to a `matlab.const_char` (i8 tensor); the
     * generic field-store paths below would mistake the tensor for a
     * matrix payload (`matlab_*_set_mat`) and leave the i8 const_char
     * unconverted.  We detect it here but DEFER the wrap to the generic
     * classdef-property / plain-struct paths only — the special-case
     * intercepts in between (TableBindings, timetable
     * `.Properties.Description`, struct arrays) consume the raw
     * const_char Rhs directly, so wrapping it up front would break their
     * call shapes.  `maybeWrapCharStr()` does the deferred wrap (through
     * `matlab_string_from_literal` -> a `matlab_string *`, kind=3); the
     * string read side (`matlab_*_get_mat`) is already kind=3 aware. */
    bool RhsIsCharStr = false;
    if (Rhs)
      if (mlir::Operation *RD = Rhs.getDefiningOp())
        if (RD->getName().getStringRef() == "matlab.const_char")
          RhsIsCharStr = true;
    auto maybeWrapCharStr = [&]() {
      if (!RhsIsCharStr) return;
      mlir::NamedAttribute SCal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
      Rhs = emitUnreg("matlab.call_builtin", {Rhs}, PtrTy,
                      loc(F.Range), {SCal});
    };
    /* v.FrameRate = fps / v.Quality = q on a VideoWriter handle. Route to
     * the dedicated setters so the opaque handle isn't misread as a struct.
     * The setters take a scalar f64, so this fires for scalar (f64 / int)
     * RHS; a non-scalar RHS or an unknown property is silently ignored
     * (property-set is a v1 subset — see docs/plotting.md §4). */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (isVideoWriterBinding(BN->Ref)) {
        const char *Setter = nullptr;
        if (F.Field == "FrameRate") Setter = "matlab_videowriter_set_framerate";
        else if (F.Field == "Quality") Setter = "matlab_videowriter_set_quality";
        bool ScalarRhs = Rhs && (Rhs.getType() == mlir::Float64Type::get(&MCtx) ||
                                 mlir::isa<mlir::IntegerType>(Rhs.getType()));
        if (Setter && ScalarRhs) {
          mlir::Value Vv = lowerExpr(*F.Base);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Setter));
          emitUnregOp("matlab.call_builtin", {Vv, Rhs},
                      {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
        }
        return;
      }
    /* Phase 5.3: T.<name> = Rhs — Base is a NameExpr in TableBindings.
     * Route to matlab_table_add_column (which auto-creates the column
     * on first write or replaces an existing one). */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && isTableBinding(BN->Ref)) {
        mlir::Value Tv = lowerExpr(*F.Base);
        mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_table_add_column"));
        emitUnregOp("matlab.call_builtin", {Tv, NameV, Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
        return;
      }
    /* Phase 5.4 (cont.): TT.<colName> = Rhs on a timetable binding.
     * Same shape as table column write — _add_column auto-replaces. */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && isTimetableBinding(BN->Ref)) {
        mlir::Value Tv = lowerExpr(*F.Base);
        mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_timetable_add_column"));
        emitUnregOp("matlab.call_builtin", {Tv, NameV, Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
        return;
      }
    /* TT.Properties.Description = 'literal' — nested FieldAccess.
     * Match exactly that two-level shape; defer other Properties
     * write targets (VariableNames rename, etc.) to later tasks. */
    if (auto *Inner = dynamic_cast<const FieldAccess *>(F.Base))
      if (auto *BN = dynamic_cast<const NameExpr *>(Inner->Base))
        if (BN->Ref && isTimetableBinding(BN->Ref) &&
            Inner->Field == "Properties" && F.Field == "Description") {
          mlir::Value Tv = lowerExpr(*Inner->Base);
          mlir::NamedAttribute VA(
              mlir::StringAttr::get(&MCtx, "value"),
              mlir::StringAttr::get(&MCtx, ""));
          /* Rhs may be a char literal (matlab.const_char) or a
           * string descriptor. We accept the const_char form
           * directly — the runtime entry takes (ptr, char*, i64). */
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_set_description"));
          emitUnregOp("matlab.call_builtin", {Tv, Rhs},
                      {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
          return;
        }
    if (auto *CI = dynamic_cast<const CallOrIndex *>(F.Base)) {
      auto *NE = dynamic_cast<const NameExpr *>(CI->Callee);
      if (NE && NE->Ref &&
          NE->Ref->Kind == BindingKind::Var &&
          !CellBindings.count(NE->Ref) &&
          CI->Args.size() == 1 && CI->Args[0]) {
        StructArrayBindings.insert(NE->Ref);
        mlir::Value Slot = ensureStructArraySlot(NE->Ref, NE->Name,
                                                  loc(F.Range));
        mlir::Value Arr = emitLoad(Slot, PtrTy, loc(F.Range));
        mlir::Value Idx = lowerExpr(*CI->Args[0]);
        mlir::NamedAttribute GCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_struct_arr_get_or_create"));
        mlir::Value Elem = emitUnreg("matlab.call_builtin", {Arr, Idx},
                                      PtrTy, loc(F.Range), {GCal});
        mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
        bool IsMat = Rhs && (Rhs.getType() == PtrTy ||
                             mlir::isa<mlir::RankedTensorType,
                                       mlir::UnrankedTensorType>(Rhs.getType()));
        llvm::StringRef Callee = IsMat ? "matlab_struct_set_mat"
                                        : "matlab_struct_set_f64";
        mlir::NamedAttribute SCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        emitUnregOp("matlab.call_builtin", {Elem, NameV, Rhs},
                    {mlir::NoneType::get(&MCtx)}, loc(F.Range), {SCal});
        /* #133: persist the struct array to the workspace so a later REPL
         * turn sees the element/field just written. The store above only
         * mutated the local-slot array; without this the array is discarded
         * at end of turn and a cross-turn `a(i).x` reads an empty array.
         * Arr (the matlab_struct_arr* itself) is stable across
         * get_or_create's internal growth, so it's the live array. */
        if (ReplMode && InScriptBody && NE->Ref->Kind == BindingKind::Var) {
          mlir::Value NameV2 = emitFieldNameChar(NE->Ref->Name, loc(F.Range));
          mlir::NamedAttribute PCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_set_struct_arr"));
          emitUnregOp("matlab.call_builtin", {NameV2, Arr},
                      {mlir::NoneType::get(&MCtx)}, loc(F.Range), {PCal});
        }
        return;
      }
    }
    const ClassDef *PinnedCls = nullptr;
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && BN->Ref->PinnedClass) PinnedCls = BN->Ref->PinnedClass;
    if (PinnedCls) {
      /* Dependent property with a user-defined set.Prop method:
       * dispatch to ClassName__set.Prop(obj, v). If the property is
       * Dependent but has NO set method defined, a write is a user
       * error at MATLAB level; we silently drop it here rather than
       * failing so the common read-only-dependent pattern works. */
      const ClassProp *DepProp = nullptr;
      const ClassDef *DepOwner = nullptr;
      for (const ClassDef *CC = PinnedCls; CC; CC = CC->Super) {
        for (const auto &P : CC->Props)
          if (P.Name == F.Field) {
            if (P.Dependent) { DepProp = &P; DepOwner = CC; }
            break;
          }
        if (DepProp) break;
      }
      if (DepProp) {
        std::string SetName = "set." + std::string(F.Field);
        const Function *SetMth = nullptr;
        for (const Function *Mm : DepOwner->Methods)
          if (Mm && Mm->Name == SetName) { SetMth = Mm; break; }
        if (SetMth) {
          mlir::Value Obj = lowerExpr(*F.Base);
          std::string Callee = std::string(DepOwner->Name) + "__set_" +
                                std::string(F.Field);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          emitUnregOp("matlab.call", {Obj, Rhs},
                      {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
        }
        return;
      }
      mlir::Value Obj = lowerExpr(*F.Base);
      maybeWrapCharStr();   // #79.2: classdef string property
      mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
      bool IsMatRhs = Rhs && (Rhs.getType() == PtrTy ||
                              mlir::isa<mlir::RankedTensorType,
                                        mlir::UnrankedTensorType>(Rhs.getType()));
      /* Property type annotation: when the classdef declares the
       * property as `Name string`, route stores through
       * matlab_obj_set_string so the runtime stores the value with
       * kind=3 and downstream reads (via the same TypeName-aware
       * read path) come back as a matlab_string *. */
      bool IsStringField = false;
      bool IsMatField = false;
      for (const ClassDef *CC = PinnedCls; CC; CC = CC->Super) {
        for (const auto &P : CC->Props)
          if (P.Name == F.Field) {
            if (P.TypeName == "string") IsStringField = true;
            else if (P.TypeName == "complex" || P.TypeName == "matrix" ||
                     P.TypeName == "double_col" || P.TypeName == "col")
              IsMatField = true;
            break;
          }
        if (IsStringField || IsMatField) break;
      }
      /* If the property has a matrix-typed annotation, force the mat
       * setter even when the Rhs source op carries an unresolved
       * `none` type (common for fresh call_builtin results before
       * type-propagation lands).  Without this, matrix RHS values get
       * mis-routed to `matlab_obj_set_f64` which silently drops the
       * payload. */
      llvm::StringRef Callee = (IsStringField || RhsIsCharStr) ? "matlab_obj_set_string"
                              : (IsMatRhs || IsMatField) ? "matlab_obj_set_mat"
                                                          : "matlab_obj_set_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      emitUnregOp("matlab.call_builtin", {Obj, NameV, Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
      return;
    }
    mlir::Value SPtr = resolveStructBase(F.Base, loc(F.Range));
    if (!SPtr) return;
    maybeWrapCharStr();   // #79.2: plain-struct char-string field
    mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
    bool IsMatRhs2 = Rhs && (Rhs.getType() == PtrTy ||
                             mlir::isa<mlir::RankedTensorType,
                                       mlir::UnrankedTensorType>(Rhs.getType()));
    /* Remember matrix-valued fields of a simple `s.field = M` so the read
     * side fetches them as a matrix (see MatStructFields).  A char-string
     * field (kind=3) is read back through the same `matlab_struct_get_mat`
     * path, so it counts as a "mat" field for read-side routing. */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref) {
        if (IsMatRhs2 || RhsIsCharStr)
          MatStructFields.insert({BN->Ref, std::string(F.Field)});
        else
          MatStructFields.erase({BN->Ref, std::string(F.Field)});
        if (RhsIsCharStr)
          StringStructFields.insert({BN->Ref, std::string(F.Field)});
        else
          StringStructFields.erase({BN->Ref, std::string(F.Field)});
      }
    llvm::StringRef Callee = RhsIsCharStr ? "matlab_struct_set_string"
                            : IsMatRhs2   ? "matlab_struct_set_mat"
                                          : "matlab_struct_set_f64";
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, Callee));
    emitUnregOp("matlab.call_builtin", {SPtr, NameV, Rhs},
                {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

mlir::Value Lowerer::lowerExpr(const Expr &E) {
  mlir::Location L = loc(E.Range);
  mlir::Type RT = mirTy(E.Ty ? E.Ty : TC.any());

  switch (E.Kind) {
  case NodeKind::IntegerLiteral: {
    int64_t V = foldInt(&E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::IntegerAttr::get(
                                mlir::IntegerType::get(&MCtx, 64), V));
    return emitUnreg("matlab.const_int", {}, RT, L, {A});
  }
  case NodeKind::FPLiteral: {
    double V = foldFloat(&E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::FloatAttr::get(
                                mlir::Float64Type::get(&MCtx), V));
    return emitUnreg("matlab.const_float", {}, RT, L, {A});
  }
  case NodeKind::ImagLiteral: {
    auto &I = static_cast<const ImagLiteral &>(E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::StringAttr::get(&MCtx, std::string(I.Text)));
    /* Always emit a ptr result — the runtime represents complex values
     * as matlab_mat_c* (1x1 for scalars). Sema may leave RT as
     * f64/complex-scalar; override to ptr so LowerTensorOps' complex
     * dispatch can pick the right runtime call. */
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    return emitUnreg("matlab.const_complex", {}, PtrTy, L, {A});
  }
  case NodeKind::StringLiteral: {
    /* Double-quoted "..." -> a matlab_string descriptor. We emit a
     * const_char carrying the literal bytes plus a call to the
     * runtime's matlab_string_from_literal which heap-copies them
     * into a { data, len } struct. This distinguishes real strings
     * from char arrays ('...' still lowers via matlab.const_char
     * directly) so later `+` / disp / strlen can dispatch on kind. */
    auto &S = static_cast<const StringLiteral &>(E);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::NamedAttribute VA(
        mlir::StringAttr::get(&MCtx, "value"),
        mlir::StringAttr::get(&MCtx, S.Value));
    mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                mlir::NoneType::get(&MCtx), L, {VA});
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
    return emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
  }
  case NodeKind::CharLiteral: {
    auto &S = static_cast<const CharLiteral &>(E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::StringAttr::get(&MCtx, S.Value));
    return emitUnreg("matlab.const_char", {}, RT, L, {A});
  }
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(E);
    /* No-paren constructor on the RHS: `m = occupancyMap;` (#79.1).
     * A bare name resolving to a classdef invokes the no-arg
     * constructor, exactly like `occupancyMap()` — call the emitted
     * `ClassName__ClassName` when the class has an explicit ctor, else
     * `matlab_obj_new(class_id)` plus any property defaults.  Mirrors
     * the positional / kwarg ctor paths in the CallOrIndex handler. */
    if (N.Ref && N.Ref->Kind == BindingKind::Class && N.Ref->ClassDef) {
      const ClassDef *CD = N.Ref->ClassDef;
      auto PtrTyC = mlir::LLVM::LLVMPointerType::get(&MCtx);
      bool HasCtor = false;
      for (const Function *Mth : CD->Methods)
        if (Mth && Mth->Name == CD->Name) { HasCtor = true; break; }
      if (HasCtor) {
        std::string Callee =
            std::string(CD->Name) + "__" + std::string(CD->Name);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        return emitUnreg("matlab.call", {}, PtrTyC, L, {Cal});
      }
      auto I32 = mlir::IntegerType::get(&MCtx, 32);
      mlir::Value ClsId = mlir::arith::ConstantOp::create(
          B, L, I32, mlir::IntegerAttr::get(I32, (int64_t)CD->ClassId));
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_obj_new"));
      mlir::Value Obj =
          emitUnreg("matlab.call_builtin", {ClsId}, PtrTyC, L, {Cal});
      for (const auto &P : CD->Props) {
        if (!P.Default) continue;
        mlir::Value DV = lowerExpr(*P.Default);
        mlir::Value NameV = emitFieldNameChar(P.Name, L);
        bool IsMat = DV && (DV.getType() == PtrTyC ||
                            mlir::isa<mlir::RankedTensorType,
                                      mlir::UnrankedTensorType>(DV.getType()));
        llvm::StringRef Cn = IsMat ? "matlab_obj_set_mat"
                                    : "matlab_obj_set_f64";
        mlir::NamedAttribute Cal2(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Cn));
        emitUnregOp("matlab.call_builtin", {Obj, NameV, DV},
                    {mlir::NoneType::get(&MCtx)}, L, {Cal2});
      }
      return Obj;
    }
    return loadBinding(N.Ref, E.Ty ? E.Ty : TC.any(), L);
  }
  case NodeKind::EndExpr: {
    // If we're inside a subscript arg, emit matlab.end with (base, dim)
    // operands so LowerTensorOps can rewrite it to matlab_end_of_dim.
    // Otherwise fall back to the zero-operand form — it won't survive
    // later passes, but the parser already errors on end-outside-indexing
    // so this path is really only reachable for weird IR.
    if (!SubscriptCtx.empty()) {
      auto [Base, Dim] = SubscriptCtx.back();
      auto F64End = mlir::Float64Type::get(&MCtx);
      // Sentinel dim -1: the base is a cell, so `end` (in `c{end}`) is
      // numel(cell) via matlab_cell_numel — matlab_end_of_dim would misread
      // the cell descriptor as a matlab_mat.
      if (Dim == -1) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_cell_numel"));
        return emitUnreg("matlab.call_builtin", {Base}, F64End, L, {Cal});
      }
      mlir::NamedAttribute VA(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::FloatAttr::get(F64End, (double)Dim));
      mlir::Value DimV = emitUnreg("matlab.const_float", {},
                                   F64End, L, {VA});
      return emitUnreg("matlab.end", {Base, DimV}, RT, L);
    }
    return emitUnreg("matlab.end", {}, RT, L);
  }
  case NodeKind::ColonExpr:
    return emitUnreg("matlab.colon", {}, RT, L);
  case NodeKind::BinaryOp: {
    auto &Bi = static_cast<const BinaryOpExpr &>(E);
    /* String concatenation: `"a" + "b"`, `s1 + s2`, or `s + n` /
     * `n + s` where one side is a known string (literal, binding,
     * or string-returning builtin call) and the other is a scalar.
     * Detect BEFORE lowering operands so we pick the right runtime
     * call and attach the ptr result type up front (the generic
     * matlab.add path would produce f64). When exactly one side is
     * a string and the other is a scalar, the scalar is coerced via
     * matlab_num2str so `"x = " + 25` produces "x = 25". */
    if (Bi.Op == BinOp::Add &&
        (isStringExpr(Bi.LHS) || isStringExpr(Bi.RHS))) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      auto F64 = mlir::Float64Type::get(&MCtx);
      mlir::Value LHS = lowerExpr(*Bi.LHS);
      mlir::Value RHS = lowerExpr(*Bi.RHS);
      /* The lowered MLIR type for a num2str/upper/sprintf/... call
       * may come back as `none` (the builtin's result type isn't
       * always refined). When the AST tells us it's a string, force
       * the type to ptr so matlab_string_concat sees PtrTy operands. */
      if (isStringExpr(Bi.LHS) && LHS.getType() != PtrTy)
        LHS.setType(PtrTy);
      if (isStringExpr(Bi.RHS) && RHS.getType() != PtrTy)
        RHS.setType(PtrTy);
      auto coerce = [&](mlir::Value V) -> mlir::Value {
        if (V.getType() == PtrTy) return V;
        /* Integer scalars (i8/i16/i32/i64) — extend to f64 first. */
        if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
          if (IT.getWidth() == 1)
            V = mlir::arith::UIToFPOp::create(B, L, F64, V);
          else
            V = mlir::arith::SIToFPOp::create(B, L, F64, V);
        } else if (V.getType() != F64) {
          /* Unknown scalar shape (e.g. f32) — best-effort: bitcast
           * isn't right, so fall back to a runtime cast via fptrunc/
           * fpext when possible. For now, only f64 is wired. */
          if (mlir::isa<mlir::FloatType>(V.getType())) {
            V = mlir::arith::ExtFOp::create(B, L, F64, V);
          } else {
            return V; /* leave as-is; downstream will error loudly */
          }
        }
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "num2str"));
        return emitUnreg("matlab.call_builtin", {V}, PtrTy, L, {Cal});
      };
      LHS = coerce(LHS);
      RHS = coerce(RHS);
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_string_concat"));
      return emitUnreg("matlab.call_builtin", {LHS, RHS}, PtrTy, L, {Cal});
    }
    /* Operator overloading: when either operand is a class-pinned
     * binding whose class defines a method named after the operator
     * (e.g. `plus`, `minus`, `times`, `mtimes`, `eq`, `ne`, `lt`,
     * `le`, `gt`, `ge`), dispatch to that method. MATLAB picks the
     * dominant class when both operands are objects of different
     * classes; for v1 we just prefer the LHS's class. */
    /* Walk an expression tree to find a class hint. NameExpr is the
     * direct case (a class-pinned variable). For inline composite
     * expressions like `s*s + 3*s` the LHS sub-expression is a
     * BinaryOp whose operator dispatch returns a fresh class
     * instance — recurse through BinaryOp/UnaryOp so the outer `+`
     * still picks up the class. CallOrIndex on a class constructor
     * also returns a class instance. */
    std::function<const ClassDef *(const Expr *)> pinnedFromExpr =
        [&pinnedFromExpr, this](const Expr *X) -> const ClassDef * {
      if (!X) return nullptr;
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        if (NE->Ref && NE->Ref->PinnedClass) return NE->Ref->PinnedClass;
      if (auto *Bi2 = dynamic_cast<const BinaryOpExpr *>(X)) {
        bool IsCmp =
            Bi2->Op == BinOp::Eq || Bi2->Op == BinOp::Ne ||
            Bi2->Op == BinOp::Lt || Bi2->Op == BinOp::Le ||
            Bi2->Op == BinOp::Gt || Bi2->Op == BinOp::Ge;
        if (!IsCmp) {
          if (auto *L = pinnedFromExpr(Bi2->LHS)) return L;
          if (auto *R = pinnedFromExpr(Bi2->RHS)) return R;
        }
        return nullptr;
      }
      if (auto *U2 = dynamic_cast<const UnaryOpExpr *>(X))
        return pinnedFromExpr(U2->Operand);
      if (auto *CX = dynamic_cast<const CallOrIndex *>(X)) {
        if (auto *NX = dynamic_cast<const NameExpr *>(CX->Callee)) {
          if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
              NX->Ref->ClassDef)
            return NX->Ref->ClassDef;
          /* dlarray-returning function-style calls (`relu`/`sigmoid`/
           * `mse`/...).  Without this branch, two dlarray-returning
           * calls combined by `+` (`mse(Y1,T1) + mse(Y2,T2)`) miss the
           * classdef-operator-overloading dispatch and crash through
           * `matlab_add_mm` interpreting dlarray pointers as matrices. */
          static const llvm::StringSet<> DlRet2 = {
              "relu", "sigmoid", "tanh", "softmax", "sum", "mean",
              "log", "exp", "crossentropy", "mse", "lstm",
              "transpose", "ctranspose", "embed",
              "gru", "bilstm", "lstmp", "dlarray",
              "sqrt", "leakyrelu", "gelu", "swish",
              "softplus", "elu", "conv2d_batch", "conv2d_full",
              "reshape", "maxpool2d", "avgpool2d", "batchnorm",
              "layernorm", "batchnorm_eval",
              "groupnorm", "batchnorm_train",
              "instancenorm", "rmsnorm"};
          if (DlRet2.contains(NX->Name) && this->CurTU) {
            bool argPinned = false;
            for (size_t i = 0; i < CX->Args.size(); ++i)
              if (pinnedFromExpr(CX->Args[i])) { argPinned = true; break; }
            if (argPinned) {
              for (const ClassDef *DC : this->CurTU->Classes)
                if (DC && DC->Name == "dlarray") return DC;
            }
          }
        }
      }
      return nullptr;
    };
    const ClassDef *OpCls = pinnedFromExpr(Bi.LHS);
    if (!OpCls) OpCls = pinnedFromExpr(Bi.RHS);
    if (OpCls) {
      llvm::StringRef OpMethod;
      switch (Bi.Op) {
        case BinOp::Add:          OpMethod = "plus";     break;
        case BinOp::Sub:          OpMethod = "minus";    break;
        case BinOp::Mul:          OpMethod = "mtimes";   break;
        case BinOp::Div:          OpMethod = "mrdivide"; break;
        case BinOp::LeftDiv:      OpMethod = "mldivide"; break;
        case BinOp::Pow:          OpMethod = "mpower";   break;
        case BinOp::ElemMul:      OpMethod = "times";    break;
        case BinOp::ElemDiv:      OpMethod = "rdivide";  break;
        case BinOp::ElemLeftDiv:  OpMethod = "ldivide";  break;
        case BinOp::ElemPow:      OpMethod = "power";    break;
        case BinOp::Eq:           OpMethod = "eq";       break;
        case BinOp::Ne:           OpMethod = "ne";       break;
        case BinOp::Lt:           OpMethod = "lt";       break;
        case BinOp::Le:           OpMethod = "le";       break;
        case BinOp::Gt:           OpMethod = "gt";       break;
        case BinOp::Ge:           OpMethod = "ge";       break;
        default: break;
      }
      if (!OpMethod.empty()) {
        const ClassDef *Owner = nullptr;
        std::string_view OpSV(OpMethod.data(), OpMethod.size());
        for (const ClassDef *CC = OpCls; CC; CC = CC->Super) {
          for (const Function *Mm : CC->Methods)
            if (Mm && Mm->Name == OpSV) { Owner = CC; break; }
          if (Owner) break;
        }
        if (Owner) {
          // #191 P3 acceptance probe. When MATLAB_LLVM_PROBE_LATE_MONO is set,
          // report every class operator that reaches the LOWERING synthesis
          // path — i.e. one the Sema-time dispatch-desynth pass did NOT rewrite
          // into an explicit method call. For a fully-migrated class this site
          // must never fire in whole-program (AOT) compilation; a fire names a
          // gap to close before the synthesis site can be removed. Diagnostic
          // only: gated off the env var, zero behaviour change when unset.
          static const bool ProbeLateMono =
              ::getenv("MATLAB_LLVM_PROBE_LATE_MONO") != nullptr;
          if (ProbeLateMono) {
            bool LhsObj = pinnedFromExpr(Bi.LHS) != nullptr;
            bool RhsObj = pinnedFromExpr(Bi.RHS) != nullptr;
            llvm::errs() << "[late-mono-probe] op-synth " << Owner->Name
                         << "::" << OpMethod << " lhs_obj=" << LhsObj
                         << " rhs_obj=" << RhsObj << "\n";
          }
          mlir::Value LHS = Bi.LHS ? lowerExpr(*Bi.LHS) : mlir::Value{};
          mlir::Value RHS = Bi.RHS ? lowerExpr(*Bi.RHS) : mlir::Value{};
          /* Scalar-mixing boxing: when one operand is a class
           * instance (Owner) and the other is a non-class value
           * (scalar f64, integer, raw matrix ptr), wrap the non-
           * class operand in a one-arg `Owner(value)` constructor
           * call so the class's operator method body sees two
           * class-pinned operands. MATLAB's convention for `G + 2`
           * is `G + Owner(2)`. Restricted to CST prelude classes
           * (tf / ss / zpk / pid / frd) — other user classdefs
           * (Vec2, BasicClass, …) typically don't have a 1-arg
           * constructor handling the scalar-promotion case and
           * would crash on a 1-arg invocation. The CST classes
           * explicitly support `tf(c)` → constant-tf semantics. */
          llvm::StringRef OCN = Owner->Name;
          bool BoxScalars = (OCN == "tf" || OCN == "ss" ||
                              OCN == "zpk" || OCN == "pid" ||
                              OCN == "frd" ||
                              /* Optimization Toolbox problem-based
                               * expressions: `2*x` / `x + 3` box the
                               * scalar into a constant-node
                               * OptimizationExpression(c). */
                              OCN == "OptimizationExpression");
          if (BoxScalars) {
            auto PtrTyW = mlir::LLVM::LLVMPointerType::get(&MCtx);
            auto wrapIfScalar = [&](mlir::Value V, const Expr *Op) -> mlir::Value {
              if (!V) return V;
              const ClassDef *Pinned = pinnedFromExpr(Op);
              if (Pinned) return V;
              mlir::Type T = V.getType();
              if (T == PtrTyW) return V;
              std::string Ctor = std::string(Owner->Name) + "__" +
                                  std::string(Owner->Name);
              mlir::NamedAttribute CC(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Ctor));
              mlir::NamedAttribute UA(
                  mlir::StringAttr::get(&MCtx, "user_arity"),
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(&MCtx, 64), 1));
              return emitUnreg("matlab.call", {V}, PtrTyW, L, {CC, UA});
            };
            LHS = wrapIfScalar(LHS, Bi.LHS);
            RHS = wrapIfScalar(RHS, Bi.RHS);
          }
          std::string Callee = std::string(Owner->Name) + "__" +
                                std::string(OpMethod);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          /* Pick a concrete result type: comparison operators return
           * f64 (logical 0/1); arithmetic operators return a class
           * instance (matlab_obj*). Leaving RT as `none` would force
           * the slot receiving this value to stay none-typed through
           * all the pipelines. */
          bool IsCmp = (Bi.Op == BinOp::Eq || Bi.Op == BinOp::Ne ||
                        Bi.Op == BinOp::Lt || Bi.Op == BinOp::Le ||
                        Bi.Op == BinOp::Gt || Bi.Op == BinOp::Ge);
          mlir::Type ResTy = IsCmp
              ? (mlir::Type)mlir::Float64Type::get(&MCtx)
              : (mlir::Type)mlir::LLVM::LLVMPointerType::get(&MCtx);
          return emitUnreg("matlab.call", {LHS, RHS}, ResTy, L, {Cal});
        }
      }
    }
    /* #234: a char literal in ARITHMETIC / comparison promotes to its numeric
     * code(s) (`'A' + 1 == 66`), unlike a string ("A") which concatenates. By
     * here the string-concat and class-overload paths have already returned,
     * so a remaining CharLiteral operand is genuinely numeric (matched by the
     * Sema re-typing in visitBinary). Lowering it the default way yields a
     * matlab_string* and matlab.add(ptr, f64) can't convert; emit its code(s)
     * instead — a scalar f64 for a single char, a 1xN row matrix otherwise. */
    bool CharArithOp =
        Bi.Op == BinOp::Add || Bi.Op == BinOp::Sub || Bi.Op == BinOp::Mul ||
        Bi.Op == BinOp::Div || Bi.Op == BinOp::LeftDiv || Bi.Op == BinOp::Pow ||
        Bi.Op == BinOp::ElemMul || Bi.Op == BinOp::ElemDiv ||
        Bi.Op == BinOp::ElemLeftDiv || Bi.Op == BinOp::ElemPow ||
        /* comparisons too — Sema only re-types the char operand numeric when
         * the other operand is numeric (`'A' == 65`); a char-vs-string compare
         * keeps a string result type and stays on its existing path. */
        Bi.Op == BinOp::Eq || Bi.Op == BinOp::Ne || Bi.Op == BinOp::Lt ||
        Bi.Op == BinOp::Le || Bi.Op == BinOp::Gt || Bi.Op == BinOp::Ge;
    auto lowerArithOperand = [&](const Expr *E) -> mlir::Value {
      if (CharArithOp)
        if (auto *CL = dynamic_cast<const CharLiteral *>(E)) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto codeConst = [&](unsigned char c) {
            return emitUnreg(
                "matlab.const_float", {}, F64, L,
                {mlir::NamedAttribute(
                    mlir::StringAttr::get(&MCtx, "value"),
                    mlir::FloatAttr::get(F64, (double)c))});
          };
          if (CL->Value.size() == 1)
            return codeConst((unsigned char)CL->Value[0]);
          if (CL->Value.size() > 1) {
            /* Multi-char literal -> a 1xN row of codes, built the same way a
             * numeric row literal `[65 66]` lowers (concat_row of const_float).
             * `'AB' + 1` then takes the matrix+scalar arithmetic path. */
            llvm::SmallVector<mlir::Value, 8> Codes;
            for (char c : CL->Value)
              Codes.push_back(codeConst((unsigned char)c));
            auto RowTy = mlir::RankedTensorType::get(
                {(int64_t)CL->Value.size()}, F64);
            return emitUnreg("matlab.concat_row", Codes, RowTy, L);
          }
        }
      return E ? lowerExpr(*E) : mlir::Value{};
    };
    mlir::Value LHS = Bi.LHS ? lowerArithOperand(Bi.LHS) : mlir::Value{};
    mlir::Value RHS = Bi.RHS ? lowerArithOperand(Bi.RHS) : mlir::Value{};
    /* Eagerly refine the result type when Sema left the expression
     * type as `any`/none:
     *   - both operands same primitive scalar type  -> same scalar
     *   - either operand is a ptr (matrix handle)    -> ptr
     *     (matrix-matrix and matrix-scalar ops return a matrix)
     *   - comparison operators on ptr operands       -> ptr
     * This lets downstream rewrites match (the scalar-to-arith and
     * matrix-runtime paths both require a concrete result type), and
     * lets implicit-display in ExprStmt see a non-None value. */
    auto MLPtr = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Type ResTy = RT;
    /* Refine when Sema left the type open (None). */
    if (mlir::isa<mlir::NoneType>(ResTy) && LHS && RHS) {
      if (LHS.getType() == RHS.getType() &&
          mlir::isa<mlir::Float64Type, mlir::IntegerType>(LHS.getType())) {
        ResTy = LHS.getType();
      } else if (LHS.getType() == MLPtr || RHS.getType() == MLPtr) {
        ResTy = MLPtr;
      } else if (mlir::isa<mlir::RankedTensorType, mlir::UnrankedTensorType>(
                     LHS.getType()) &&
                 mlir::isa<mlir::Float64Type, mlir::IntegerType>(RHS.getType())) {
        /* tensor <op> scalar — scalar broadcasts elementwise, so the result
         * keeps the tensor shape (Sema leaves this None when the scalar side
         * is a computed value like `2*pi`, which then poisoned sin/cos etc.). */
        ResTy = LHS.getType();
      } else if (mlir::isa<mlir::RankedTensorType, mlir::UnrankedTensorType>(
                     RHS.getType()) &&
                 mlir::isa<mlir::Float64Type, mlir::IntegerType>(LHS.getType())) {
        /* scalar <op> tensor — symmetric broadcast. */
        ResTy = RHS.getType();
      }
    }
    /* REPL override: workspace reads always come back as ptr (the
     * runtime auto-boxes scalars to 1x1 matrices on get_mat). Sema
     * may still have inferred the binop result as f64 from source-
     * level "x = 1; y = 2; z = x + y" propagation, but at MLIR level
     * both operands are ptr, so the result must be ptr to stay
     * well-typed through LowerTensorOps — otherwise we'd emit a
     * `matlab.add(ptr, ptr) : f64`, and the downstream set_f64
     * picker commits to the scalar path while add_mm replaces the
     * operand with a ptr, leaving an ill-typed llvm.call. */
    if (LHS && RHS &&
        (LHS.getType() == MLPtr || RHS.getType() == MLPtr)) {
      ResTy = MLPtr;
    }
    /* Fixed-Point Designer: tag the binop with the result FixedSpec so
     * LowerFixedPoint can rewrite it into integer-shift sequences. We
     * also forward the operand specs as separate attributes (the rewrite
     * needs to know each operand's FL to emit the alignment shift). */
    auto fiSpec = [](const Expr *X) -> std::optional<FixedSpec> {
      if (!X || !X->Ty || X->Ty->K != Type::Kind::Array) return std::nullopt;
      auto &A = static_cast<const ArrayType &>(*X->Ty);
      if (A.Elt != Dtype::Fixed || !A.FxSpec) return std::nullopt;
      return *A.FxSpec;
    };
    auto LhsSpec = fiSpec(Bi.LHS);
    auto RhsSpec = fiSpec(Bi.RHS);
    auto ResSpec = (Bi.Ty && Bi.Ty->K == Type::Kind::Array)
        ? static_cast<const ArrayType &>(*Bi.Ty).FxSpec
        : std::nullopt;
    if (ResSpec && (LhsSpec || RhsSpec)) {
      /* Mixed fi + double: cast the double side into the fi side's spec
       * (MATLAB's Phase-1 promotion rule). When the double is a literal,
       * we constant-fold here; otherwise we emit a runtime quantize.
       * Either way, both operands become integer-typed by the time we
       * emit the binop, so LowerFixedPoint sees the canonical shape. */
      auto promoteOperand = [&](mlir::Value V, const Expr *Src,
                                const std::optional<FixedSpec> &OwnSpec,
                                const FixedSpec &Target,
                                std::optional<FixedSpec> &OutSpec) {
        if (OwnSpec) { OutSpec = OwnSpec; return V; }
        if (!V || !mlir::isa<mlir::Float64Type, mlir::Float32Type>(V.getType()))
          return V;
        // Literal-fold: fold the AST node directly so the stored value
        // is already in the IR as an arith.constant.
        auto isLit = [](const Expr *X) {
          if (!X) return false;
          if (X->Kind == NodeKind::IntegerLiteral ||
              X->Kind == NodeKind::FPLiteral) return true;
          if (auto *U = dynamic_cast<const UnaryOpExpr *>(X))
            if (U->Op == UnOp::Minus || U->Op == UnOp::Plus)
              return U->Operand && (
                  U->Operand->Kind == NodeKind::IntegerLiteral ||
                  U->Operand->Kind == NodeKind::FPLiteral);
          return false;
        };
        OutSpec = Target;
        uint8_t Bits = Target.storageBits();
        auto IT = mlir::IntegerType::get(&MCtx, Bits == 0 ? 64 : Bits);
        if (isLit(Src)) {
          double Val = foldFloat(Src);
          int64_t Stored = Target.Signed
              ? quantizeFixedSigned(Val, Target)
              : (int64_t)quantizeFixedUnsigned(Val, Target);
          return (mlir::Value)mlir::arith::ConstantOp::create(
              B, L, IT, mlir::IntegerAttr::get(IT, Stored));
        }
        // Runtime path: call matlab_fi_quantize_{s,u}.
        auto F64 = mlir::Float64Type::get(&MCtx);
        if (V.getType() != F64)
          V = mlir::arith::ExtFOp::create(B, L, F64, V);
        llvm::SmallVector<mlir::NamedAttribute, 8> CA;
        auto QAttrs = buildFixedAttrs(&MCtx, Target);
        for (auto &E0 : QAttrs) CA.push_back(E0);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx,
                Target.Signed ? "matlab_fi_quantize_s"
                              : "matlab_fi_quantize_u"));
        CA.push_back(Cal);
        return emitUnreg("matlab.fi.cast", {V}, IT, L, CA);
      };
      LHS = promoteOperand(LHS, Bi.LHS, LhsSpec, *ResSpec, LhsSpec);
      RHS = promoteOperand(RHS, Bi.RHS, RhsSpec, *ResSpec, RhsSpec);

      llvm::SmallVector<mlir::NamedAttribute, 16> A;
      auto Outer = buildFixedAttrs(&MCtx, *ResSpec);
      for (auto &E0 : Outer) A.push_back(E0);
      auto I32 = mlir::IntegerType::get(&MCtx, 32);
      auto I1 = mlir::IntegerType::get(&MCtx, 1);
      auto pushOperand = [&](llvm::StringRef Pre,
                             const std::optional<FixedSpec> &S) {
        if (!S) return;
        A.emplace_back(mlir::StringAttr::get(&MCtx, (Pre + "_signed").str()),
                       mlir::IntegerAttr::get(I1, S->Signed ? 1 : 0));
        A.emplace_back(mlir::StringAttr::get(&MCtx, (Pre + "_wl").str()),
                       mlir::IntegerAttr::get(I32, (int64_t)S->WordLength));
        A.emplace_back(mlir::StringAttr::get(&MCtx, (Pre + "_fl").str()),
                       mlir::IntegerAttr::get(I32, (int64_t)S->FractionLength));
      };
      pushOperand("fi_lhs", LhsSpec);
      pushOperand("fi_rhs", RhsSpec);
      // Override ResTy to the fi storage class — Sema's mapType already
      // returns the right thing, but if ResTy was refined to ptr above
      // by the REPL guard it would be wrong for fi. Restore from RT.
      mlir::Type FiResTy = mirTy(Bi.Ty);
      if (mlir::isa<mlir::IntegerType>(FiResTy)) ResTy = FiResTy;
      return emitUnreg(binOpName(Bi.Op), {LHS, RHS}, ResTy, L, A);
    }
    /* Phase 5.1: datetime / duration arithmetic. Detect the operand
     * kinds via DatetimeBindings / DurationBindings (NameExpr LHS /
     * RHS only — chained binops are out of scope for v1) and route
     * to the correct runtime entry. */
    auto isDtName = [&](const Expr *X) -> bool {
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        return NE->Ref && DatetimeBindings.count(NE->Ref);
      if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
        if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee))
          if (NE->Name == "datetime") return true;
      return false;
    };
    auto isDurName = [&](const Expr *X) -> bool {
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        return NE->Ref && DurationBindings.count(NE->Ref);
      if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
        if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee))
          if (NE->Name == "seconds" || NE->Name == "minutes" ||
              NE->Name == "hours"   || NE->Name == "days"    ||
              NE->Name == "years"   || NE->Name == "duration")
            return true;
      return false;
    };
    /* Phase 5.4: vec equivalents. A unit-constructor call counts as a
     * vec when its single arg is a ColonExpr (the `0:251` form that
     * drives `datetime(...) + days(0:251)`); for NameExpr LHS / RHS
     * the *Vec*Bindings set carries the tag set by AssignStmt. */
    auto isDtVecName = [&](const Expr *X) -> bool {
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        return NE->Ref && DatetimeVecBindings.count(NE->Ref);
      return false;
    };
    auto isDurVecName = [&](const Expr *X) -> bool {
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        return NE->Ref && DurationVecBindings.count(NE->Ref);
      if (auto *CX = dynamic_cast<const CallOrIndex *>(X))
        if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee))
          if (NE->Name == "seconds" || NE->Name == "minutes" ||
              NE->Name == "hours"   || NE->Name == "days"    ||
              NE->Name == "years")
            if (!CX->Args.empty() && CX->Args[0] &&
                (dynamic_cast<const RangeExpr *>(CX->Args[0]) ||
                 dynamic_cast<const ColonExpr *>(CX->Args[0]) ||
                 dynamic_cast<const MatrixLiteral *>(CX->Args[0])))
              return true;
      return false;
    };
    {
      auto PtrTy2 = mlir::LLVM::LLVMPointerType::get(&MCtx);
      llvm::StringRef DtCallee;
      bool SwapOperands = false;
      /* Phase 5.4 vec dispatch — check vec combos first so a single
       * NameExpr binding can be scalar or vec without ambiguity. */
      if (Bi.Op == BinOp::Add && isDtName(Bi.LHS) && isDurVecName(Bi.RHS))
        DtCallee = "matlab_datetime_add_duration_vec";
      else if (Bi.Op == BinOp::Add && isDurVecName(Bi.LHS) && isDtName(Bi.RHS)) {
        DtCallee = "matlab_datetime_add_duration_vec";
        SwapOperands = true;
      }
      else if (Bi.Op == BinOp::Add && isDtVecName(Bi.LHS) && isDurName(Bi.RHS))
        DtCallee = "matlab_datetime_vec_add_duration";
      else if (Bi.Op == BinOp::Add && isDurName(Bi.LHS) && isDtVecName(Bi.RHS)) {
        DtCallee = "matlab_datetime_vec_add_duration";
        SwapOperands = true;
      }
      else if (Bi.Op == BinOp::Sub && isDtVecName(Bi.LHS) && isDurName(Bi.RHS))
        DtCallee = "matlab_datetime_vec_sub_duration";
      else if (Bi.Op == BinOp::Add && isDtVecName(Bi.LHS) && isDurVecName(Bi.RHS))
        DtCallee = "matlab_datetime_vec_add_duration_vec";
      else if (Bi.Op == BinOp::Sub && isDtVecName(Bi.LHS) && isDtVecName(Bi.RHS))
        DtCallee = "matlab_datetime_vec_sub_datetime_vec";
      else if (Bi.Op == BinOp::Sub && isDtVecName(Bi.LHS) && isDtName(Bi.RHS))
        DtCallee = "matlab_datetime_vec_sub_datetime";
      /* Scalar fall-through (matches the original block). */
      else if (Bi.Op == BinOp::Sub && isDtName(Bi.LHS) && isDtName(Bi.RHS))
        DtCallee = "matlab_datetime_sub_datetime";
      else if (Bi.Op == BinOp::Add && isDtName(Bi.LHS) && isDurName(Bi.RHS))
        DtCallee = "matlab_datetime_add_duration";
      else if (Bi.Op == BinOp::Add && isDurName(Bi.LHS) && isDtName(Bi.RHS)) {
        DtCallee = "matlab_datetime_add_duration";
        SwapOperands = true;
      }
      else if (Bi.Op == BinOp::Sub && isDtName(Bi.LHS) && isDurName(Bi.RHS))
        DtCallee = "matlab_datetime_sub_duration";
      else if (Bi.Op == BinOp::Add && isDurName(Bi.LHS) && isDurName(Bi.RHS))
        DtCallee = "matlab_duration_add";
      else if (Bi.Op == BinOp::Sub && isDurName(Bi.LHS) && isDurName(Bi.RHS))
        DtCallee = "matlab_duration_sub";
      if (!DtCallee.empty()) {
        mlir::Value LO = LHS, RO = RHS;
        if (SwapOperands) std::swap(LO, RO);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, std::string(DtCallee)));
        return emitUnreg("matlab.call_builtin", {LO, RO},
                         PtrTy2, L, {Cal});
      }
    }
    /* Phase 6: symbolic arithmetic dispatch. When either operand is a
     * sym-bound NameExpr or a sym-producing call, route the binary
     * operators (+, -, mul, div, pow, ==) to the matlab_sym_* runtime.
     * Mixed-mode arithmetic (sym op double) goes through the _d variants
     * without boxing the literal. */
    {
      bool LhsIsSym = exprIsSym(Bi.LHS);
      bool RhsIsSym2 = exprIsSym(Bi.RHS);
      if (LhsIsSym || RhsIsSym2) {
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto PtrTy3 = mlir::LLVM::LLVMPointerType::get(&MCtx);
        const char *Pure = nullptr;     /* sym <op> sym */
        const char *MixR = nullptr;     /* sym <op> double */
        const char *MixL = nullptr;     /* double <op> sym */
        switch (Bi.Op) {
          case BinOp::Add: Pure = "matlab_sym_add"; MixR = "matlab_sym_add_d"; MixL = "matlab_sym_add_d"; break;
          case BinOp::Sub: Pure = "matlab_sym_sub"; MixR = "matlab_sym_sub_d"; MixL = "matlab_sym_d_sub"; break;
          case BinOp::Mul: Pure = "matlab_sym_mul"; MixR = "matlab_sym_mul_d"; MixL = "matlab_sym_mul_d"; break;
          case BinOp::ElemMul: Pure = "matlab_sym_mul"; MixR = "matlab_sym_mul_d"; MixL = "matlab_sym_mul_d"; break;
          case BinOp::Div: Pure = "matlab_sym_div"; MixR = "matlab_sym_div_d"; MixL = "matlab_sym_d_div"; break;
          case BinOp::ElemDiv: Pure = "matlab_sym_div"; MixR = "matlab_sym_div_d"; MixL = "matlab_sym_d_div"; break;
          case BinOp::Pow: Pure = "matlab_sym_pow"; MixR = "matlab_sym_pow_d"; MixL = "matlab_sym_d_pow"; break;
          case BinOp::ElemPow: Pure = "matlab_sym_pow"; MixR = "matlab_sym_pow_d"; MixL = "matlab_sym_d_pow"; break;
          case BinOp::Eq: Pure = "matlab_sym_eq"; MixR = "matlab_sym_eq_d"; MixL = nullptr; break;
          default: break;
        }
        if (Pure) {
          llvm::StringRef Callee;
          mlir::Value Lo = LHS, Ro = RHS;
          if (LhsIsSym && RhsIsSym2) Callee = Pure;
          else if (LhsIsSym && RHS && RHS.getType() == F64) Callee = MixR;
          else if (RhsIsSym2 && LHS && LHS.getType() == F64) Callee = MixL;
          if (!Callee.empty()) {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee.str()));
            return emitUnreg("matlab.call_builtin", {Lo, Ro},
                             PtrTy3, L, {Cal});
          }
        }
      }
    }
    /* Phase 1.1.D: typed-int matrix dispatch. When either operand is a
     * non-scalar Int32 / UInt8 array, attach a `dtype` StringAttr to the
     * matlab.{add,sub,emul,ediv,...} op. LowerTensorOps reads this attr
     * to route the call to the typed runtime ABI (matlab_mat_i32_add_mm
     * vs matlab_add_mm). The attribute is the only signal — by the time
     * the rewrite runs both lanes look like opaque !llvm.ptr operands. */
    llvm::StringRef IntSuf = intDtypeSuffixOf(Bi.LHS);
    if (IntSuf.empty()) IntSuf = intDtypeSuffixOf(Bi.RHS);
    /* Comparisons fold to Logical at Sema, so Bi.Ty itself is no longer
     * Int32/UInt8; operand types remain the source of truth. */
    if (!IntSuf.empty()) {
      mlir::NamedAttribute Dt(
          mlir::StringAttr::get(&MCtx, "dtype"),
          mlir::StringAttr::get(&MCtx, IntSuf));
      return emitUnreg(binOpName(Bi.Op), {LHS, RHS}, ResTy, L, {Dt});
    }
    return emitUnreg(binOpName(Bi.Op), {LHS, RHS}, ResTy, L);
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(E);
    mlir::Value A = U.Operand ? lowerExpr(*U.Operand) : mlir::Value{};
    /* Phase 6 / #241: unary minus on any sym-valued operand routes to
     * matlab_sym_neg.  exprIsSym recognises not just a sym-bound *name* but
     * also sym function-calls (e.g. -sin(theta)) and sym sub-expressions;
     * the old NameExpr-only check let those fall through to the numeric
     * matlab_neg_m, which segfaults when handed a sym pointer. */
    if (U.Op == UnOp::Minus && U.Operand) {
      if (exprIsSym(U.Operand) && A) {
        auto PtrTyU = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_sym_neg"));
        return emitUnreg("matlab.call_builtin", {A}, PtrTyU, L, {Cal});
      }
    }
    /* Class-method operator overload for unary ops. Same dispatch
     * shape as BinaryOp's class-pinned path: when the operand is
     * pinned to a class that defines a method named after the
     * operator (uminus / uplus / not / ctranspose / transpose),
     * route the call through the class. Without this, `-tf_obj`
     * calls matlab_neg_m on the class-instance pointer and reads
     * garbage. */
    if (U.Operand) {
      auto pinnedFromExpr = [](const Expr *X) -> const ClassDef * {
        if (auto *NE = dynamic_cast<const NameExpr *>(X))
          if (NE->Ref && NE->Ref->PinnedClass) return NE->Ref->PinnedClass;
        return nullptr;
      };
      const ClassDef *OpCls = pinnedFromExpr(U.Operand);
      if (OpCls && A) {
        llvm::StringRef OpMethod;
        switch (U.Op) {
          case UnOp::Minus: OpMethod = "uminus"; break;
          case UnOp::Plus:  OpMethod = "uplus";  break;
          case UnOp::Not:   OpMethod = "not";    break;
        }
        if (!OpMethod.empty()) {
          const ClassDef *Owner = nullptr;
          std::string_view OpSV(OpMethod.data(), OpMethod.size());
          for (const ClassDef *CC = OpCls; CC; CC = CC->Super) {
            for (const Function *Mm : CC->Methods)
              if (Mm && Mm->Name == OpSV) { Owner = CC; break; }
            if (Owner) break;
          }
          if (Owner) {
            std::string Callee = std::string(Owner->Name) + "__" +
                                  std::string(OpMethod);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            mlir::Type ResTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            return emitUnreg("matlab.call", {A}, ResTy, L, {Cal});
          }
        }
      }
    }
    /* Same refinement as BinaryOp/PostfixOp: a unary op on a matrix
     * returns a matrix, on a scalar returns the same scalar type. */
    auto MLPtr = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Type ResTy = RT;
    if (mlir::isa<mlir::NoneType>(ResTy) && A) {
      if (A.getType() == MLPtr) ResTy = MLPtr;
      else if (mlir::isa<mlir::Float64Type, mlir::IntegerType>(A.getType()))
        ResTy = A.getType();
    }
    /* REPL override, same as BinaryOp above. */
    if (A && A.getType() == MLPtr) ResTy = MLPtr;
    return emitUnreg(unOpName(U.Op), {A}, ResTy, L);
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<const PostfixOpExpr &>(E);
    mlir::Value A = P.Operand ? lowerExpr(*P.Operand) : mlir::Value{};
    /* Eagerly refine: transpose of a matrix (ptr) is still a matrix.
     * Sema leaves Ty=any for Var operands, so without this the
     * result stays NoneType — which breaks downstream disp / store
     * dispatch and the REPL implicit-display check. */
    auto MLPtr = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Type ResTy = RT;
    if (mlir::isa<mlir::NoneType>(ResTy) && A && A.getType() == MLPtr)
      ResTy = MLPtr;
    /* REPL override, same as BinaryOp above. */
    if (A && A.getType() == MLPtr) ResTy = MLPtr;
    return emitUnreg(postfixName(P.Op), {A}, ResTy, L);
  }
  case NodeKind::RangeExpr: {
    auto &R = static_cast<const RangeExpr &>(E);
    llvm::SmallVector<mlir::Value, 3> Os;
    if (R.Start) Os.push_back(lowerExpr(*R.Start));
    if (R.Step)  Os.push_back(lowerExpr(*R.Step));
    if (R.End)   Os.push_back(lowerExpr(*R.End));
    mlir::NamedAttribute HS(
        mlir::StringAttr::get(&MCtx, "has_step"),
        mlir::BoolAttr::get(&MCtx, R.Step != nullptr));
    return emitUnreg("matlab.range", Os, RT, L, {HS});
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(E);
    /* Phase 4: containers.Map(...) — runs before the CallKind::Call
     * gate because the resolver doesn't classify `containers.Map`
     * as a call (it's a FieldAccess on a builtin namespace). */
    if (auto *FAEarly = dynamic_cast<const FieldAccess *>(C.Callee))
      if (auto *BNEarly = dynamic_cast<const NameExpr *>(FAEarly->Base))
        if (BNEarly->Name == "containers" && FAEarly->Field == "Map") {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_dict_new"));
          return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
        }
    /* struct('f1', v1, 'f2', v2, ...) constructor (#28).  Build a fresh
     * matlab_struct and set each name/value pair; field names must be
     * string/char literals (the common scalar-struct form).  Matrix /
     * struct values route through _set_mat, scalars through _set_f64. */
    if (auto *SNE = dynamic_cast<const NameExpr *>(C.Callee))
      if (SNE->Ref && SNE->Ref->Kind == BindingKind::Builtin &&
          SNE->Name == "struct" && C.Args.size() >= 2 &&
          (C.Args.size() % 2) == 0) {
        auto literalName = [](const Expr *A) -> const std::string * {
          if (auto *S = dynamic_cast<const StringLiteral *>(A)) return &S->Value;
          if (auto *Ch = dynamic_cast<const CharLiteral *>(A)) return &Ch->Value;
          return nullptr;
        };
        bool AllNames = true;
        for (size_t i = 0; i < C.Args.size(); i += 2)
          if (!C.Args[i] || !literalName(C.Args[i])) { AllNames = false; break; }
        if (AllNames) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          mlir::NamedAttribute NewCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_struct_new"));
          mlir::Value S = emitUnreg("matlab.call_builtin", {}, PtrTy, L, {NewCal});
          for (size_t i = 0; i + 1 < C.Args.size(); i += 2) {
            mlir::Value NameV = emitFieldNameChar(*literalName(C.Args[i]), L);
            mlir::Value Val = lowerExpr(*C.Args[i + 1]);
            bool IsMat = Val.getType() == PtrTy ||
                         mlir::isa<mlir::RankedTensorType,
                                   mlir::UnrankedTensorType>(Val.getType());
            llvm::StringRef Callee = IsMat ? "matlab_struct_set_mat"
                                            : "matlab_struct_set_f64";
            mlir::NamedAttribute SCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            emitUnregOp("matlab.call_builtin", {S, NameV, Val},
                        {mlir::NoneType::get(&MCtx)}, L, {SCal});
          }
          return S;
        }
      }
    /* Phase 5.4 (cont.): plot(dt_vec, y, ...) — auto-wrap the
     * first arg with matlab_datetime_vec_to_mat so the existing
     * matrix-only plot backend gets a usable numeric x-axis (days
     * from start). Date-formatted tick labels live downstream.    */
    if (auto *PNE = dynamic_cast<const NameExpr *>(C.Callee))
      if (PNE->Ref && PNE->Ref->Kind == BindingKind::Builtin &&
          PNE->Name == "plot" && !C.Args.empty() && C.Args[0]) {
        bool FirstIsDtVec = false;
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && DatetimeVecBindings.count(ArgN->Ref))
            FirstIsDtVec = true;
        if (auto *FA = dynamic_cast<const FieldAccess *>(C.Args[0]))
          if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
            if (BN->Ref && isTimetableBinding(BN->Ref) &&
                FA->Field == "Time")
              FirstIsDtVec = true;
        if (FirstIsDtVec) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          auto F64 = mlir::Float64Type::get(&MCtx);
          /* Lower all args; replace arg 0 with the to_mat conversion. */
          llvm::SmallVector<mlir::Value, 8> Args;
          mlir::Value DtV = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute ConvCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_datetime_vec_to_mat"));
          mlir::Value XMat = emitUnreg("matlab.call_builtin", {DtV},
                                       PtrTy, L, {ConvCal});
          Args.push_back(XMat);
          for (size_t i = 1; i < C.Args.size(); ++i)
            if (C.Args[i]) Args.push_back(lowerExpr(*C.Args[i]));
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "plot"));
          return emitUnreg("matlab.call_builtin", Args,
                           mlir::NoneType::get(&MCtx), L, {Cal});
          (void)F64;
        }
      }
    /* Phase 5.4 (cont.): TMW(rowIdx, colSel) read on a timetable
     * binding. Shape menu:
     *   TMW(:, 'colName') | TMW(:, "colName")   -> matlab_timetable
     *                                              with just that column
     *   TMW(idx, :)                              -> matlab_timetable
     *                                              with rows selected
     * timerange() row-subscripting lives in Task 5; this arm rejects
     * args it can't recognise and falls through to the polymorphic
     * indexing path (which would error). */
    if (C.Args.size() == 2 && C.Args[0] && C.Args[1])
      if (auto *N = dynamic_cast<const NameExpr *>(C.Callee))
        if (N->Ref && isTimetableBinding(N->Ref)) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          /* Column-by-name form: TMW(:, 'colName'). */
          if (dynamic_cast<const ColonExpr *>(C.Args[0])) {
            std::string ColName;
            if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[1]))
              ColName = std::string(CL->Value);
            else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[1]))
              ColName = std::string(SL->Value);
            if (!ColName.empty()) {
              mlir::Value Tv = lowerExpr(*C.Callee);
              mlir::Value NameV = emitFieldNameChar(ColName, L);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_timetable_select_var"));
              return emitUnreg("matlab.call_builtin", {Tv, NameV},
                               PtrTy, L, {Cal});
            }
          }
          /* Row-subscript forms: TMW(rowSel, :).
           *   rowSel is a Timerange binding   -> matlab_timetable_select_rows_timerange
           *   rowSel is anything else (mat)   -> matlab_timetable_select_rows_mat
           *                                      (numeric 1-based OR logical, runtime-detected)
           */
          if (dynamic_cast<const ColonExpr *>(C.Args[1])) {
            bool RowIsTimerange = false;
            if (auto *RN = dynamic_cast<const NameExpr *>(C.Args[0]))
              if (RN->Ref && TimerangeBindings.count(RN->Ref))
                RowIsTimerange = true;
            if (auto *RC = dynamic_cast<const CallOrIndex *>(C.Args[0]))
              if (auto *RCN = dynamic_cast<const NameExpr *>(RC->Callee))
                if (RCN->Name == "timerange") RowIsTimerange = true;
            mlir::Value Tv  = lowerExpr(*C.Callee);
            mlir::Value Idx = lowerExpr(*C.Args[0]);
            const char *Callee = RowIsTimerange
                ? "matlab_timetable_select_rows_timerange"
                : "matlab_timetable_select_rows_mat";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call_builtin", {Tv, Idx},
                             PtrTy, L, {Cal});
          }
        }
    /* Phase 4: m(k) read on a dict binding. Detect via DictBindings,
     * dispatch to matlab_dict_get_<str|num>_<f64|mat>. The expected
     * value type comes from the call site's expected type RT (ptr =
     * matrix, anything else = f64). CharLiteral / StringLiteral
     * keys are coerced to matlab_string* via matlab_string_from_literal. */
    if (C.Args.size() == 1 && C.Args[0])
      if (auto *N = dynamic_cast<const NameExpr *>(C.Callee))
        if (N->Ref && DictBindings.count(N->Ref)) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::Value D = lowerExpr(*C.Callee);
          const Expr *KeyExpr = C.Args[0];
          mlir::Value K;
          bool KeyIsStr = false;
          if (auto *CL = dynamic_cast<const CharLiteral *>(KeyExpr)) {
            mlir::NamedAttribute VA(
                mlir::StringAttr::get(&MCtx, "value"),
                mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
            mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                        mlir::NoneType::get(&MCtx), L, {VA});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
            K = emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
            KeyIsStr = true;
          } else {
            K = lowerExpr(*KeyExpr);
            KeyIsStr = K && (K.getType() == PtrTy || isStringExpr(KeyExpr));
          }
          bool WantMat = mlir::isa<mlir::RankedTensorType,
                                    mlir::UnrankedTensorType>(RT);
          std::string Callee = "matlab_dict_get_";
          Callee += KeyIsStr ? "str_" : "num_";
          Callee += WantMat ? "mat" : "f64";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          mlir::Type ResTy = WantMat ? (mlir::Type)PtrTy : (mlir::Type)F64;
          return emitUnreg("matlab.call_builtin", {D, K}, ResTy, L, {Cal});
        }
    /* Phase 6 — Symbolic Math Toolbox dispatch. Recognise direct calls
     * to the MATLAB-named sym builtins and route to the matlab_sym_*
     * runtime. Type-overloaded calls (diff / int / double / disp / +
     * etc.) are dispatched separately based on operand kind. */
    if (auto *NS = dynamic_cast<const NameExpr *>(C.Callee)) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      auto F64Ty = mlir::Float64Type::get(&MCtx);
      auto isSymExpr = [&](const Expr *X) -> bool {
        return exprIsSym(X);
      };
      auto emitSymCall = [&](llvm::StringRef Callee,
                              llvm::ArrayRef<mlir::Value> Args,
                              mlir::Type Res = {}) -> mlir::Value {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee.str()));
        return emitUnreg("matlab.call_builtin", Args,
                         Res ? Res : (mlir::Type)PtrTy, L, {Cal});
      };
      auto emitConstStr = [&](std::string_view s) -> mlir::Value {
        return emitFieldNameChar(s, L);
      };
      const auto &Nm = NS->Name;

      /* str2sym('expr') — argument must be a string literal at parse time. */
      if (Nm == "str2sym" && C.Args.size() == 1 && C.Args[0]) {
        if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[0])) {
          mlir::Value SV = emitConstStr(CL->Value);
          return emitSymCall("matlab_sym_str2sym", {SV});
        }
        if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[0])) {
          mlir::Value SV = emitConstStr(SL->Value);
          return emitSymCall("matlab_sym_str2sym", {SV});
        }
      }
      /* sym('expr') / sym("expr") / sym(numeric) / sym(name). String
       * argument routes through matlab_sym_from_str; numeric → from_double. */
      if (Nm == "sym" && C.Args.size() == 1 && C.Args[0]) {
        if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[0]))
          return emitSymCall("matlab_sym_from_str", {emitConstStr(CL->Value)});
        if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[0]))
          return emitSymCall("matlab_sym_from_str", {emitConstStr(SL->Value)});
        /* Numeric argument → matlab_sym_from_double. */
        mlir::Value V = lowerExpr(*C.Args[0]);
        if (V && V.getType() == F64Ty)
          return emitSymCall("matlab_sym_from_double", {V});
      }
      /* simplify / expand / clearAssumptions — single-sym-arg → sym. */
      if ((Nm == "simplify" || Nm == "expand" || Nm == "clearAssumptions") &&
          C.Args.size() == 1 && C.Args[0] && isSymExpr(C.Args[0])) {
        std::string Callee = "matlab_sym_" + std::string(Nm);
        mlir::Value V = lowerExpr(*C.Args[0]);
        return emitSymCall(Callee, {V});
      }
      /* Elementary functions on sym — sin / cos / tan / etc. The numeric
       * matrix lowering would route to matlab_sin_m / cos_m / etc. for a
       * matlab_mat*; here we override when the operand is a sym so we
       * call matlab_sym_<name> instead. */
      {
        static const llvm::StringSet<> Elementary = {
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh",
            "exp", "log", "sqrt", "abs"};
        if (Elementary.contains(Nm) && C.Args.size() == 1 &&
            C.Args[0] && isSymExpr(C.Args[0])) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          std::string Callee = "matlab_sym_" + std::string(Nm);
          return emitSymCall(Callee, {V});
        }
      }
      /* assume(x, "prop") / assumeAlso(x, "prop") — sym + char-literal.
       * MATLAB semantics: the side-effect applies to the named symbol
       * for future references. SymPP returns a fresh sym carrying the
       * registered mask; we rebind it back to the original name so
       * `simplify` / `refine` downstream sees the new (masked) symbol.
       *
       * The property argument must be a string literal at parse time
       * so the const_char lowering can flow through (same shape as
       * str2sym). */
      if ((Nm == "assume" || Nm == "assumeAlso") &&
          C.Args.size() == 2 && C.Args[0] && C.Args[1] &&
          isSymExpr(C.Args[0])) {
        std::string Callee = "matlab_sym_" + std::string(Nm);
        mlir::Value SymV = lowerExpr(*C.Args[0]);
        mlir::Value PropV;
        if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[1]))
          PropV = emitConstStr(CL->Value);
        else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[1]))
          PropV = emitConstStr(SL->Value);
        if (PropV) {
          mlir::Value Fresh = emitSymCall(Callee, {SymV, PropV});
          /* Rebind the fresh sym onto the original name so subsequent
           * reads pick up the assumption mask. The arg must be a
           * NameExpr for MATLAB's `assume(x, ...)` shape — anything
           * else (a sub-expression) doesn't have a stable name to
           * write back to. */
          if (auto *NE0 = dynamic_cast<const NameExpr *>(C.Args[0])) {
            mlir::Value NameV = emitFieldNameChar(NE0->Name, L);
            mlir::NamedAttribute WsCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_ws_set_sym"));
            emitUnregOp("matlab.call_builtin", {NameV, Fresh},
                        {mlir::NoneType::get(&MCtx)}, L, {WsCal});
            /* Also update the local slot if one exists so non-REPL
             * scripts see the rebinding too. */
            if (NE0->Ref) {
              auto It = Slots.find(NE0->Ref);
              if (It != Slots.end()) emitStore(Fresh, It->second, L);
            }
          }
          return Fresh;
        }
      }
      /* clearAssumptions(x) — same rebinding shape, no property arg. */
      if (Nm == "clearAssumptions" && C.Args.size() == 1 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value SymV = lowerExpr(*C.Args[0]);
        mlir::Value Fresh = emitSymCall("matlab_sym_clearAssumptions", {SymV});
        if (auto *NE0 = dynamic_cast<const NameExpr *>(C.Args[0])) {
          mlir::Value NameV = emitFieldNameChar(NE0->Name, L);
          mlir::NamedAttribute WsCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_set_sym"));
          emitUnregOp("matlab.call_builtin", {NameV, Fresh},
                      {mlir::NoneType::get(&MCtx)}, L, {WsCal});
          if (NE0->Ref) {
            auto It = Slots.find(NE0->Ref);
            if (It != Slots.end()) emitStore(Fresh, It->second, L);
          }
        }
        return Fresh;
      }
      /* assumptions(x) — returns a string (matlab_string*-shaped sym).
       * Phase A: the C ABI returns a malloc'd char*; we return a sym
       * for type uniformity at the language level. */
      if (Nm == "assumptions" && C.Args.size() == 1 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value V = lowerExpr(*C.Args[0]);
        /* Returns a char* (raw); for now, route through an opaque ptr
         * — language-level use is tested via matlab_sym_assumptions
         * directly. Skip for now and let it fall through. */
        (void)V;
      }
      /* vpa(e, dps) — variable-precision evaluation. */
      if (Nm == "vpa" && C.Args.size() >= 1 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value V = lowerExpr(*C.Args[0]);
        mlir::Value Dps;
        if (C.Args.size() >= 2 && C.Args[1])
          Dps = lowerExpr(*C.Args[1]);
        else
          Dps = mlir::arith::ConstantOp::create(
              B, L, mlir::Float64Type::get(&MCtx),
              mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), 32.0));
        /* matlab_sym_vpa takes (sym*, i64). Cast f64 → i64. */
        if (Dps && Dps.getType() == F64Ty)
          Dps = mlir::arith::FPToSIOp::create(
              B, L, mlir::IntegerType::get(&MCtx, 64), Dps);
        return emitSymCall("matlab_sym_vpa", {V, Dps});
      }
      /* taylor(f, x, a, n). MATLAB's signature is taylor(f, x, a, 'Order',n)
       * but the simpler 4-arg form is more common. */
      if (Nm == "taylor" && C.Args.size() == 4 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value F = lowerExpr(*C.Args[0]);
        mlir::Value Vv = lowerExpr(*C.Args[1]);
        mlir::Value Av = lowerExpr(*C.Args[2]);
        mlir::Value Nv = lowerExpr(*C.Args[3]);
        if (Av && Av.getType() == F64Ty)
          Av = emitSymCall("matlab_sym_from_double", {Av});
        if (Nv && Nv.getType() == F64Ty)
          Nv = mlir::arith::FPToSIOp::create(
              B, L, mlir::IntegerType::get(&MCtx, 64), Nv);
        return emitSymCall("matlab_sym_taylor", {F, Vv, Av, Nv});
      }
      /* limit(f, x, target). */
      if (Nm == "limit" && C.Args.size() == 3 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value F = lowerExpr(*C.Args[0]);
        mlir::Value Vv = lowerExpr(*C.Args[1]);
        mlir::Value Tg = lowerExpr(*C.Args[2]);
        if (Tg && Tg.getType() == F64Ty)
          Tg = emitSymCall("matlab_sym_from_double", {Tg});
        return emitSymCall("matlab_sym_limit", {F, Vv, Tg});
      }
      /* dsolve(eq, y, yp, x) / dsolve_2(eq, y, yp, ypp, x). */
      if (Nm == "dsolve" && C.Args.size() == 4 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value E = lowerExpr(*C.Args[0]);
        mlir::Value Y = lowerExpr(*C.Args[1]);
        mlir::Value Yp = lowerExpr(*C.Args[2]);
        mlir::Value X = lowerExpr(*C.Args[3]);
        return emitSymCall("matlab_sym_dsolve", {E, Y, Yp, X});
      }
      if (Nm == "dsolve" && C.Args.size() == 5 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value E = lowerExpr(*C.Args[0]);
        mlir::Value Y = lowerExpr(*C.Args[1]);
        mlir::Value Yp = lowerExpr(*C.Args[2]);
        mlir::Value Ypp = lowerExpr(*C.Args[3]);
        mlir::Value X = lowerExpr(*C.Args[4]);
        return emitSymCall("matlab_sym_dsolve_2", {E, Y, Yp, Ypp, X});
      }
      /* pdsolve(a, b, c, x, y). */
      if (Nm == "pdsolve" && C.Args.size() == 5 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value A = lowerExpr(*C.Args[0]);
        mlir::Value B2 = lowerExpr(*C.Args[1]);
        mlir::Value Cc = lowerExpr(*C.Args[2]);
        mlir::Value X = lowerExpr(*C.Args[3]);
        mlir::Value Y = lowerExpr(*C.Args[4]);
        return emitSymCall("matlab_sym_pdsolve", {A, B2, Cc, X, Y});
      }
      if (Nm == "pdsolve_heat" && C.Args.size() == 4 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value K2 = lowerExpr(*C.Args[0]);
        mlir::Value Lam = lowerExpr(*C.Args[1]);
        mlir::Value X = lowerExpr(*C.Args[2]);
        mlir::Value T = lowerExpr(*C.Args[3]);
        return emitSymCall("matlab_sym_pdsolve_heat", {K2, Lam, X, T});
      }
      if (Nm == "pdsolve_wave" && C.Args.size() == 3 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value Cc = lowerExpr(*C.Args[0]);
        mlir::Value X = lowerExpr(*C.Args[1]);
        mlir::Value T = lowerExpr(*C.Args[2]);
        return emitSymCall("matlab_sym_pdsolve_wave", {Cc, X, T});
      }
      /* Integral transforms — all (f, var1, var2) → sym. */
      if ((Nm == "laplace" || Nm == "ilaplace" ||
           Nm == "fourier" || Nm == "ifourier" ||
           Nm == "ztrans" || Nm == "iztrans") &&
          C.Args.size() == 3 && C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value F = lowerExpr(*C.Args[0]);
        mlir::Value V1 = lowerExpr(*C.Args[1]);
        mlir::Value V2 = lowerExpr(*C.Args[2]);
        std::string Callee = "matlab_sym_" + std::string(Nm);
        return emitSymCall(Callee, {F, V1, V2});
      }
      /* nsolve(eq, var, x0, dps) / vpasolve(...) — Newton + variable-
       * precision numeric solve. dps optional, defaults to 15 / 32. */
      if ((Nm == "nsolve" || Nm == "vpasolve") && C.Args.size() >= 3 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value E = lowerExpr(*C.Args[0]);
        mlir::Value V = lowerExpr(*C.Args[1]);
        mlir::Value X0 = lowerExpr(*C.Args[2]);
        if (X0 && X0.getType() == F64Ty)
          X0 = emitSymCall("matlab_sym_from_double", {X0});
        mlir::Value Dps;
        if (C.Args.size() >= 4 && C.Args[3])
          Dps = lowerExpr(*C.Args[3]);
        else
          Dps = mlir::arith::ConstantOp::create(
              B, L, mlir::Float64Type::get(&MCtx),
              mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx),
                                     Nm == "vpasolve" ? 32.0 : 15.0));
        if (Dps && Dps.getType() == F64Ty)
          Dps = mlir::arith::FPToSIOp::create(
              B, L, mlir::IntegerType::get(&MCtx, 64), Dps);
        std::string Callee = "matlab_sym_" + std::string(Nm);
        return emitSymCall(Callee, {E, V, X0, Dps});
      }
      /* checkodesol(eq, sol, y, yp, x) → sym (residual). */
      if (Nm == "checkodesol" && C.Args.size() == 5 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value E = lowerExpr(*C.Args[0]);
        mlir::Value S = lowerExpr(*C.Args[1]);
        mlir::Value Y = lowerExpr(*C.Args[2]);
        mlir::Value Yp = lowerExpr(*C.Args[3]);
        mlir::Value X = lowerExpr(*C.Args[4]);
        return emitSymCall("matlab_sym_checkodesol", {E, S, Y, Yp, X});
      }
      /* dsolve_ivp(eq, y, yp, x, x0, y0) — first-order with one IC.
       * Multi-condition shape: dsolve_ivp(eq, y, yp, x, [x0, x1, ...],
       * [y0, y1, ...]) where the last two args are 1-row MatrixLiterals
       * of equal length. Routes to the runtime's variadic
       * matlab_sym_dsolve_ivp by building two parallel stack arrays. */
      auto buildSymArr = [&](llvm::SmallVectorImpl<mlir::Value> &vals) -> mlir::Value {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        auto I64 = mlir::IntegerType::get(&MCtx, 64);
        auto ArrTy = mlir::LLVM::LLVMArrayType::get(PtrTy, vals.size());
        mlir::Value One = mlir::LLVM::ConstantOp::create(
            B, L, I64, mlir::IntegerAttr::get(I64, 1)).getResult();
        mlir::Value Arr = mlir::LLVM::AllocaOp::create(
            B, L, PtrTy, ArrTy, One, /*alignment=*/0).getResult();
        for (size_t k = 0; k < vals.size(); ++k) {
          mlir::Value V = vals[k];
          if (V.getType() != PtrTy) V.setType(PtrTy);
          mlir::Value Z = mlir::LLVM::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, 0)).getResult();
          mlir::Value Idx = mlir::LLVM::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, (int)k)).getResult();
          auto Gep = mlir::LLVM::GEPOp::create(
              B, L, PtrTy, ArrTy, Arr, mlir::ValueRange{Z, Idx});
          mlir::LLVM::StoreOp::create(B, L, V, Gep);
        }
        return Arr;
      };
      auto i64ConstA = [&](int64_t v) -> mlir::Value {
        auto I64 = mlir::IntegerType::get(&MCtx, 64);
        return mlir::LLVM::ConstantOp::create(
            B, L, I64, mlir::IntegerAttr::get(I64, v)).getResult();
      };
      if (Nm == "dsolve_ivp" && C.Args.size() == 6 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        /* Detect multi-condition: args 4 and 5 are MatrixLiterals. */
        auto *XML = dynamic_cast<const MatrixLiteral *>(C.Args[4]);
        auto *YML = dynamic_cast<const MatrixLiteral *>(C.Args[5]);
        if (XML && YML &&
            XML->Rows.size() == 1 && YML->Rows.size() == 1 &&
            XML->Rows[0].size() == YML->Rows[0].size() &&
            XML->Rows[0].size() >= 1) {
          mlir::Value E = lowerExpr(*C.Args[0]);
          mlir::Value Y = lowerExpr(*C.Args[1]);
          mlir::Value Yp = lowerExpr(*C.Args[2]);
          mlir::Value X = lowerExpr(*C.Args[3]);
          llvm::SmallVector<mlir::Value, 4> Xs, Ys;
          for (const Expr *Xi : XML->Rows[0]) Xs.push_back(lowerExpr(*Xi));
          for (const Expr *Yi : YML->Rows[0]) Ys.push_back(lowerExpr(*Yi));
          return emitSymCall("matlab_sym_dsolve_ivp",
                             {E, Y, Yp, X, i64ConstA((int64_t)Xs.size()),
                              buildSymArr(Xs), buildSymArr(Ys)});
        }
        llvm::SmallVector<mlir::Value, 6> A;
        for (auto *X : C.Args) A.push_back(lowerExpr(*X));
        return emitSymCall("matlab_sym_dsolve_ivp_1", A);
      }
      /* apply_ivp(general_solution, x, x0, y0) — single-cond.
       * Multi-cond shape: apply_ivp(general, x, [x0, x1, ...], [y0, y1, ...]). */
      if (Nm == "apply_ivp" && C.Args.size() == 4 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        auto *XML = dynamic_cast<const MatrixLiteral *>(C.Args[2]);
        auto *YML = dynamic_cast<const MatrixLiteral *>(C.Args[3]);
        if (XML && YML &&
            XML->Rows.size() == 1 && YML->Rows.size() == 1 &&
            XML->Rows[0].size() == YML->Rows[0].size() &&
            XML->Rows[0].size() >= 1) {
          mlir::Value G = lowerExpr(*C.Args[0]);
          mlir::Value X = lowerExpr(*C.Args[1]);
          llvm::SmallVector<mlir::Value, 4> Xs, Ys;
          for (const Expr *Xi : XML->Rows[0]) Xs.push_back(lowerExpr(*Xi));
          for (const Expr *Yi : YML->Rows[0]) Ys.push_back(lowerExpr(*Yi));
          return emitSymCall("matlab_sym_apply_ivp",
                             {G, X, i64ConstA((int64_t)Xs.size()),
                              buildSymArr(Xs), buildSymArr(Ys)});
        }
        llvm::SmallVector<mlir::Value, 4> A;
        for (auto *X : C.Args) A.push_back(lowerExpr(*X));
        return emitSymCall("matlab_sym_apply_ivp_1", A);
      }
      /* --- Phase 6.1 symbolic-matrix builtins. -------------------------
       * sym_matrix(R, C, e11, e12, ..., eRC) — construct a symbolic
       * matrix from scalar sym entries. R and C must be integer
       * literals so the row-major flattening is resolved at compile
       * time. Emits sym_matrix_zeros(R, C) followed by R*C set calls.
       * Result is a matlab_symmat*.
       *
       * This bypasses the standard `[a 1; 2 b]` matrix-literal syntax —
       * extending matrix literals to detect sym entries is bigger work
       * (the literal lowering currently routes through the f64 path).
       * `sym_matrix` gives users an explicit constructor in the
       * meantime; same shape as `containers.Map(...)`. */
      auto foldI64 = [&](const Expr *X, int64_t &Out) -> bool {
        if (!X) return false;
        if (auto *IL = dynamic_cast<const IntegerLiteral *>(X)) {
          Out = std::strtoll(std::string(IL->Text).c_str(), nullptr, 10);
          return true;
        }
        if (auto *FL = dynamic_cast<const FPLiteral *>(X)) {
          Out = static_cast<int64_t>(
              std::strtod(std::string(FL->Text).c_str(), nullptr));
          return true;
        }
        return false;
      };
      auto I64Ty = mlir::IntegerType::get(&MCtx, 64);
      auto i64Const = [&](int64_t v) {
        return mlir::arith::ConstantOp::create(
            B, L, I64Ty, mlir::IntegerAttr::get(I64Ty, v)).getResult();
      };
      if (Nm == "sym_matrix" && C.Args.size() >= 2) {
        int64_t R = 0, Cc = 0;
        if (!foldI64(C.Args[0], R) || !foldI64(C.Args[1], Cc))
          return mlir::Value{};
        if (static_cast<int64_t>(C.Args.size()) != 2 + R * Cc)
          return mlir::Value{};
        mlir::Value M = emitSymCall("matlab_symmat_zeros",
                                      {i64Const(R), i64Const(Cc)});
        for (int64_t i = 0; i < R; ++i)
          for (int64_t j = 0; j < Cc; ++j) {
            mlir::Value V = lowerExpr(*C.Args[2 + i * Cc + j]);
            if (V && V.getType() == F64Ty)
              V = emitSymCall("matlab_sym_from_double", {V});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_symmat_set"));
            emitUnregOp("matlab.call_builtin",
                        {M, i64Const(i), i64Const(j), V},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
          }
        return M;
      }
      if ((Nm == "sym_eye" || Nm == "sym_zeros") && C.Args.size() >= 1) {
        if (Nm == "sym_eye" && C.Args.size() == 1) {
          mlir::Value N = lowerExpr(*C.Args[0]);
          if (N && N.getType() == F64Ty)
            N = mlir::arith::FPToSIOp::create(B, L, I64Ty, N);
          return emitSymCall("matlab_symmat_eye", {N});
        }
        if (Nm == "sym_zeros" && C.Args.size() == 2) {
          mlir::Value R = lowerExpr(*C.Args[0]);
          mlir::Value Cc2 = lowerExpr(*C.Args[1]);
          if (R && R.getType() == F64Ty)
            R = mlir::arith::FPToSIOp::create(B, L, I64Ty, R);
          if (Cc2 && Cc2.getType() == F64Ty)
            Cc2 = mlir::arith::FPToSIOp::create(B, L, I64Ty, Cc2);
          return emitSymCall("matlab_symmat_zeros", {R, Cc2});
        }
      }
      /* sym_det / sym_inv / sym_transpose / sym_trace / sym_rank — single-
       * matrix operations. Result is sym (det/trace) or symmat. */
      if ((Nm == "sym_det" || Nm == "sym_trace") && C.Args.size() == 1) {
        mlir::Value M = lowerExpr(*C.Args[0]);
        std::string Callee = "matlab_symmat_" + std::string(Nm).substr(4);
        return emitSymCall(Callee, {M});
      }
      if ((Nm == "sym_inv" || Nm == "sym_transpose") && C.Args.size() == 1) {
        mlir::Value M = lowerExpr(*C.Args[0]);
        std::string Callee = "matlab_symmat_" +
            std::string(Nm == "sym_inv" ? "inverse" : "transpose");
        return emitSymCall(Callee, {M});
      }
      if (Nm == "sym_rank" && C.Args.size() == 1) {
        mlir::Value M = lowerExpr(*C.Args[0]);
        return emitSymCall("matlab_symmat_rank", {M}, I64Ty);
      }
      /* sym_linsolve(A, b) — A·x = b. Returns symmat column. */
      if (Nm == "sym_linsolve" && C.Args.size() == 2) {
        mlir::Value A = lowerExpr(*C.Args[0]);
        mlir::Value Bv = lowerExpr(*C.Args[1]);
        return emitSymCall("matlab_symmat_linsolve", {A, Bv});
      }
      /* sym_dsolve_system(A, x) — y' = A·y. Returns symmat. */
      if (Nm == "sym_dsolve_system" && C.Args.size() == 2 &&
          C.Args[1] && isSymExpr(C.Args[1])) {
        mlir::Value A = lowerExpr(*C.Args[0]);
        mlir::Value X = lowerExpr(*C.Args[1]);
        return emitSymCall("matlab_symmat_dsolve_system", {A, X});
      }
      /* sym_solve_2x2 / sym_solve_3x3 — fixed-arity multi-equation
       * solve. Returns a symmat with one row per joint solution and
       * one column per variable. The variadic-array form
       * (sym_solve_sys) ships in the runtime but the language-level
       * lowering for it lands in Phase 6.2 — until then, callers use
       * these explicit small-system entries. */
      if (Nm == "sym_solve_2x2" && C.Args.size() == 4) {
        llvm::SmallVector<mlir::Value, 4> A;
        for (auto *X : C.Args) A.push_back(lowerExpr(*X));
        return emitSymCall("matlab_sym_solve_2x2", A);
      }
      if (Nm == "sym_solve_3x3" && C.Args.size() == 6) {
        llvm::SmallVector<mlir::Value, 6> A;
        for (auto *X : C.Args) A.push_back(lowerExpr(*X));
        return emitSymCall("matlab_sym_solve_3x3", A);
      }
      /* Phase 6.2 — variadic sym_solve_sys for systems of any size.
       * Shape: sym_solve_sys([eq1, eq2, ...], [x1, x2, ...]) where
       * each argument must be a 1-row MatrixLiteral of sym entries.
       * Lowers to: alloca [N x ptr] for eqs, fill, alloca [M x ptr]
       * for vars, fill, call matlab_sym_solve_sys(eqs, N, vars, M).
       *
       * Each sym arg already lowers to !llvm.ptr; we materialise the
       * arrays as llvm.alloca + per-element llvm.getelementptr +
       * llvm.store, then pass the array base pointers. */
      if (Nm == "sym_solve_sys" && C.Args.size() == 2 && C.Args[0] && C.Args[1]) {
        auto *EqsML = dynamic_cast<const MatrixLiteral *>(C.Args[0]);
        auto *VarsML = dynamic_cast<const MatrixLiteral *>(C.Args[1]);
        if (EqsML && VarsML &&
            EqsML->Rows.size() == 1 && VarsML->Rows.size() == 1) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          auto I32 = mlir::IntegerType::get(&MCtx, 32);
          auto I64 = mlir::IntegerType::get(&MCtx, 64);
          auto i32Const = [&](int v) -> mlir::Value {
            return mlir::LLVM::ConstantOp::create(
                B, L, I32, mlir::IntegerAttr::get(I32, v)).getResult();
          };
          auto i64Const = [&](int64_t v) -> mlir::Value {
            return mlir::LLVM::ConstantOp::create(
                B, L, I64, mlir::IntegerAttr::get(I64, v)).getResult();
          };
          /* Build a stack array of `count` ptr slots, fill from `vals`,
           * return the base pointer. The `vals` may have NoneType
           * placeholders from upstream lowering paths that haven't
           * settled — coerce by setting the value's type to PtrTy
           * since downstream we know they're sym ptrs. */
          auto buildArr = [&](llvm::SmallVectorImpl<mlir::Value> &vals) -> mlir::Value {
            auto ArrTy = mlir::LLVM::LLVMArrayType::get(PtrTy, vals.size());
            mlir::Value One = i64Const(1);
            mlir::Value Arr = mlir::LLVM::AllocaOp::create(
                B, L, PtrTy, ArrTy, One, /*alignment=*/0).getResult();
            for (size_t k = 0; k < vals.size(); ++k) {
              mlir::Value V = vals[k];
              if (V.getType() != PtrTy) V.setType(PtrTy);
              auto Gep = mlir::LLVM::GEPOp::create(
                  B, L, PtrTy, ArrTy, Arr,
                  mlir::ValueRange{i32Const(0), i32Const((int)k)});
              mlir::LLVM::StoreOp::create(B, L, V, Gep);
            }
            return Arr;
          };
          llvm::SmallVector<mlir::Value, 4> Eqs;
          for (const Expr *X : EqsML->Rows[0]) Eqs.push_back(lowerExpr(*X));
          llvm::SmallVector<mlir::Value, 4> Vars;
          for (const Expr *X : VarsML->Rows[0]) Vars.push_back(lowerExpr(*X));
          mlir::Value EqArr = buildArr(Eqs);
          mlir::Value VarArr = buildArr(Vars);
          return emitSymCall("matlab_sym_solve_sys",
                             {EqArr, i64Const((int64_t)Eqs.size()),
                              VarArr, i64Const((int64_t)Vars.size())});
        }
      }
      /* factor(expr, var). */
      if (Nm == "factor" && C.Args.size() == 2 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value E = lowerExpr(*C.Args[0]);
        mlir::Value V = lowerExpr(*C.Args[1]);
        return emitSymCall("matlab_sym_factor", {E, V});
      }
      /* subs(expr, old, new). */
      if (Nm == "subs" && C.Args.size() == 3 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value Ea = lowerExpr(*C.Args[0]);
        mlir::Value Eo = lowerExpr(*C.Args[1]);
        mlir::Value En = lowerExpr(*C.Args[2]);
        /* If `new` is a numeric literal, box it into a sym first. */
        if (En && En.getType() == F64Ty)
          En = emitSymCall("matlab_sym_from_double", {En});
        return emitSymCall("matlab_sym_subs", {Ea, Eo, En});
      }
      /* solve(eq, var). Routes to the single-root variant for Phase A. */
      if (Nm == "solve" && C.Args.size() == 2 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value Eq = lowerExpr(*C.Args[0]);
        mlir::Value V = lowerExpr(*C.Args[1]);
        return emitSymCall("matlab_sym_solve_one", {Eq, V});
      }
      /* diff(f, x) / diff(f, x, n) — sym overload. */
      if (Nm == "diff" && C.Args.size() >= 2 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value F = lowerExpr(*C.Args[0]);
        mlir::Value Vv = lowerExpr(*C.Args[1]);
        if (C.Args.size() == 2)
          return emitSymCall("matlab_sym_diff", {F, Vv});
        if (C.Args.size() == 3) {
          mlir::Value Nv = lowerExpr(*C.Args[2]);
          if (Nv && Nv.getType() == F64Ty)
            Nv = mlir::arith::FPToSIOp::create(
                B, L, mlir::IntegerType::get(&MCtx, 64), Nv);
          return emitSymCall("matlab_sym_diff_n", {F, Vv, Nv});
        }
      }
      /* int(f, x) / int(f, x, a, b) — sym overload. */
      if (Nm == "int" && (C.Args.size() == 2 || C.Args.size() == 4) &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value F = lowerExpr(*C.Args[0]);
        mlir::Value Vv = lowerExpr(*C.Args[1]);
        if (C.Args.size() == 2)
          return emitSymCall("matlab_sym_int", {F, Vv});
        mlir::Value Aa = lowerExpr(*C.Args[2]);
        mlir::Value Bb = lowerExpr(*C.Args[3]);
        if (Aa && Aa.getType() == F64Ty)
          Aa = emitSymCall("matlab_sym_from_double", {Aa});
        if (Bb && Bb.getType() == F64Ty)
          Bb = emitSymCall("matlab_sym_from_double", {Bb});
        return emitSymCall("matlab_sym_int_def", {F, Vv, Aa, Bb});
      }
      /* double(s) — numeric eval of a sym. Returns f64. */
      if (Nm == "double" && C.Args.size() == 1 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value V = lowerExpr(*C.Args[0]);
        return emitSymCall("matlab_sym_double", {V}, F64Ty);
      }
      /* disp(s) — user wrote disp(sym) explicitly. Mirrors the bare-
       * expression disp dispatch above; routes to matlab_sym_disp so
       * the value is pretty-printed via SymPP rather than f64-formatted
       * via matlab_disp_*. Returns void. */
      if (Nm == "disp" && C.Args.size() == 1 &&
          C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value V = lowerExpr(*C.Args[0]);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_sym_disp"));
        return emitUnreg("matlab.call_builtin", {V},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      /* disp(symmat) — same shape, different runtime entry. */
      if (Nm == "disp" && C.Args.size() == 1 &&
          C.Args[0] && exprIsSymmat(C.Args[0])) {
        mlir::Value V = lowerExpr(*C.Args[0]);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_symmat_disp"));
        return emitUnreg("matlab.call_builtin", {V},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      /* #156: disp(struct) / disp(cell). These ptrs would otherwise hit the
       * polymorphic matlab_disp_mat path, which reads them as a matrix
       * descriptor and SIGSEGVs. Route a struct- or cell-bound argument to a
       * dedicated display entry. Detected via the binding tags (IsStruct /
       * StructBindings / CellBindings). */
      if (Nm == "disp" && C.Args.size() == 1) {
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref && (AN->Ref->IsStruct || StructBindings.count(AN->Ref) ||
                          StructInitialised.count(AN->Ref))) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_disp_struct"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (AN->Ref && CellBindings.count(AN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_disp_cell"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
        }
      }
      /* latex / pretty / ccode — char-returning printers. Result is a
       * matlab_string* (matches matlab_num2str shape). */
      if ((Nm == "latex" || Nm == "pretty" || Nm == "ccode") &&
          C.Args.size() == 1 && C.Args[0] && isSymExpr(C.Args[0])) {
        mlir::Value V = lowerExpr(*C.Args[0]);
        std::string Callee = "matlab_sym_" + std::string(Nm);
        return emitSymCall(Callee, {V});
      }
    }
    if (C.Resolved == CallKind::Call) {
      auto *N = dynamic_cast<const NameExpr *>(C.Callee);
      auto PtrTyConst = mlir::LLVM::LLVMPointerType::get(&MCtx);
      /* ===== Robotics factory functions returning a fresh rigidBodyTree =====
       * `importrobot(file)` / `loadrobot(name)` allocate a rigidBodyTree
       * shell and populate it from a URDF file / baked model.  These are
       * free functions (Builtin binding), so they're intercepted here
       * before the generic builtin dispatch. */
      if (N && (N->Name == "importrobot" || N->Name == "loadrobot") &&
          C.Args.size() == 1) {
        mlir::NamedAttribute CtorCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "rigidBodyTree__rigidBodyTree"));
        mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
        mlir::Value Arg = lowerExpr(*C.Args[0]);
        const char *rt = (N->Name == "importrobot") ? "matlab_robotics_importrobot"
                                                     : "matlab_robotics_loadrobot";
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, rt));
        emitUnregOp("matlab.call_builtin", {Obj, Arg},
                    {mlir::NoneType::get(&MCtx)}, L, {Cal});
        return Obj;
      }
      /* Reinforcement Learning Tier 1 — rlPredefinedEnv("BasicGridWorld").
       * A free function returning an rlMDPEnv carrier; the runtime fills the
       * grid-world transition/reward tensors.  (Only BasicGridWorld ships in
       * T1; other predefined names are a documented carve to later tiers.) */
      if (N && N->Name == "rlPredefinedEnv" && C.Args.size() == 1) {
        std::string envName;
        if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[0])) envName = CL->Value;
        else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[0])) envName = SL->Value;
        bool isCartPole = envName.find("CartPole") != std::string::npos ||
                          envName.find("cartpole") != std::string::npos;
        bool isPendulum = envName.find("Pendulum") != std::string::npos ||
                          envName.find("pendulum") != std::string::npos;
        bool isCountdown = envName.find("Countdown") != std::string::npos ||
                           envName.find("countdown") != std::string::npos;
        mlir::NamedAttribute CtorCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "rlMDPEnv__rlMDPEnv"));
        mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
        const char *rt = isPendulum ? "matlab_rl_pendulum_init"
                       : isCartPole ? "matlab_rl_cartpole_init"
                       : isCountdown ? "matlab_rl_countdown_init"
                                     : "matlab_rl_gridworld_init";
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, rt));
        emitUnregOp("matlab.call_builtin", {Obj},
                    {mlir::NoneType::get(&MCtx)}, L, {Cal});
        return Obj;
      }
      /* Constructor call: `ClassName(args)` where ClassName resolves to
       * a user classdef. Route to the emitted `ClassName__ClassName`
       * function, returning a matlab_obj*. If the class has no explicit
       * constructor, emit `matlab_obj_new(class_id)` directly and skip
       * arg-binding — MATLAB's implicit default constructor is no-arg. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Class &&
          N->Ref->ClassDef) {
        const ClassDef *CD = N->Ref->ClassDef;
        /* Global Optimization Tier-6 — optimoptions(solver, 'Name', val,
         * ...).  `optimoptions` is a classdef, so it lands here (not in
         * the builtin block).  Allocate the zero-arg carrier shell and
         * write the named fields (scalars via _set_f64; IntCon via
         * _set_mat); the leading solver-name string is skipped. */
        if (CD->Name == "optimoptions" && C.Args.size() >= 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "optimoptions__optimoptions"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          for (size_t i = 1; i + 1 < C.Args.size(); i += 2) {
            std::string key;
            if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[i])) key = CL->Value;
            else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[i])) key = SL->Value;
            if (key.empty()) continue;
            bool isMat = (key == "IntCon");
            mlir::Value Val = lowerExpr(*C.Args[i + 1]);
            mlir::Value NameV = emitFieldNameChar(key, L);
            mlir::NamedAttribute SetCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, isMat ? "matlab_obj_set_mat"
                                                   : "matlab_obj_set_f64"));
            emitUnregOp("matlab.call_builtin", {Obj, NameV, Val},
                        {mlir::NoneType::get(&MCtx)}, L, {SetCal});
          }
          return Obj;
        }
        /* Curve Fitting Tier-2 — fitoptions('Name', val, …).  A classdef
         * carrier; scan name-value pairs from index 0.  StartPoint / Lower /
         * Upper / Weights are matrices (_set_mat); Robust maps a string to a
         * RobustCode scalar (_set_f64); Method is accepted-and-ignored. */
        if (CD->Name == "fitoptions") {
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "fitoptions__fitoptions"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          for (size_t i = 0; i + 1 < C.Args.size(); i += 2) {
            std::string key;
            if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[i])) key = CL->Value;
            else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[i])) key = SL->Value;
            if (key.empty() || key == "Method") continue;
            if (key == "Robust") {
              std::string rv;
              if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[i + 1])) rv = CL->Value;
              else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[i + 1])) rv = SL->Value;
              double code = 0.0;
              if (rv == "Bisquare" || rv == "bisquare" || rv == "on" || rv == "On") code = 1.0;
              else if (rv == "LAR" || rv == "lar") code = 2.0;
              mlir::Value CodeV = emitUnreg("matlab.const_float", {}, F64, L,
                  {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                                        mlir::FloatAttr::get(F64, code))});
              mlir::Value NameV = emitFieldNameChar("RobustCode", L);
              mlir::NamedAttribute SetCal(mlir::StringAttr::get(&MCtx, "callee"),
                                          mlir::StringAttr::get(&MCtx, "matlab_obj_set_f64"));
              emitUnregOp("matlab.call_builtin", {Obj, NameV, CodeV},
                          {mlir::NoneType::get(&MCtx)}, L, {SetCal});
              continue;
            }
            bool isMat = (key == "StartPoint" || key == "Lower" ||
                          key == "Upper" || key == "Weights");
            mlir::Value Val = lowerExpr(*C.Args[i + 1]);
            mlir::Value NameV = emitFieldNameChar(key, L);
            mlir::NamedAttribute SetCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, isMat ? "matlab_obj_set_mat"
                                                   : "matlab_obj_set_f64"));
            emitUnregOp("matlab.call_builtin", {Obj, NameV, Val},
                        {mlir::NoneType::get(&MCtx)}, L, {SetCal});
          }
          return Obj;
        }
        /* Curve Fitting Tier-3 — fittype('a*exp(-b*x)+c'): alloc the custom-
         * equation descriptor and store the equation string (const_char →
         * matlab_string by the pde_table coercion). */
        if (CD->Name == "fittype" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "fittype__fittype"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Eq = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_curvefit_fittype_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Eq},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* Image Processing Tier-3 — affine2d(M) / projective2d(M): alloc the
         * transform shell and write its 3×3 forward matrix T (Kind 1/2 is
         * the ctor default). */
        if ((CD->Name == "affine2d" || CD->Name == "projective2d") && C.Args.size() == 1) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value M = lowerExpr(*C.Args[0]);
          mlir::Value NameV = emitFieldNameChar("T", L);
          mlir::NamedAttribute SetCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_obj_set_mat"));
          emitUnregOp("matlab.call_builtin", {Obj, NameV, M},
                      {mlir::NoneType::get(&MCtx)}, L, {SetCal});
          return Obj;
        }
        /* System Identification Tier-5 — recursiveLS(np) / recursiveARX(
         * [na nb nk]).  Alloc-then-populate via the runtime init. */
        if ((CD->Name == "recursiveLS" || CD->Name == "recursiveARX") &&
            C.Args.size() == 1) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Arg = lowerExpr(*C.Args[0]);
          const char *rt = (CD->Name == "recursiveLS") ? "matlab_ident_rls_init"
                                                        : "matlab_ident_rarx_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, Arg},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* System Identification Tier-5 — extendedKalmanFilter(x0,P0,Q,R)
         * / unscentedKalmanFilter(...).  Allocate the zero-arg shell and
         * populate via the runtime (alloc-then-populate, like arx). */
        if ((CD->Name == "extendedKalmanFilter" ||
             CD->Name == "unscentedKalmanFilter") && C.Args.size() == 4) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value X0 = lowerExpr(*C.Args[0]);
          mlir::Value P0 = lowerExpr(*C.Args[1]);
          mlir::Value Qm = lowerExpr(*C.Args[2]);
          mlir::Value Rm = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_ekf_init"));
          emitUnregOp("matlab.call_builtin", {Obj, X0, P0, Qm, Rm},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Sensor Fusion Tier-1 — quaternion(...) constructors ============
         * Three forms: quaternion(w,x,y,z) -> 1×4 row; quaternion(M) where M is
         * an N×4/1×4/4×1 matrix; zero-arg (handled by the classdef ctor that
         * sets Data = [1 0 0 0]). */
        if (CD->Name == "quaternion" && C.Args.size() == 4) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "quaternion__quaternion"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value W = lowerExpr(*C.Args[0]);
          mlir::Value X = lowerExpr(*C.Args[1]);
          mlir::Value Y = lowerExpr(*C.Args[2]);
          mlir::Value Z = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_quat_init_wxyz"));
          emitUnregOp("matlab.call_builtin", {Obj, W, X, Y, Z},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "quaternion" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "quaternion__quaternion"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value M = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_quat_init_mat"));
          emitUnregOp("matlab.call_builtin", {Obj, M},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Sensor Fusion Tier-2 — trackingKF(...) constructor =============
         * Signature: trackingKF(F, H, Q, R, x0) — five matrix args.  The
         * runtime stores F/H/Q/R/x0 and initialises State + StateCovariance. */
        if (CD->Name == "trackingKF" && C.Args.size() == 5) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "trackingKF__trackingKF"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Fm = lowerExpr(*C.Args[0]);
          mlir::Value Hm = lowerExpr(*C.Args[1]);
          mlir::Value Qm = lowerExpr(*C.Args[2]);
          mlir::Value Rm = lowerExpr(*C.Args[3]);
          mlir::Value X0 = lowerExpr(*C.Args[4]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_trackingkf_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Fm, Hm, Qm, Rm, X0},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* trackingEKF(x0, P0, Q, R) / trackingUKF(x0, P0, Q, R) — re-skin of
         * the Ident EKF/UKF but with VECTOR measurement noise (ny×ny). */
        if ((CD->Name == "trackingEKF" || CD->Name == "trackingUKF") &&
            C.Args.size() == 4) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value X0 = lowerExpr(*C.Args[0]);
          mlir::Value P0 = lowerExpr(*C.Args[1]);
          mlir::Value Qm = lowerExpr(*C.Args[2]);
          mlir::Value Rm = lowerExpr(*C.Args[3]);
          const char *rt = (CD->Name == "trackingEKF")
                               ? "matlab_fusion_trackingekf_init"
                               : "matlab_fusion_trackingukf_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, X0, P0, Qm, Rm},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Sensor Fusion Tier-3 — sensor / filter constructors ============
         * imuSensor(fs, hasMag) — both args optional (zero-arg ctor handles
         * defaults; 1-2-arg form populates the SampleRate / HasMagnetometer
         * flags). */
        if (CD->Name == "imuSensor" && C.Args.size() >= 1 && C.Args.size() <= 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "imuSensor__imuSensor"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Fs = lowerExpr(*C.Args[0]);
          mlir::Value HasMag = (C.Args.size() == 2)
              ? lowerExpr(*C.Args[1])
              : emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
                  {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                       mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), 0.0))});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_imu_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Fs, HasMag},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "gpsSensor" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "gpsSensor__gpsSensor"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Fs = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_gps_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Fs},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if ((CD->Name == "ahrsfilter" || CD->Name == "imufilter" ||
             CD->Name == "complementaryFilter" || CD->Name == "insfilterMARG") &&
            C.Args.size() == 1) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Fs = lowerExpr(*C.Args[0]);
          const char *rt;
          if      (CD->Name == "ahrsfilter")          rt = "matlab_fusion_ahrs_init";
          else if (CD->Name == "imufilter")           rt = "matlab_fusion_imufilter_init";
          else if (CD->Name == "complementaryFilter") rt = "matlab_fusion_compfilter_init";
          else                                        rt = "matlab_fusion_insmarg_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, Fs},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Sensor Fusion Tier-4 — waypointTrajectory(wp, toa) ============
         * Two-arg ctor: N×3 waypoints + N×1 times.  Position-only interpolation
         * (lookupPose). */
        if (CD->Name == "waypointTrajectory" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "waypointTrajectory__waypointTrajectory"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Wp  = lowerExpr(*C.Args[0]);
          mlir::Value To  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_waypoint_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Wp, To},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Robotics Tier-1 — se3(T) / so3(R) constructor intercepts ===== */
        if (CD->Name == "se3" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "se3__se3"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value T = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_se3_init"));
          emitUnregOp("matlab.call_builtin", {Obj, T},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Deep Learning Toolbox — dlarray(X) leaf wrap ============== */
        if (CD->Name == "dlarray" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "dlarray__dlarray"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value X = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_dlnet_dlarray_init"));
          emitUnregOp("matlab.call_builtin", {Obj, X},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "so3" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "so3__so3"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value R = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_so3_init"));
          emitUnregOp("matlab.call_builtin", {Obj, R},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Robotics Tier-2 — rigidBodyTree (zero-arg + populator) ======== */
        if (CD->Name == "rigidBodyTree" && C.Args.size() == 0) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rigidBodyTree__rigidBodyTree"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_tree_init"));
          emitUnregOp("matlab.call_builtin", {Obj},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Robotics Tier-3 — inverseKinematics(rb) ====================== */
        if (CD->Name == "inverseKinematics" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "inverseKinematics__inverseKinematics"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Tr  = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_ik_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Tr},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* constraintPoseTarget(target_tform [, weights]) */
        if (CD->Name == "constraintPoseTarget" &&
            (C.Args.size() == 1 || C.Args.size() == 2)) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "constraintPoseTarget__constraintPoseTarget"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value T = lowerExpr(*C.Args[0]);
          mlir::Value W = (C.Args.size() == 2) ? lowerExpr(*C.Args[1]) : T;
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_constraint_pose_init"));
          emitUnregOp("matlab.call_builtin", {Obj, T, W},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* constraintPositionTarget(target) / constraintOrientationTarget(target). */
        if ((CD->Name == "constraintPositionTarget" ||
             CD->Name == "constraintOrientationTarget") &&
            (C.Args.size() == 1 || C.Args.size() == 2)) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Tg = lowerExpr(*C.Args[0]);
          mlir::Value W  = (C.Args.size() == 2) ? lowerExpr(*C.Args[1]) : Tg;
          const char *rt = (CD->Name == "constraintPositionTarget")
              ? "matlab_robotics_constraint_position_init"
              : "matlab_robotics_constraint_orientation_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, Tg, W},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* generalizedInverseKinematics(rb). */
        if (CD->Name == "generalizedInverseKinematics" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "generalizedInverseKinematics__generalizedInverseKinematics"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Tr  = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_gik_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Tr},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Robotics Tier-5 — diffdrive / occupancy map / PRM / pursuit == */
        if (CD->Name == "differentialDriveKinematics" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "differentialDriveKinematics__differentialDriveKinematics"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Wr = lowerExpr(*C.Args[0]);
          mlir::Value Tw = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_diffdrive_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Wr, Tw},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "unicycleKinematics" && C.Args.size() == 0) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "unicycleKinematics__unicycleKinematics"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_unicycle_init"));
          mlir::Value WR = emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
              {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                   mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), 0.1))});
          emitUnregOp("matlab.call_builtin", {Obj, WR}, {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if ((CD->Name == "bicycleKinematics" || CD->Name == "ackermannKinematics") &&
            (C.Args.size() == 0 || C.Args.size() == 1)) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value WB = (C.Args.size() == 1) ? lowerExpr(*C.Args[0])
              : emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
                  {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                       mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), 1.0))});
          const char *rt = (CD->Name == "bicycleKinematics")
              ? "matlab_robotics_bicycle_init" : "matlab_robotics_ackermann_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, WB}, {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "binaryOccupancyMap" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "binaryOccupancyMap__binaryOccupancyMap"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value R = lowerExpr(*C.Args[0]);
          mlir::Value Cl = lowerExpr(*C.Args[1]);
          mlir::Value Rs = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_occmap_init"));
          emitUnregOp("matlab.call_builtin", {Obj, R, Cl, Rs},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "mobileRobotPRM" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "mobileRobotPRM__mobileRobotPRM"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Mp = lowerExpr(*C.Args[0]);
          mlir::Value Nn = lowerExpr(*C.Args[1]);
          mlir::Value Cd = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_prm_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Mp, Nn, Cd},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "controllerPurePursuit" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "controllerPurePursuit__controllerPurePursuit"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Wp = lowerExpr(*C.Args[0]);
          mlir::Value Lh = lowerExpr(*C.Args[1]);
          mlir::Value Vm = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_pursuit_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Wp, Lh, Vm},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Robotics Tier-6 — collision primitives + manipulatorRRT ===== */
        if (CD->Name == "collisionBox" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "collisionBox__collisionBox"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value X = lowerExpr(*C.Args[0]);
          mlir::Value Y = lowerExpr(*C.Args[1]);
          mlir::Value Z = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_collbox_init"));
          emitUnregOp("matlab.call_builtin", {Obj, X, Y, Z},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "collisionSphere" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "collisionSphere__collisionSphere"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Rv = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_collsphere_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Rv},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if ((CD->Name == "collisionCylinder" || CD->Name == "collisionCapsule") &&
            C.Args.size() == 2) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Rv = lowerExpr(*C.Args[0]);
          mlir::Value Lv = lowerExpr(*C.Args[1]);
          const char *rt = (CD->Name == "collisionCylinder")
              ? "matlab_robotics_collcyl_init" : "matlab_robotics_collcap_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, Rv, Lv},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "manipulatorRRT" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "manipulatorRRT__manipulatorRRT"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Tr = lowerExpr(*C.Args[0]);
          mlir::Value Cn = lowerExpr(*C.Args[1]);
          mlir::Value Rd = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_rrt_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Tr, Cn, Rd},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Navigation Toolbox Tier-1 — occupancyMap / state spaces /
         * validator / navPath constructors ================================= */
        if (CD->Name == "occupancyMap" &&
            (C.Args.size() == 2 || C.Args.size() == 3)) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "occupancyMap__occupancyMap"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value W = lowerExpr(*C.Args[0]);
          mlir::Value H = lowerExpr(*C.Args[1]);
          mlir::Value Rs = (C.Args.size() == 3) ? lowerExpr(*C.Args[2])
              : emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
                  {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                       mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), 1.0))});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_occmap_init"));
          emitUnregOp("matlab.call_builtin", {Obj, W, H, Rs},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if ((CD->Name == "stateSpaceSE2" || CD->Name == "stateSpaceDubins") &&
            C.Args.size() == 1) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value B = lowerExpr(*C.Args[0]);
          const char *rt = (CD->Name == "stateSpaceSE2")
              ? "matlab_nav_ss_se2_init" : "matlab_nav_ss_dubins_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, B},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* validatorOccupancyMap(ss, map) — 2-arg idiom (the `.Map=` property
         * form is a documented carve-out; we clone the map at construction). */
        if (CD->Name == "validatorOccupancyMap" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "validatorOccupancyMap__validatorOccupancyMap"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Ss = lowerExpr(*C.Args[0]);
          mlir::Value Mp = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_validator_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Ss, Mp},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "navPath" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "navPath__navPath"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value St = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_path_init"));
          emitUnregOp("matlab.call_builtin", {Obj, St},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Navigation Tier-2 — sampling planners ===================== */
        if ((CD->Name == "plannerRRT" || CD->Name == "plannerRRTStar") &&
            C.Args.size() == 2) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Ss = lowerExpr(*C.Args[0]);
          mlir::Value Sv = lowerExpr(*C.Args[1]);
          double isStar = (CD->Name == "plannerRRTStar") ? 1.0 : 0.0;
          mlir::Value Sf = emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
              {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                   mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), isStar))});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_planner_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Ss, Sv, Sf},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "plannerAStarGrid" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "plannerAStarGrid__plannerAStarGrid"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Mp = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_astar_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Mp},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Reinforcement Learning Tier 1 — constructors ============= */
        /* rlMDPEnv(nextState, reward) — direct deterministic-MDP builder from
         * S×A next-state + reward tables (terminals self-loop). */
        if (CD->Name == "rlMDPEnv" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rlMDPEnv__rlMDPEnv"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Ns = lowerExpr(*C.Args[0]);
          mlir::Value Rw = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_mdp_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Ns, Rw},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* rlDQNAgent / rlPGAgent / rlDDPGAgent (obsInfo, actInfo) — auto-build
         * the critic/actor networks from the obs + action specs. */
        if ((CD->Name == "rlDQNAgent" || CD->Name == "rlPGAgent" ||
             CD->Name == "rlDDPGAgent" || CD->Name == "rlTD3Agent" ||
             CD->Name == "rlPPOAgent" || CD->Name == "rlSACAgent" ||
             CD->Name == "rlGRPOAgent" || CD->Name == "rlTRPOAgent") && C.Args.size() == 2) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Oi = lowerExpr(*C.Args[0]);
          mlir::Value Ai = lowerExpr(*C.Args[1]);
          const char *rt = (CD->Name == "rlPGAgent")   ? "matlab_rl_pg_init"
                         : (CD->Name == "rlDDPGAgent")  ? "matlab_rl_ddpg_init"
                         : (CD->Name == "rlTD3Agent")   ? "matlab_rl_td3_init"
                         : (CD->Name == "rlPPOAgent")   ? "matlab_rl_ppo_init"
                         : (CD->Name == "rlSACAgent")   ? "matlab_rl_sac_init"
                         : (CD->Name == "rlGRPOAgent")  ? "matlab_rl_grpo_init"
                         : (CD->Name == "rlTRPOAgent")  ? "matlab_rl_trpo_init"
                                                        : "matlab_rl_dqn_init";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Obj, Oi, Ai},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* rlTable(obsInfo, actInfo) — zeros(S,A) action-value table. */
        if (CD->Name == "rlTable" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rlTable__rlTable"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Oi = lowerExpr(*C.Args[0]);
          mlir::Value Ai = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_table_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Oi, Ai},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* rlQValueFunction(table, obsInfo, actInfo) — wrap the table; the
         * spec args are accepted for fidelity but only the table is read. */
        if (CD->Name == "rlQValueFunction" && C.Args.size() >= 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rlQValueFunction__rlQValueFunction"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Tb = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_qvf_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Tb},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* rlQAgent(critic[,opts]) / rlSARSAAgent(critic[,opts]) — copy the
         * critic's Q table; hyperparameters are read from the agent's scalar
         * properties at train time (the AgentOptions struct nesting is a
         * documented Tier-1 simplification). */
        if ((CD->Name == "rlQAgent" || CD->Name == "rlSARSAAgent") &&
            C.Args.size() >= 1) {
          std::string Ctor = std::string(CD->Name) + "__" + std::string(CD->Name);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Ctor));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Cr = lowerExpr(*C.Args[0]);
          double isSarsa = (CD->Name == "rlSARSAAgent") ? 1.0 : 0.0;
          mlir::Value Sf = emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
              {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                   mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), isSarsa))});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_agent_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Cr, Sf},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Navigation Tier-3 — lidarScan / lidarSLAM ================= */
        if (CD->Name == "lidarScan" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "lidarScan__lidarScan"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Rg = lowerExpr(*C.Args[0]);
          mlir::Value An = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_lidarscan_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Rg, An},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "lidarSLAM" && C.Args.size() == 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "lidarSLAM__lidarSLAM"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Rs = lowerExpr(*C.Args[0]);
          mlir::Value Mr = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_slam_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Rs, Mr},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Navigation Tier-5/6 — single-obj-arg ctor intercepts ======= */
        if (CD->Name == "monteCarloLocalization" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "monteCarloLocalization__monteCarloLocalization"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Mp = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_mcl_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Mp},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "referencePathFrenet" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "referencePathFrenet__referencePathFrenet"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Wp = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_frenet_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Wp},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (CD->Name == "trajectoryGeneratorFrenet" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "trajectoryGeneratorFrenet__trajectoryGeneratorFrenet"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Rp = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_trajgen_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Rp},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* ===== Sensor Fusion Tier-5 — trackerGNN(maxTracks) ==================
         * One-arg ctor.  Empty tracker. */
        if (CD->Name == "trackerGNN" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "trackerGNN__trackerGNN"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value M   = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_gnn_init"));
          emitUnregOp("matlab.call_builtin", {Obj, M},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* objectDetection(time, z, R) — set Time / Measurement /
         * MeasurementNoise.  2-arg form (without R) is a documented follow-on
         * (runtime defaults R to eye(ny) when MeasurementNoise is empty). */
        if (CD->Name == "objectDetection" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "objectDetection__objectDetection"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Tm = lowerExpr(*C.Args[0]);
          mlir::Value Zm = lowerExpr(*C.Args[1]);
          mlir::Value Rm = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_objdet_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Tm, Zm, Rm},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        /* System Identification Tier-1.8 — ss(idpoly) / tf(idpoly)
         * conversion.  `ss`/`tf` resolve as classes (not builtins), so
         * this interception lives in the constructor-call path rather
         * than the BindingKind::Builtin block above.  ss(model) builds
         * the controllable-canonical realisation (carrying the discrete
         * Ts); tf(model) extracts B/A (the CST tf is Ts-less). */
        if (CD->Name == "ss" && C.Args.size() == 1 && C.Args[0]) {
          const ClassDef *Arg0 = nullptr;
          if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]))
            if (AN->Ref) Arg0 = AN->Ref->PinnedClass;
          /* ss(idss) / ss(idgrey) — the model already is a realization;
           * copy its A/B/C/D/Ts straight into a CST ss. */
          if (Arg0 && (Arg0->Name == "idss" || Arg0->Name == "idgrey")) {
            mlir::Value Model = lowerExpr(*C.Args[0]);
            if (Model.getType() != PtrTyConst) Model.setType(PtrTyConst);
            auto F64c = mlir::Float64Type::get(&MCtx);
            auto callRT = [&](const char *sym) -> mlir::Value {
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, sym));
              return emitUnreg("matlab.call_builtin", {Model}, PtrTyConst, L, {Cal});
            };
            mlir::Value Am = callRT("matlab_ident_ss_A");
            mlir::Value Bm = callRT("matlab_ident_ss_B");
            mlir::Value Cm = callRT("matlab_ident_ss_C");
            mlir::Value Dm = callRT("matlab_ident_ss_D");
            mlir::Value NameTs = emitFieldNameChar("Ts", L);
            mlir::NamedAttribute TsCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_obj_get_f64"));
            mlir::Value Ts = emitUnreg("matlab.call_builtin", {Model, NameTs},
                                       F64c, L, {TsCal});
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "ss__ss"));
            return emitUnreg("matlab.call", {Am, Bm, Cm, Dm, Ts},
                             PtrTyConst, L, {CtorCal});
          }
        }
        if ((CD->Name == "ss" || CD->Name == "tf") && C.Args.size() == 1 &&
            C.Args[0]) {
          const ClassDef *Arg0 = nullptr;
          if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]))
            if (AN->Ref) Arg0 = AN->Ref->PinnedClass;
          if (Arg0 && Arg0->Name == "idpoly") {
            mlir::Value Model = lowerExpr(*C.Args[0]);
            if (Model.getType() != PtrTyConst) Model.setType(PtrTyConst);
            auto F64c = mlir::Float64Type::get(&MCtx);
            auto getMat = [&](const char *field) -> mlir::Value {
              mlir::Value NameV = emitFieldNameChar(field, L);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_obj_get_mat"));
              return emitUnreg("matlab.call_builtin", {Model, NameV},
                               PtrTyConst, L, {Cal});
            };
            auto callRT = [&](const char *sym) -> mlir::Value {
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, sym));
              return emitUnreg("matlab.call_builtin", {Model},
                               PtrTyConst, L, {Cal});
            };
            if (CD->Name == "tf") {
              mlir::Value Bp = getMat("B");
              mlir::Value Ap = getMat("A");
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "tf__tf"));
              return emitUnreg("matlab.call", {Bp, Ap}, PtrTyConst, L, {Cal});
            }
            mlir::Value Am = callRT("matlab_ident_poly2ss_A");
            mlir::Value Bm = callRT("matlab_ident_poly2ss_B");
            mlir::Value Cm = callRT("matlab_ident_poly2ss_C");
            mlir::Value Dm = callRT("matlab_ident_poly2ss_D");
            mlir::Value NameTs = emitFieldNameChar("Ts", L);
            mlir::NamedAttribute TsCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_obj_get_f64"));
            mlir::Value Ts = emitUnreg("matlab.call_builtin", {Model, NameTs},
                                       F64c, L, {TsCal});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "ss__ss"));
            return emitUnreg("matlab.call", {Am, Bm, Cm, Dm, Ts},
                             PtrTyConst, L, {Cal});
          }
        }
        bool HasCtor = false;
        for (const Function *Mth : CD->Methods)
          if (Mth && Mth->Name == CD->Name) { HasCtor = true; break; }
        /* Name-value pair sugar (MathWorks `txsite('Name','X',...)` shape).
         *
         * Detect the kwarg-only pattern: every arg pair is
         * `(string_literal, value)` where the string literal matches
         * a property name of `CD` (or any superclass).  When the
         * pattern holds, emit a zero-arg ctor call (or `matlab_obj_new`
         * for ctor-less classes), then a `matlab_obj_set_<kind>` per
         * pair to populate the named properties.  Skips when args
         * don't fit the pattern so the normal positional-ctor path
         * stays intact for `AntDipole(2.0, 0.05, 0.0)` etc.
         *
         * The class must have either no ctor or a ctor whose body
         * tolerates `nargin == 0` (which our generated classdefs do
         * via `if nargin >= 1 ... end` guards).  MathWorks-style
         * classes that use this kwarg pattern conventionally do; if
         * a class doesn't, the user can fall back to positional. */
        auto allPropsByName = [&]() -> std::vector<llvm::StringRef> {
          std::vector<llvm::StringRef> Out;
          for (const ClassDef *CC = CD; CC; CC = CC->Super)
            for (const auto &P : CC->Props) Out.push_back(P.Name);
          return Out;
        };
        bool IsKwargPattern = false;
        std::vector<std::string> KwargKeys;
        std::vector<const Expr *> KwargVals;
        if (C.Args.size() >= 2 && (C.Args.size() % 2) == 0) {
          IsKwargPattern = true;
          auto Props = allPropsByName();
          for (size_t i = 0; i < C.Args.size(); i += 2) {
            const Expr *KE = C.Args[i];
            std::string Key;
            if (auto *CL = dynamic_cast<const CharLiteral *>(KE))
              Key = CL->Value;
            else if (auto *SL = dynamic_cast<const StringLiteral *>(KE))
              Key = SL->Value;
            if (Key.empty()) { IsKwargPattern = false; break; }
            bool Match = false;
            for (auto &P : Props) if (P == Key) { Match = true; break; }
            if (!Match) { IsKwargPattern = false; break; }
            KwargKeys.push_back(std::move(Key));
            KwargVals.push_back(C.Args[i + 1]);
          }
        }
        if (IsKwargPattern) {
          /* Step 1: create the instance.  If the class has an
           * explicit ctor, call it with no args (the body's nargin
           * guards keep it well-defined).  Otherwise allocate via
           * `matlab_obj_new(class_id)`. */
          mlir::Value Obj;
          if (HasCtor) {
            std::string Callee = std::string(CD->Name) + "__" +
                                  std::string(CD->Name);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            Obj = emitUnreg("matlab.call", {}, PtrTyConst, L, {Cal});
          } else {
            auto I32 = mlir::IntegerType::get(&MCtx, 32);
            mlir::Value ClsId = mlir::arith::ConstantOp::create(
                B, L, I32, mlir::IntegerAttr::get(I32, (int64_t)CD->ClassId));
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_obj_new"));
            Obj = emitUnreg("matlab.call_builtin", {ClsId},
                            PtrTyConst, L, {Cal});
          }
          /* Step 2: set each named property.  Lower the value, then
           * pick `matlab_obj_set_f64` / `_set_mat` / `_set_string`
           * based on the value's MLIR type.  Char/string literals
           * coerce through `matlab_string_from_literal` so the obj
           * receives a `matlab_string *`. */
          auto F64 = mlir::Float64Type::get(&MCtx);
          for (size_t i = 0; i < KwargKeys.size(); ++i) {
            mlir::Value NameV = emitFieldNameChar(KwargKeys[i], L);
            const Expr *VE = KwargVals[i];
            mlir::Value V;
            bool IsString = false;
            if (auto *CL = dynamic_cast<const CharLiteral *>(VE)) {
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
              mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                          mlir::NoneType::get(&MCtx), L, {VA});
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
              V = emitUnreg("matlab.call_builtin", {Ch}, PtrTyConst, L, {Cal});
              IsString = true;
            } else if (auto *SL = dynamic_cast<const StringLiteral *>(VE)) {
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::StringAttr::get(&MCtx, std::string(SL->Value)));
              mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                          mlir::NoneType::get(&MCtx), L, {VA});
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
              V = emitUnreg("matlab.call_builtin", {Ch}, PtrTyConst, L, {Cal});
              IsString = true;
            } else {
              V = lowerExpr(*VE);
            }
            llvm::StringRef Setter;
            if (IsString) {
              Setter = "matlab_obj_set_string";
            } else if (V.getType() == PtrTyConst ||
                       mlir::isa<mlir::RankedTensorType,
                                 mlir::UnrankedTensorType>(V.getType())) {
              Setter = "matlab_obj_set_mat";
            } else if (V.getType() == F64) {
              Setter = "matlab_obj_set_f64";
            } else {
              /* Int / bool — coerce to f64. */
              Setter = "matlab_obj_set_f64";
            }
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Setter));
            emitUnregOp("matlab.call_builtin", {Obj, NameV, V},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
          }
          return Obj;
        }
        if (HasCtor) {
          /* §3.1 sugar: `tf('s')` and `tf('z')` — MATLAB's idiom to
           * mint the Laplace / z-transform variable. Rewrite as
           * `tf([1, 0], 1)` here at the call site (char literals don't
           * survive the constructor body's AST→MLIR lowering, but a
           * 1×2 row matrix + scalar denominator does — this is the
           * same shape as `s = tf([1 0], 1)` that operator overloads
           * already exercise). The discrete-time `tf('z')` variant
           * lands the same nominal coefficients; sample-time
           * carry-through is a follow-on. */
          if (CD->Name == "tf" && C.Args.size() == 1 && C.Args[0]) {
            const CharLiteral *CL =
                dynamic_cast<const CharLiteral *>(C.Args[0]);
            const StringLiteral *SL =
                CL ? nullptr
                   : dynamic_cast<const StringLiteral *>(C.Args[0]);
            llvm::StringRef Tok = CL ? CL->Value : (SL ? SL->Value : "");
            if (Tok == "s" || Tok == "z") {
              auto F64 = mlir::Float64Type::get(&MCtx);
              mlir::Value One = mlir::arith::ConstantOp::create(
                  B, L, F64, mlir::FloatAttr::get(F64, 1.0)).getResult();
              mlir::Value Zero = mlir::arith::ConstantOp::create(
                  B, L, F64, mlir::FloatAttr::get(F64, 0.0)).getResult();
              auto NumTy = mlir::RankedTensorType::get({1, 2}, F64);
              mlir::Value Num = emitUnreg(
                  "matlab.concat_row", {One, Zero}, NumTy, L);
              mlir::Value Den = mlir::arith::ConstantOp::create(
                  B, L, F64, mlir::FloatAttr::get(F64, 1.0)).getResult();
              std::string Callee = std::string(CD->Name) + "__" +
                                    std::string(CD->Name);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Callee));
              return emitUnreg("matlab.call", {Num, Den},
                               PtrTyConst, L, {Cal});
            }
          }
          /* Wrap char/string literal args through
           * `matlab_string_from_literal` so the ctor body sees a
           * `matlab_string *` (ptr) rather than the raw char tensor.
           * Without this, a positional `PropagationModel('freespace')`
           * call passes `tensor<1x9xi8>` and the ctor body's
           * `obj.Kind = kind` lowers as `matlab_obj_set_string(_, _,
           * none)` which can't lower further.  The kwarg sugar path
           * already does this wrap; mirror the behavior here for
           * positional ctor calls. */
          llvm::SmallVector<mlir::Value, 4> Args;
          for (const Expr *A : C.Args) {
            if (!A) continue;
            const CharLiteral *CL = dynamic_cast<const CharLiteral *>(A);
            const StringLiteral *SL = CL ? nullptr
                : dynamic_cast<const StringLiteral *>(A);
            if (CL || SL) {
              llvm::StringRef Tok = CL ? CL->Value : SL->Value;
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::StringAttr::get(&MCtx, std::string(Tok)));
              mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                          mlir::NoneType::get(&MCtx),
                                          L, {VA});
              mlir::NamedAttribute SCal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
              Args.push_back(
                  emitUnreg("matlab.call_builtin", {Ch}, PtrTyConst,
                            L, {SCal}));
            } else {
              Args.push_back(lowerExpr(*A));
            }
          }
          std::string Callee = std::string(CD->Name) + "__" +
                                std::string(CD->Name);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          return emitUnreg("matlab.call", Args, PtrTyConst, L, {Cal});
        }
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        mlir::Value ClsId = mlir::arith::ConstantOp::create(
            B, L, I32, mlir::IntegerAttr::get(I32, (int64_t)CD->ClassId));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_obj_new"));
        mlir::Value Obj = emitUnreg("matlab.call_builtin", {ClsId},
                                     PtrTyConst, L, {Cal});
        /* Apply default property values (constructor-less path). */
        for (const auto &P : CD->Props) {
          if (!P.Default) continue;
          mlir::Value DV = lowerExpr(*P.Default);
          mlir::Value NameV = emitFieldNameChar(P.Name, L);
          bool IsMat = DV && (DV.getType() == PtrTyConst ||
                              mlir::isa<mlir::RankedTensorType,
                                        mlir::UnrankedTensorType>(DV.getType()));
          llvm::StringRef Cn = IsMat ? "matlab_obj_set_mat"
                                      : "matlab_obj_set_f64";
          mlir::NamedAttribute Cal2(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Cn));
          emitUnregOp("matlab.call_builtin", {Obj, NameV, DV},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal2});
        }
        return Obj;
      }
      /* extractdata(x) -> the underlying matrix; dlgradient(loss, v) -> the
       * gradient matrix (reverse sweep of the autodiff tape).  Both names are
       * deep-learning-exclusive, so route unconditionally. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "extractdata" && C.Args.size() == 1) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value X = lowerExpr(*C.Args[0]);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_dlnet_extractdata"));
        return emitUnreg("matlab.call_builtin", {X}, PtrTy, L, {Cal});
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "dlgradient" && C.Args.size() == 2) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value Lv = lowerExpr(*C.Args[0]);
        mlir::Value Vv = lowerExpr(*C.Args[1]);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_dlnet_grad"));
        return emitUnreg("matlab.call_builtin", {Lv, Vv}, PtrTy, L, {Cal});
      }
      /* ===== Deep Learning Toolbox — dlarray activation/reduction/loss ====
       * relu/sigmoid/tanh/softmax/sum/mean/log/exp/crossentropy/mse on a
       * dlarray-pinned argument route to the dlarray method (recording onto
       * the autodiff tape).  pinnedDl recurses through operators + ctor calls
       * + dlarray-returning calls so `relu(W*X+b)` is recognised via the
       * pinned leaf W.  Falls through to the numeric builtin when no operand
       * is a dlarray (so matrix `tanh`/`sum`/`log`/`exp` are unaffected). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          C.Args.size() >= 1) {
        static const llvm::StringSet<> DlFns = {
            "relu", "sigmoid", "tanh", "softmax", "sum", "mean",
            "log", "exp", "crossentropy", "mse", "lstm",
            "transpose", "ctranspose", "embed",
            "gru", "bilstm", "lstmp",
            /* Phase 1 small ops. */
            "sqrt", "leakyrelu", "gelu", "swish", "softplus", "elu",
            /* Tier C: rank-4 batched conv + reshape + pooling + full norm family. */
            "conv2d_batch", "conv2d_full", "reshape",
            "maxpool2d", "avgpool2d", "batchnorm",
            "layernorm", "batchnorm_eval",
            "groupnorm", "batchnorm_train",
            "instancenorm", "rmsnorm"};
        if (DlFns.contains(N->Name)) {
          std::function<bool(const Expr *)> pinnedDl =
              [&pinnedDl](const Expr *X) -> bool {
            if (!X) return false;
            if (auto *NE = dynamic_cast<const NameExpr *>(X))
              return NE->Ref && NE->Ref->PinnedClass &&
                     NE->Ref->PinnedClass->Name == "dlarray";
            if (auto *Bi2 = dynamic_cast<const BinaryOpExpr *>(X))
              return pinnedDl(Bi2->LHS) || pinnedDl(Bi2->RHS);
            if (auto *U2 = dynamic_cast<const UnaryOpExpr *>(X))
              return pinnedDl(U2->Operand);
            if (auto *CX = dynamic_cast<const CallOrIndex *>(X)) {
              if (auto *NX = dynamic_cast<const NameExpr *>(CX->Callee)) {
                if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
                    NX->Ref->ClassDef && NX->Ref->ClassDef->Name == "dlarray")
                  return true;
                static const llvm::StringSet<> DlRet = {
                    "relu", "sigmoid", "tanh", "softmax", "sum", "mean",
                    "log", "exp", "crossentropy", "mse", "lstm",
                    "transpose", "ctranspose", "embed",
                    "gru", "bilstm", "lstmp", "dlarray",
                    "sqrt", "leakyrelu", "gelu", "swish",
                    "softplus", "elu", "conv2d_batch", "conv2d_full",
                    "reshape", "maxpool2d", "avgpool2d", "batchnorm",
                    "layernorm", "batchnorm_eval",
                    "groupnorm", "batchnorm_train",
                    "instancenorm", "rmsnorm"};
                if (DlRet.contains(NX->Name))
                  for (size_t i = 0; i < CX->Args.size(); ++i)
                    if (pinnedDl(CX->Args[i])) return true;
              }
            }
            return false;
          };
          bool anyDl = false;
          for (size_t i = 0; i < C.Args.size(); ++i)
            if (pinnedDl(C.Args[i])) { anyDl = true; break; }
          if (anyDl) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            std::vector<mlir::Value> Vs;
            for (size_t i = 0; i < C.Args.size(); ++i)
              Vs.push_back(lowerExpr(*C.Args[i]));
            /* Per-arity rename: `mean(X)` -> dlarray__mean (1-arg) but
             * `mean(X, dim)` -> dlarray__mean_dim (2-arg); same shape
             * for `reshape(X, m, n)` vs `reshape(X, d1, d2, d3, d4)`,
             * `softmax(X)` vs `softmax(X, dim)`. */
            std::string MethodName(N->Name);
            if (MethodName == "mean" && C.Args.size() == 2)
              MethodName = "mean_dim";
            else if (MethodName == "reshape" && C.Args.size() == 3)
              MethodName = "reshape2";
            else if (MethodName == "reshape" && C.Args.size() == 5)
              MethodName = "reshape4";
            else if (MethodName == "softmax" && C.Args.size() == 2)
              MethodName = "softmax_dim";
            std::string Callee = std::string("dlarray__") + MethodName;
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call", Vs, PtrTy, L, {Cal});
          }
        }
      }
      /* bin(fi) / hex(fi) / dec(fi) — render the stored integer as a
       * matlab_string. The result is tagged through StringBindings on
       * assignment (see AssignStmt lowering) so disp(s) routes to
       * matlab_string_disp. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "bin" || N->Name == "hex" || N->Name == "dec") &&
          C.Args.size() == 1 && C.Args[0] &&
          C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AT = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AT.Elt == Dtype::Fixed && AT.FxSpec) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          auto I64 = mlir::IntegerType::get(&MCtx, 64);
          auto I8  = mlir::IntegerType::get(&MCtx, 8);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          mlir::Value Wide = V;
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
            if (IT.getWidth() < 64) {
              Wide = AT.FxSpec->Signed
                  ? (mlir::Value)mlir::arith::ExtSIOp::create(B, L, I64, V)
                  : (mlir::Value)mlir::arith::ExtUIOp::create(B, L, I64, V);
            }
          }
          mlir::Value WL = mlir::arith::ConstantOp::create(
              B, L, I8,
              mlir::IntegerAttr::get(I8, (int64_t)AT.FxSpec->WordLength));
          std::string Callee = std::string("matlab_fi_") +
              std::string(N->Name) + "_" + (AT.FxSpec->Signed ? "s" : "u");
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          return emitUnreg("matlab.call_builtin", {Wide, WL}, PtrTy, L, {Cal});
        }
      }

      /* reinterpretcast(n, T) — bit-reinterpret the stored integer as
       * a new numerictype. Same-width: just type-change the value (which
       * for signless integers is identity). Different storage widths
       * extend or truncate; semantically the user is asking for the
       * raw bits, so signed extension follows the *target* signedness. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "reinterpretcast" &&
          C.Args.size() >= 1 && C.Args[0] && C.Args[0]->Ty &&
          C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &SrcA = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (SrcA.Elt == Dtype::Fixed && SrcA.FxSpec) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          mlir::Type ResTy = mirTy(E.Ty ? E.Ty : TC.any());
          if (V.getType() == ResTy) return V;
          if (auto SrcIT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
            if (auto DstIT = mlir::dyn_cast<mlir::IntegerType>(ResTy)) {
              if (SrcIT.getWidth() == DstIT.getWidth())
                return V; // signless ↔ signless of same width is a no-op
              if (SrcIT.getWidth() < DstIT.getWidth()) {
                bool DstSigned = false;
                if (auto *DA = E.Ty;
                    DA && DA->K == Type::Kind::Array) {
                  auto &DD = static_cast<const ArrayType &>(*DA);
                  if (DD.Elt == Dtype::Fixed && DD.FxSpec)
                    DstSigned = DD.FxSpec->Signed;
                }
                return DstSigned
                    ? (mlir::Value)mlir::arith::ExtSIOp::create(B, L, ResTy, V)
                    : (mlir::Value)mlir::arith::ExtUIOp::create(B, L, ResTy, V);
              }
              return mlir::arith::TruncIOp::create(B, L, ResTy, V);
            }
          }
          return V;
        }
      }

      /* setfimath(n, F) / removefimath(n) — compile-time spec mutation
       * only. The stored integer is unchanged; we return the operand's
       * value directly with the result expression's (new) type. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "setfimath" || N->Name == "removefimath") &&
          !C.Args.empty() && C.Args[0] && C.Args[0]->Ty &&
          C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AT = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AT.Elt == Dtype::Fixed && AT.FxSpec) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          // The result type's storage class equals the operand's (WL/FL
          // are unchanged), so a plain pass-through is correct.
          return V;
        }
      }

      /* int(fi) / storedInteger(fi) — return the underlying stored
       * integer in its native lane. Sema already typed the result as the
       * matching int8/16/32/64, so all lowering needs to do is pass the
       * value through; the fi storage class IS the native int. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "int" || N->Name == "storedInteger") &&
          C.Args.size() == 1 && C.Args[0] &&
          C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AT = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AT.Elt == Dtype::Fixed && AT.FxSpec) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          mlir::Type ResTy = mirTy(E.Ty ? E.Ty : TC.any());
          if (V.getType() == ResTy) return V;
          if (mlir::isa<mlir::IntegerType>(V.getType()) &&
              mlir::isa<mlir::IntegerType>(ResTy))
            return mlir::arith::BitcastOp::create(B, L, ResTy, V);
          return V; // type mapper covers the storage class.
        }
      }

      /* Phase 1.1.G — typed-int matrix cross-lane / to-double casts.
       * `int32(uint8_matrix)`, `uint8(int32_matrix)`, `double(int32_matrix)`,
       * and `double(uint8_matrix)` need to consult the operand's type
       * because the Spec-table dispatch in LowerTensorOps is shape-blind
       * (every matrix arg is opaque !llvm.ptr by then). Route to the
       * lane-aware runtime entry point directly so the typed storage
       * isn't reinterpreted as f64. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "int32" || N->Name == "uint8" || N->Name == "double") &&
          C.Args.size() == 1 && C.Args[0] &&
          C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Array) {
        llvm::StringRef SrcSuf = intDtypeSuffixOf(C.Args[0]);
        if (!SrcSuf.empty()) {
          /* Source is a non-scalar Int32 / UInt8 array. Pick the runtime
           * entry that takes the typed source. */
          std::string Callee;
          if (N->Name == "double")
            Callee = ("matlab_mat_" + SrcSuf + "_to_double").str();
          else if (N->Name == "int32" && SrcSuf == "u8")
            Callee = "matlab_mat_i32_from_u8";
          else if (N->Name == "uint8" && SrcSuf == "i32")
            Callee = "matlab_mat_u8_from_i32";
          /* Same-lane casts (int32(int32_matrix) / uint8(uint8_matrix))
           * fall through to the default Spec dispatch — those are no-ops
           * in MATLAB and the from_double path is harmless because Sema
           * collapses them; let LowerTensorOps see the call. */
          if (!Callee.empty()) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call_builtin", {V}, PtrTy, L, {Cal});
          }
        }
      }

      /* storedIntegerToDouble(fi) / double(fi) — render the real-world
       * value. We multiply the stored int by 2^-FL at runtime; for now
       * we route through the runtime helper matlab_fi_to_double_*, which
       * is shorter than emitting an extsi + sitofp + mul sequence. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "double" || N->Name == "storedIntegerToDouble") &&
          C.Args.size() == 1 && C.Args[0] &&
          C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AT = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AT.Elt == Dtype::Fixed && AT.FxSpec &&
            AT.S.K == Shape::Rank::Scalar) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto I64 = mlir::IntegerType::get(&MCtx, 64);
          /* Sign- or zero-extend stored int to i64 so we have headroom
           * for the multiply, then SIToFP, then multiply by 2^-FL. */
          mlir::Value Wide = V;
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
            if (IT.getWidth() < 64) {
              if (AT.FxSpec->Signed)
                Wide = mlir::arith::ExtSIOp::create(B, L, I64, V);
              else
                Wide = mlir::arith::ExtUIOp::create(B, L, I64, V);
            }
          }
          mlir::Value AsF = AT.FxSpec->Signed
              ? (mlir::Value)mlir::arith::SIToFPOp::create(B, L, F64, Wide)
              : (mlir::Value)mlir::arith::UIToFPOp::create(B, L, F64, Wide);
          double Scale = std::ldexp(1.0, -AT.FxSpec->FractionLength);
          mlir::Value ScaleC = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, Scale));
          return mlir::arith::MulFOp::create(B, L, AsF, ScaleC);
        }
      }

      /* fi(zeros(m, n), signed, WL, FL) — fi array zero-init.
       * Folds at lower-time to a direct call to matlab_mat_{i,u}64_zeros,
       * bypassing the f64 zeros + cast detour. This is the FIR delay-line
       * shape from plan §7.3. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "fi" && C.Args.size() >= 4 && C.Args[0] &&
          E.Ty && E.Ty->K == Type::Kind::Array) {
        auto &OutAT = static_cast<const ArrayType &>(*E.Ty);
        if (OutAT.Elt == Dtype::Fixed && OutAT.FxSpec &&
            OutAT.S.K != Shape::Rank::Scalar) {
          if (auto *ZC = dynamic_cast<const CallOrIndex *>(C.Args[0])) {
            auto *ZN = dynamic_cast<const NameExpr *>(ZC->Callee);
            if (ZN && ZN->Ref && ZN->Ref->Kind == BindingKind::Builtin &&
                (ZN->Name == "zeros" || ZN->Name == "ones") &&
                !ZC->Args.empty()) {
              auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
              mlir::Value M = lowerExpr(*ZC->Args[0]);
              mlir::Value Ncols = ZC->Args.size() >= 2
                  ? lowerExpr(*ZC->Args[1])
                  : M;
              llvm::StringRef Callee = OutAT.FxSpec->Signed
                  ? "matlab_mat_i64_zeros"
                  : "matlab_mat_u64_zeros";
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Callee));
              llvm::SmallVector<mlir::NamedAttribute, 8> A;
              A.push_back(Cal);
              auto FAttrs = buildFixedAttrs(&MCtx, *OutAT.FxSpec);
              for (auto &E0 : FAttrs) A.push_back(E0);
              mlir::Value Z = emitUnreg("matlab.call_builtin", {M, Ncols},
                                         PtrTy, L, A);
              if (ZN->Name == "ones") {
                /* fi(ones(m,n), s, WL, FL) — fill with stored value
                 * representing 1.0 (i.e. 1 << FL), saturated to WL. */
                int64_t StoredOne = (int64_t)1 << OutAT.FxSpec->FractionLength;
                if (OutAT.FxSpec->WordLength < 64 &&
                    StoredOne >= ((int64_t)1 << (OutAT.FxSpec->WordLength -
                                                 (OutAT.FxSpec->Signed ? 1 : 0))))
                  StoredOne = ((int64_t)1 << (OutAT.FxSpec->WordLength -
                                              (OutAT.FxSpec->Signed ? 1 : 0))) - 1;
                auto I64 = mlir::IntegerType::get(&MCtx, 64);
                mlir::Value Vone = mlir::arith::ConstantOp::create(
                    B, L, I64, mlir::IntegerAttr::get(I64, StoredOne));
                mlir::NamedAttribute Cf(
                    mlir::StringAttr::get(&MCtx, "callee"),
                    mlir::StringAttr::get(&MCtx,
                        OutAT.FxSpec->Signed
                            ? "matlab_mat_i64_fill"
                            : "matlab_mat_u64_fill"));
                emitUnregOp("matlab.call_builtin", {Z, Vone},
                            {mlir::NoneType::get(&MCtx)}, L, {Cf});
              }
              return Z;
            }
          }
        }
      }

      /* fi([lit, lit, ...], signed, WL, FL) — fi array literal init.
       * Folds at lower-time into a `matlab_mat_i64_zeros(1, N)` followed
       * by per-element `__subscript_store(arr, k+1, quantized_lit_k)`
       * calls — the same shape that `fi(zeros(1, N), ...) ; v(k) = ...;`
       * lowers to, which the existing `LowerStaticFiArrays` pass folds
       * to `llvm.alloca [N x iW]` with constant-init stores. Drops out
       * the `concat_row + matlab_mat_from_buf + fi.cast(tensor)` runtime
       * detour that doesn't lower in any backend. Coefficient table
       * shape (`h = fi([0.1, 0.2, ...], 1, 16, 15)`) for FIR / IIR / CIC
       * filter implementations. Phase 5.6 Stage C. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "fi" && C.Args.size() >= 4 && C.Args[0] &&
          E.Ty && E.Ty->K == Type::Kind::Array) {
        auto &OutAT = static_cast<const ArrayType &>(*E.Ty);
        if (OutAT.Elt == Dtype::Fixed && OutAT.FxSpec &&
            OutAT.S.K != Shape::Rank::Scalar) {
          if (auto *ML = dynamic_cast<const MatrixLiteral *>(C.Args[0])) {
            // 1-row literal `[v0, v1, ..., vN]`. (2-D matrix literals
            // are out of v1 scope; the static-fi-array infrastructure
            // is 1-D today.)
            if (ML->Rows.size() == 1 && !ML->Rows[0].empty()) {
              auto &Row = ML->Rows[0];
              auto isLiteralFold = [](const Expr *X) -> bool {
                if (!X) return false;
                if (X->Kind == NodeKind::IntegerLiteral ||
                    X->Kind == NodeKind::FPLiteral) return true;
                if (auto *U = dynamic_cast<const UnaryOpExpr *>(X))
                  if (U->Op == UnOp::Minus || U->Op == UnOp::Plus)
                    return U->Operand && (
                        U->Operand->Kind == NodeKind::IntegerLiteral ||
                        U->Operand->Kind == NodeKind::FPLiteral);
                return false;
              };
              bool AllLit = true;
              for (Expr *E0 : Row) {
                if (!isLiteralFold(E0)) { AllLit = false; break; }
              }
              if (AllLit) {
                FixedSpec Spec = *OutAT.FxSpec;
                auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
                auto F64 = mlir::Float64Type::get(&MCtx);
                // Step 1: emit matlab_mat_i64_zeros(1, N).
                mlir::Value RowsV = mlir::arith::ConstantOp::create(
                    B, L, F64, mlir::FloatAttr::get(F64, 1.0));
                mlir::Value ColsV = mlir::arith::ConstantOp::create(
                    B, L, F64, mlir::FloatAttr::get(F64, (double)Row.size()));
                llvm::StringRef ZeroCallee = Spec.Signed
                    ? "matlab_mat_i64_zeros"
                    : "matlab_mat_u64_zeros";
                mlir::NamedAttribute ZCal(
                    mlir::StringAttr::get(&MCtx, "callee"),
                    mlir::StringAttr::get(&MCtx, ZeroCallee));
                llvm::SmallVector<mlir::NamedAttribute, 8> ZA;
                ZA.push_back(ZCal);
                auto FAttrs = buildFixedAttrs(&MCtx, Spec);
                for (auto &E0 : FAttrs) ZA.push_back(E0);
                mlir::Value Z = emitUnreg("matlab.call_builtin",
                                           {RowsV, ColsV}, PtrTy, L, ZA);
                // Step 2: per-element __subscript_store with the
                // compile-time-quantized stored integer.
                unsigned StorBits = Spec.WordLength <= 8 ? 8
                                  : Spec.WordLength <= 16 ? 16
                                  : Spec.WordLength <= 32 ? 32 : 64;
                auto IT = mlir::IntegerType::get(&MCtx, StorBits);
                for (size_t k = 0; k < Row.size(); ++k) {
                  double Val = foldFloat(Row[k]);
                  int64_t Stored = Spec.Signed
                      ? quantizeFixedSigned(Val, Spec)
                      : (int64_t)quantizeFixedUnsigned(Val, Spec);
                  mlir::Value KV = mlir::arith::ConstantOp::create(
                      B, L, F64, mlir::FloatAttr::get(F64, (double)(k + 1)));
                  llvm::SmallVector<mlir::NamedAttribute, 8> CA;
                  mlir::NamedAttribute VA(
                      mlir::StringAttr::get(&MCtx, "value"),
                      mlir::IntegerAttr::get(IT, Stored));
                  CA.push_back(VA);
                  for (auto &E0 : FAttrs) CA.push_back(E0);
                  mlir::Value QV = emitUnreg("matlab.fi.const", {}, IT, L, CA);
                  mlir::NamedAttribute SCal(
                      mlir::StringAttr::get(&MCtx, "callee"),
                      mlir::StringAttr::get(&MCtx, "__subscript_store"));
                  llvm::SmallVector<mlir::NamedAttribute, 8> SA;
                  SA.push_back(SCal);
                  for (auto &E0 : FAttrs) SA.push_back(E0);
                  emitUnregOp("matlab.call_builtin", {Z, KV, QV},
                              {mlir::NoneType::get(&MCtx)}, L, SA);
                }
                return Z;
              }
            }
          }
        }
      }

      /* Fixed-Point Designer constructor: `fi(value, signed, WL, FL)` and
       * its shorter variants. When all spec args are constants — the
       * common case — we constant-fold to an `arith.constant` of the
       * stored integer plus fi_* attributes; otherwise we emit a call to
       * matlab_fi_quantize_{s,u} and tag the result. Result type comes
       * from Sema (mapType picks the smallest native int). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "fi" && !C.Args.empty()) {
        const Type *ResT = E.Ty ? E.Ty : TC.any();
        if (ResT->K == Type::Kind::Array) {
          auto &AT = static_cast<const ArrayType &>(*ResT);
          if (AT.Elt == Dtype::Fixed && AT.FxSpec) {
            FixedSpec Spec = *AT.FxSpec;
            mlir::Type StorTy = mirTy(ResT);
            auto Attrs = buildFixedAttrs(&MCtx, Spec);
            // Constant-fold scalar literal inputs (the apply_gain shape).
            auto isLiteralFold = [](const Expr *X) -> bool {
              if (!X) return false;
              if (X->Kind == NodeKind::IntegerLiteral ||
                  X->Kind == NodeKind::FPLiteral) return true;
              if (auto *U = dynamic_cast<const UnaryOpExpr *>(X))
                if (U->Op == UnOp::Minus || U->Op == UnOp::Plus)
                  return U->Operand && (
                      U->Operand->Kind == NodeKind::IntegerLiteral ||
                      U->Operand->Kind == NodeKind::FPLiteral);
              return false;
            };
            if (AT.S.K == Shape::Rank::Scalar && isLiteralFold(C.Args[0])) {
              double Val = foldFloat(C.Args[0]);
              int64_t Stored = Spec.Signed
                  ? quantizeFixedSigned(Val, Spec)
                  : (int64_t)quantizeFixedUnsigned(Val, Spec);
              auto IT = mlir::dyn_cast<mlir::IntegerType>(StorTy);
              if (!IT) IT = mlir::IntegerType::get(&MCtx, 64);
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::IntegerAttr::get(IT, Stored));
              llvm::SmallVector<mlir::NamedAttribute, 8> A;
              A.push_back(VA);
              for (auto &E0 : Attrs) A.push_back(E0);
              return emitUnreg("matlab.fi.const", {}, IT, L, A);
            }
            // Runtime quantize for non-literal value.
            mlir::Value V = lowerExpr(*C.Args[0]);
            // Phase 5.6 Stage A.1 — fi-on-fi re-cast. When the
            // input is itself a fi-typed value (Sema's type is
            // an Array{Fixed, ...}), emit a *clamp* cast carrying
            // `fi_lhs_*` attrs naming the source spec.
            // LowerFixedPoint's clamp path then does the proper
            // shift+saturate+truncate; otherwise we'd round-trip
            // through f64 and fail downstream because the
            // integer-scaled-by-2^FL semantic would be wrong.
            // Trigger examples in fir / sequential_processor:
            //   delay_line = [fi(x, 1, 16, 14), ...];   % x is fi(1,16,14)
            //   y = fi(full_res, 1, 16, 12, 'Saturate'); % fi-of-fi
            const Type *InT = C.Args[0]->Ty;
            std::optional<FixedSpec> InFi;
            if (InT && InT->K == Type::Kind::Array) {
              auto &IAT = static_cast<const ArrayType &>(*InT);
              if (IAT.Elt == Dtype::Fixed && IAT.FxSpec)
                InFi = *IAT.FxSpec;
            }
            if (InFi && V && mlir::isa<mlir::IntegerType>(V.getType())) {
              auto I1 = mlir::IntegerType::get(&MCtx, 1);
              auto I32 = mlir::IntegerType::get(&MCtx, 32);
              llvm::SmallVector<mlir::NamedAttribute, 12> A;
              for (auto &E0 : Attrs) A.push_back(E0);
              A.emplace_back(mlir::StringAttr::get(&MCtx, "fi_clamp"),
                             mlir::IntegerAttr::get(I1, 1));
              A.emplace_back(mlir::StringAttr::get(&MCtx, "fi_lhs_signed"),
                             mlir::IntegerAttr::get(I1, InFi->Signed ? 1 : 0));
              A.emplace_back(mlir::StringAttr::get(&MCtx, "fi_lhs_wl"),
                             mlir::IntegerAttr::get(I32, (int64_t)InFi->WordLength));
              A.emplace_back(mlir::StringAttr::get(&MCtx, "fi_lhs_fl"),
                             mlir::IntegerAttr::get(I32, (int64_t)InFi->FractionLength));
              return emitUnreg("matlab.fi.cast", {V}, StorTy, L, A);
            }
            // Cast V to f64 if it isn't already (callee expects double).
            if (V && !mlir::isa<mlir::Float64Type>(V.getType())) {
              auto F64 = mlir::Float64Type::get(&MCtx);
              if (mlir::isa<mlir::IntegerType>(V.getType()))
                V = mlir::arith::SIToFPOp::create(B, L, F64, V);
              else if (mlir::isa<mlir::Float32Type>(V.getType()))
                V = mlir::arith::ExtFOp::create(B, L, F64, V);
            }
            llvm::SmallVector<mlir::NamedAttribute, 8> A;
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx,
                    Spec.Signed ? "matlab_fi_quantize_s"
                                : "matlab_fi_quantize_u"));
            A.push_back(Cal);
            for (auto &E0 : Attrs) A.push_back(E0);
            // Result type is the storage class — LowerFixedPoint inserts
            // the truncate from i64 to this width.
            return emitUnreg("matlab.fi.cast", {V}, StorTy, L, A);
          }
        }
      }

      /* Dot-method call: `obj.method(args)` where `obj` is pinned to a
       * class whose own methods — or any ancestor's — contain `method`.
       * The mangled name uses the *defining* class, so subclasses reach
       * inherited methods via the ancestor's function without needing
       * duplicate emission. */
      auto findMethod = [](const ClassDef *Start, std::string_view Nm)
          -> std::pair<const ClassDef *, const Function *> {
        for (const ClassDef *CC = Start; CC; CC = CC->Super) {
          for (const Function *Mm : CC->Methods)
            if (Mm && Mm->Name == Nm) return {CC, Mm};
        }
        return {nullptr, nullptr};
      };
      auto findStatic = [](const ClassDef *Start, std::string_view Nm)
          -> std::pair<const ClassDef *, const Function *> {
        for (const ClassDef *CC = Start; CC; CC = CC->Super) {
          for (const Function *Mm : CC->StaticMethods)
            if (Mm && Mm->Name == Nm) return {CC, Mm};
        }
        return {nullptr, nullptr};
      };
      if (auto *FA = dynamic_cast<const FieldAccess *>(C.Callee)) {
        /* Phase 4: containers.Map(...) and containers.Map() — produce
         * an empty matlab_dict. We accept the optional KeyType /
         * ValueType arguments but ignore them (the runtime stores any
         * mix of key/value kinds dynamically). The user calls this
         * once at the start of the binding, so a fresh empty dict is
         * the right thing. */
        if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base)) {
          if (BN->Name == "containers" && FA->Field == "Map") {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_dict_new"));
            return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
          }
        }
        const ClassDef *PCls = nullptr;
        if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
          if (BN->Ref && BN->Ref->PinnedClass) PCls = BN->Ref->PinnedClass;
        /* #191 P3: recover the class from the base's inferred object type so a
         * method call whose base is itself an expression (chained-operator
         * rewrite `(a*b).plus(c)`) dispatches via this path. */
        if (!PCls && FA->Base && FA->Base->Ty &&
            FA->Base->Ty->K == Type::Kind::Object)
          PCls = static_cast<const ObjectType &>(*FA->Base->Ty).Class;
        if (PCls) {
          auto [Owner, Mth] = findMethod(PCls, FA->Field);
          if (Mth) {
            mlir::Value Obj = lowerExpr(*FA->Base);
            llvm::SmallVector<mlir::Value, 4> Args;
            Args.push_back(Obj);
            for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
            std::string Callee = std::string(Owner->Name) + "__" +
                                  std::string(FA->Field);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call", Args,
                /* A matrix / array return flows as !llvm.ptr in the
                 * runtime ABI — coerce so the receiving slot retypes
                 * and downstream indexing lowers. */
                (mlir::isa<mlir::RankedTensorType,
                           mlir::UnrankedTensorType>(RT)
                     ? (mlir::Type)mlir::LLVM::LLVMPointerType::get(&MCtx)
                     : RT),
                L, {Cal});
          }
        }
        /* Static method dispatch: `ClassName.method(args)` — the Base
         * resolves to a Class binding, so lowerExpr on it would try to
         * produce a value. Intercept here and route to the class's
         * static method table (walking the inheritance chain). */
        if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base)) {
          if (BN->Ref && BN->Ref->Kind == BindingKind::Class &&
              BN->Ref->ClassDef) {
            auto [Owner, Mth] = findStatic(BN->Ref->ClassDef, FA->Field);
            if (Mth) {
              llvm::SmallVector<mlir::Value, 4> Args;
              for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
              std::string Callee = std::string(Owner->Name) + "__" +
                                    std::string(FA->Field);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Callee));
              return emitUnreg("matlab.call", Args,
                /* A matrix / array return flows as !llvm.ptr in the
                 * runtime ABI — coerce so the receiving slot retypes
                 * and downstream indexing lowers. */
                (mlir::isa<mlir::RankedTensorType,
                           mlir::UnrankedTensorType>(RT)
                     ? (mlir::Type)mlir::LLVM::LLVMPointerType::get(&MCtx)
                     : RT),
                L, {Cal});
            }
          }
        }
      }
      /* Free-function method call: `method(obj, args)` where `obj` is
       * pinned to a class whose method list contains `method`. Same
       * emission as the dot form. */
      if (N && N->Ref && N->Ref->Kind != BindingKind::Class &&
          !C.Args.empty()) {
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref && AN->Ref->PinnedClass) {
            auto [Owner, Mth] = findMethod(AN->Ref->PinnedClass, N->Name);
            if (Mth) {
              llvm::SmallVector<mlir::Value, 4> Args;
              for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
              std::string Callee = std::string(Owner->Name) + "__" +
                                    std::string(N->Name);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Callee));
              return emitUnreg("matlab.call", Args,
                /* A matrix / array return flows as !llvm.ptr in the
                 * runtime ABI — coerce so the receiving slot retypes
                 * and downstream indexing lowers. */
                (mlir::isa<mlir::RankedTensorType,
                           mlir::UnrankedTensorType>(RT)
                     ? (mlir::Type)mlir::LLVM::LLVMPointerType::get(&MCtx)
                     : RT),
                L, {Cal});
            }
          }
        }
      }
      /* isempty(persistent_var) — route to matlab_persistent_isempty(id)
       * which checks the typed-pointer table directly. Fires on any
       * persistent binding so the "first call initialises" idiom works
       * uniformly regardless of whether the binding will hold a fi
       * array, a regular matrix, or a scalar. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "isempty" && C.Args.size() == 1 && C.Args[0]) {
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref && AN->Ref->Kind == BindingKind::Persistent) {
            int32_t Id = globalSlotId(AN->Ref);
            auto F64 = mlir::Float64Type::get(&MCtx);
            auto I32 = mlir::IntegerType::get(&MCtx, 32);
            mlir::Value IdV = mlir::arith::ConstantOp::create(
                B, L, I32, mlir::IntegerAttr::get(I32, (int64_t)Id));
            llvm::SmallVector<mlir::NamedAttribute, 3> Attrs;
            Attrs.push_back(mlir::NamedAttribute(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_persistent_isempty")));
            // Carry the binding's source name + enclosing function
            // name so Stage F's per-element split can prefix the
            // synthesized scalar persistents with `<name>_<k>`
            // instead of falling back to `buf<idx>_<k>`. Without
            // this, the SV register declarations lose their
            // user-readable identity.
            Attrs.push_back(mlir::NamedAttribute(
                mlir::StringAttr::get(&MCtx, "persistent_name"),
                mlir::StringAttr::get(&MCtx, std::string(AN->Ref->Name))));
            Attrs.push_back(mlir::NamedAttribute(
                mlir::StringAttr::get(&MCtx, "persistent_fn"),
                mlir::StringAttr::get(&MCtx, CurFnName)));
            return emitUnreg("matlab.call_builtin", {IdV}, F64, L, Attrs);
          }
        }
      }

      /* sum / mean on fi arrays — route to the typed-int reduction
       * helper, returning the stored integer (i64). LowerFixedPoint sees
       * the fi_signed/wl/fl attrs to pick the right narrow + cast on the
       * consumer side (typically a (:) clamp). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "sum" || N->Name == "mean") &&
          C.Args.size() == 1 && C.Args[0] && C.Args[0]->Ty &&
          C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AS = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AS.Elt == Dtype::Fixed && AS.FxSpec &&
            AS.S.K != Shape::Rank::Scalar) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          auto I64 = mlir::IntegerType::get(&MCtx, 64);
          std::string Cn = std::string("matlab_mat_") +
              (AS.FxSpec->Signed ? "i64_" : "u64_") + "sum";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Cn));
          llvm::SmallVector<mlir::NamedAttribute, 8> AA;
          AA.push_back(Cal);
          auto Atrs = buildFixedAttrs(&MCtx, *AS.FxSpec);
          for (auto &E0 : Atrs) AA.push_back(E0);
          mlir::Value Sum = emitUnreg("matlab.call_builtin", {V}, I64, L, AA);
          if (N->Name == "mean") {
            // mean = sum / numel. Compute sum/N as integer (truncated)
            // for now; a higher-fidelity path could use FullPrecision
            // division. Emit numel via the typed-int helper, divide.
            auto F64 = mlir::Float64Type::get(&MCtx);
            std::string NumelN = std::string("matlab_mat_") +
                (AS.FxSpec->Signed ? "i64_" : "u64_") + "numel";
            mlir::NamedAttribute NCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, NumelN));
            mlir::Value Nf = emitUnreg("matlab.call_builtin", {V}, F64, L, {NCal});
            mlir::Value Ni = mlir::arith::FPToSIOp::create(B, L, I64, Nf);
            Sum = AS.FxSpec->Signed
                ? (mlir::Value)mlir::arith::DivSIOp::create(B, L, Sum, Ni)
                : (mlir::Value)mlir::arith::DivUIOp::create(B, L, Sum, Ni);
          }
          return Sum;
        }
      }

      /* #233: length / numel on a cell -> matlab_cell_numel. The generic
       * matrix numel reads the cell descriptor as a matlab_mat and returns
       * garbage. (Our cells are 1-D rows, so length == numel.) */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "length" || N->Name == "numel") &&
          C.Args.size() == 1 && C.Args[0]) {
        bool IsCell = (C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Cell);
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (AN->Ref && (CellAllStrBindings.count(AN->Ref) ||
                          (AN->Ref->InferredType &&
                           AN->Ref->InferredType->K == Type::Kind::Cell)))
            IsCell = true;
        if (IsCell) {
          mlir::Value Cv = lowerExpr(*C.Args[0]);
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_cell_numel"));
          return emitUnreg("matlab.call_builtin", {Cv}, F64, L, {Cal});
        }
      }

      /* length / numel / size on a string scalar — MATLAB treats a
       * "..."-style string as a 1x1 string array (one element whose
       * value is the text), so length/numel are 1 and size is [1 1].
       * Without this fold the call survives as matlab.call_builtin
       * over a matlab_string* pointer; the generic matrix lowering
       * downstream then casts the descriptor to matlab_mat and reads
       * its length field as `rows` (the user saw `4 × <heap-garbage>`
       * for size("Test")). Detect via StringBindings or the binding's
       * persisted InferredType; literals fall through StringLiteral. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "length" || N->Name == "numel" ||
           N->Name == "size") &&
          !C.Args.empty() && C.Args[0]) {
        bool IsStr = (C.Args[0]->Kind == NodeKind::StringLiteral);
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref &&
              (StringBindings.count(AN->Ref) ||
               (AN->Ref->InferredType &&
                AN->Ref->InferredType->K == Type::Kind::StringArray)))
            IsStr = true;
        }
        if (C.Args[0]->Ty &&
            C.Args[0]->Ty->K == Type::Kind::StringArray)
          IsStr = true;
        if (IsStr) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          if (N->Name == "length" || N->Name == "numel") {
            return mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, 1.0));
          }
          /* size("..."): single-arg returns [1 1]; size(s, k) returns 1
           * for any k since strings are 1x1. */
          if (C.Args.size() == 1) {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx,
                                       "matlab_string_size_scalar"));
            return emitUnreg("matlab.call_builtin", {},
                             mlir::LLVM::LLVMPointerType::get(&MCtx),
                             L, {Cal});
          }
          /* size(s, dim) — for any dim, a string scalar reports 1. */
          return mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, 1.0));
        }
      }
      /* length/numel/size on fi arrays — route to the typed-int matrix
       * shape helpers. Sema already returns scalar Double, so the result
       * type stays f64. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "length" || N->Name == "numel" ||
           N->Name == "size") &&
          !C.Args.empty() && C.Args[0] && C.Args[0]->Ty &&
          C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AS = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AS.Elt == Dtype::Fixed && AS.FxSpec &&
            AS.S.K != Shape::Rank::Scalar) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          auto F64 = mlir::Float64Type::get(&MCtx);
          if (N->Name == "length" || N->Name == "numel") {
            std::string Cn = std::string("matlab_mat_") +
                (AS.FxSpec->Signed ? "i64_" : "u64_") +
                std::string(N->Name);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Cn));
            return emitUnreg("matlab.call_builtin", {V}, F64, L, {Cal});
          }
          // size(A, k) -> scalar; size(A) -> 1x2 row vector (deferred —
          // the FIR shape uses size_dim only).
          if (C.Args.size() >= 2) {
            mlir::Value Dim = lowerExpr(*C.Args[1]);
            std::string Cn = std::string("matlab_mat_") +
                (AS.FxSpec->Signed ? "i64_" : "u64_") + "size_dim";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Cn));
            return emitUnreg("matlab.call_builtin", {V, Dim}, F64, L, {Cal});
          }
        }
      }

      /* disp(fi_array) — route to the typed-int matrix disp helper
       * which renders each element as its real-world value. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1 && C.Args[0] &&
          C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AT0 = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AT0.Elt == Dtype::Fixed && AT0.FxSpec &&
            AT0.S.K != Shape::Rank::Scalar) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          auto I8 = mlir::IntegerType::get(&MCtx, 8);
          mlir::Value WL = mlir::arith::ConstantOp::create(
              B, L, I8,
              mlir::IntegerAttr::get(I8, (int64_t)AT0.FxSpec->WordLength));
          mlir::Value FL = mlir::arith::ConstantOp::create(
              B, L, I8,
              mlir::IntegerAttr::get(I8, (int64_t)AT0.FxSpec->FractionLength));
          llvm::StringRef Callee = AT0.FxSpec->Signed
              ? "matlab_mat_i64_disp" : "matlab_mat_u64_disp";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          return emitUnreg("matlab.call_builtin", {V, WL, FL},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }

      /* disp(fi_value) — render the real-world value via the runtime
       * helper, passing the storage int + WL + FL. The helper picks the
       * right printf format. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1 && C.Args[0] &&
          C.Args[0]->Ty && C.Args[0]->Ty->K == Type::Kind::Array) {
        auto &AT = static_cast<const ArrayType &>(*C.Args[0]->Ty);
        if (AT.Elt == Dtype::Fixed && AT.FxSpec &&
            AT.S.K == Shape::Rank::Scalar) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          // Sign- or zero-extend to i64 for the runtime call.
          auto I64 = mlir::IntegerType::get(&MCtx, 64);
          auto I8 = mlir::IntegerType::get(&MCtx, 8);
          mlir::Value Wide = V;
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
            if (IT.getWidth() < 64) {
              if (AT.FxSpec->Signed)
                Wide = mlir::arith::ExtSIOp::create(B, L, I64, V);
              else
                Wide = mlir::arith::ExtUIOp::create(B, L, I64, V);
            }
          }
          mlir::Value WL = mlir::arith::ConstantOp::create(
              B, L, I8,
              mlir::IntegerAttr::get(I8, (int64_t)AT.FxSpec->WordLength));
          mlir::Value FL = mlir::arith::ConstantOp::create(
              B, L, I8,
              mlir::IntegerAttr::get(I8, (int64_t)AT.FxSpec->FractionLength));
          llvm::StringRef Callee = AT.FxSpec->Signed
              ? "matlab_fi_disp_s" : "matlab_fi_disp_u";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          return emitUnreg("matlab.call_builtin", {Wide, WL, FL},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }

      /* Generic `methodName(obj, args)` -> `obj.methodName(args)`
       * dispatch: when the callee is a builtin name AND the first
       * arg is class-pinned AND that class (or any ancestor) defines
       * a method with the same name, route to the class method
       * instead of the generic builtin.  Covers MATLAB's function-
       * style method call idiom — `reset(crc)`, `release(filter)`,
       * `clone(obj)`, etc. — without needing per-name carve-outs.
       *
       * Excludes `disp` because the disp-on-class path immediately
       * below also handles the no-method case (falls back to the
       * runtime-helper renderer for tf / etc.); the generic path
       * would short-circuit before that fallback fires.  Excludes
       * `step` because the bare `obj(args)` sugar (in the CallOrIndex
       * Index branch) already routes there and we'd double-handle. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          !C.Args.empty() && C.Args[0] &&
          N->Name != "disp" && N->Name != "step") {
        /* The first-arg class can come from either a NameExpr
         * (most common: `design(d, freq)` where d is a class-pinned
         * variable) OR a class ctor call (`design(AntDipole(), f)`
         * — pin comes from the ctor's class).  Same shape as the
         * Resolver's `argPin` helper. */
        const ClassDef *FirstCls = nullptr;
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref) FirstCls = AN->Ref->PinnedClass;
        }
        if (!FirstCls) {
          if (auto *CI = dynamic_cast<const CallOrIndex *>(C.Args[0])) {
            if (auto *NX = dynamic_cast<const NameExpr *>(CI->Callee))
              if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
                  NX->Ref->ClassDef)
                FirstCls = NX->Ref->ClassDef;
          }
        }
        if (FirstCls) {
          const ClassDef *Owner = nullptr;
          const Function *Method = nullptr;
          for (const ClassDef *CC = FirstCls; CC; CC = CC->Super) {
            for (const Function *Mm : CC->Methods)
              if (Mm && Mm->Name == N->Name) {
                Owner = CC; Method = Mm; break;
              }
            if (Owner) break;
          }
            if (Owner) {
              llvm::SmallVector<mlir::Value, 4> Args;
              for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
              std::string Callee = std::string(Owner->Name) + "__" +
                                    std::string(N->Name);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Callee));
              /* Method bodies may return scalar f64, matrix ptr, or
               * void — pass through the call-site's expected type RT
               * when concrete and default to NoneType otherwise.  The
               * runtime helpers + downstream passes handle the type
               * widening if needed.  EXCEPT: when the method's first
               * output is class-pinned (e.g. `design(antenna, freq)`
               * returning a fresh AntDipole), the result must flow
               * as `!llvm.ptr` so the LHS slot retypes and the
               * downstream method-dispatch on it sees a class
               * instance.  Without this, `d = design(...)` stores a
               * `none`-typed value into a class-pinned slot and the
               * next `antennaGain(d, ...)` reads `d` as `none` and
               * misdispatches. */
              mlir::Type ResTy;
              auto PtrTyCM = mlir::LLVM::LLVMPointerType::get(&MCtx);
              bool ReturnsClass =
                  Method && !Method->OutputRefs.empty() &&
                  Method->OutputRefs.front() &&
                  Method->OutputRefs.front()->PinnedClass != nullptr;
              if (ReturnsClass) {
                ResTy = (mlir::Type)PtrTyCM;
              } else if (mlir::isa<mlir::RankedTensorType,
                                   mlir::UnrankedTensorType>(RT)) {
                /* A matrix / array return — the runtime ABI carries
                 * matrices as !llvm.ptr (matlab_mat* descriptors), so
                 * the call result must flow as ptr for the receiving
                 * slot to retype and downstream indexing to lower. */
                ResTy = (mlir::Type)PtrTyCM;
              } else if (mlir::isa<mlir::NoneType>(RT)) {
                ResTy = (mlir::Type)mlir::NoneType::get(&MCtx);
              } else {
                ResTy = RT;
              }
              return emitUnreg("matlab.call", Args, ResTy, L, {Cal});
            }
          }
        }

      /* disp(obj.Field) where `obj` is a class instance — route to
       * the runtime-dispatched `matlab_obj_disp_field` so the
       * property's stored kind (scalar / matrix / string / class
       * instance) picks the correct disp variant at runtime.  Without
       * this, a property holding a string (kind=3) read through the
       * static-type-inferred `matlab_obj_get_f64` returns 0.0 and
       * `disp` prints the wrong thing.  Restricted to the bare
       * `disp(NameExpr.Field)` shape — composite expressions
       * (`disp(obj.A + obj.B)`) still fall through to the standard
       * matrix/scalar paths.
       *
       * Skipped when `Field` names a Dependent property: those have
       * no backing storage and need to flow through `get.<Field>`
       * dispatch (FieldAccess lowering site below).  The runtime
       * helper would do `struct_find_field("Area") → -1` and print
       * a blank line, masking the computed value. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1 && C.Args[0]) {
        if (auto *FA = dynamic_cast<const FieldAccess *>(C.Args[0])) {
          if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base)) {
            if (BN->Ref && BN->Ref->PinnedClass) {
              bool IsDependent = false;
              for (const ClassDef *CC = BN->Ref->PinnedClass; CC; CC = CC->Super) {
                for (const auto &P : CC->Props)
                  if (P.Name == FA->Field && P.Dependent) {
                    IsDependent = true; break;
                  }
                if (IsDependent) break;
              }
              if (!IsDependent) {
                auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
                mlir::Value Obj = lowerExpr(*FA->Base);
                if (Obj.getType() != PtrTy) Obj.setType(PtrTy);
                mlir::Value NameV = emitFieldNameChar(FA->Field, L);
                mlir::NamedAttribute Cal(
                    mlir::StringAttr::get(&MCtx, "callee"),
                    mlir::StringAttr::get(&MCtx, "matlab_obj_disp_field"));
                return emitUnreg("matlab.call_builtin", {Obj, NameV},
                                  mlir::NoneType::get(&MCtx), L, {Cal});
              }
            }
          }
        }
      }

      /* disp(obj) where `obj` is a class instance whose class (or any
       * ancestor) defines `disp` as a method — route to the overload
       * instead of the generic matrix/scalar disp. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1) {
        if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref && AN->Ref->PinnedClass) {
            const ClassDef *Owner = nullptr;
            for (const ClassDef *CC = AN->Ref->PinnedClass; CC; CC = CC->Super) {
              for (const Function *Mm : CC->Methods)
                if (Mm && Mm->Name == "disp") { Owner = CC; break; }
              if (Owner) break;
            }
            if (Owner) {
              mlir::Value Obj = lowerExpr(*C.Args[0]);
              std::string Callee = std::string(Owner->Name) + "__disp";
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Callee));
              return emitUnreg("matlab.call", {Obj},
                               mlir::NoneType::get(&MCtx), L, {Cal});
            }
            /* §3.1: disp(tf) — no MATLAB-side disp method on the tf
             * classdef; route to the runtime helper that pulls the
             * Numerator / Denominator properties and renders the
             * canonical centred-fraction s-domain layout. */
            if (AN->Ref->PinnedClass->Name == "tf") {
              auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
              mlir::Value Obj = lowerExpr(*C.Args[0]);
              if (Obj.getType() != PtrTy) Obj.setType(PtrTy);
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_tf_disp"));
              return emitUnreg("matlab.call_builtin", {Obj},
                               mlir::NoneType::get(&MCtx), L, {Cal});
            }
          }
        }
      }
      /* §3.1 — model-object short forms.
       *
       * `step(sys)` / `bode(sys, w)` / `pole(sys)` / `dcgain(sys)` /
       * `lsim(sys, u, dt)` / `bandwidth(sys)` etc. take a class-
       * pinned first argument and dispatch to the matching matrix-
       * arg primitive (`step_ss`, `bode_ss`, …) by unpacking the
       * relevant properties via `matlab_obj_get_mat`. The short
       * forms only fire when the first arg is a NameExpr pinned to
       * the matching class; matrix-arg call sites (e.g. `pole(A)`)
       * fall through to the existing builtin path unchanged. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          C.Args.size() >= 1 && C.Args[0]) {
        auto *AN0 = dynamic_cast<const NameExpr *>(C.Args[0]);
        const ClassDef *Cls0 = (AN0 && AN0->Ref) ? AN0->Ref->PinnedClass
                                                  : nullptr;
        /* Recognise inline constructor calls (e.g. `c2d(ss(A,B,C,D),
         * Ts, 'zoh')`) so the short-form dispatch fires the same as
         * for a class-pinned binding. Without this, callers have to
         * spell the binding twice — once to assign, once to use —
         * which breaks user-facing examples that compose model
         * construction with conversion in a single expression. */
        if (!Cls0) {
          if (auto *AC0 = dynamic_cast<const CallOrIndex *>(C.Args[0])) {
            if (auto *CFN = dynamic_cast<const NameExpr *>(AC0->Callee)) {
              if (CFN->Ref && CFN->Ref->Kind == BindingKind::Class)
                Cls0 = CFN->Ref->ClassDef;
            }
          }
        }
        const llvm::StringRef Cn0 = Cls0
            ? llvm::StringRef(Cls0->Name.data(), Cls0->Name.size())
            : llvm::StringRef();
        const auto &Nm = N->Name;
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto getProp = [&](mlir::Value Obj,
                           llvm::StringRef Field) -> mlir::Value {
          mlir::Value FieldName = emitFieldNameChar(Field, L);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_obj_get_mat"));
          return emitUnreg("matlab.call_builtin", {Obj, FieldName},
                            PtrTy, L, {Cal});
        };
        auto loadObj = [&](const Expr *X) -> mlir::Value {
          mlir::Value V = lowerExpr(*X);
          if (V.getType() != PtrTy) V.setType(PtrTy);
          return V;
        };

        /* ============================================================
         * Global Optimization Toolbox Tier-1 — stochastic global
         * solvers.  Each takes the objective as a function handle
         * (arg 0, retyped to ptr by LowerAnonCalls); the runtime
         * entries are 5-arg (fn, nvars/x0, lb, ub, hybrid).  We remap
         * the MATLAB call forms and inject the hybrid flag (Tier-1
         * always polishes with fmincon — the options-controlled
         * HybridFcn is Tier-6).
         * ============================================================ */
        /* Global Optimization Tier-6 — ga with an optimoptions object.
         * The trailing arg is the options carrier; route to the 6-arg
         * `matlab_gads_ga_opts` runtime entry (reads PopulationSize /
         * MaxGenerations / IntCon).  Supported call forms:
         *   ga(fun, nvars, lb, ub, opts)                          (5-arg)
         *   ga(fun, nvars, A, b, Aeq, beq, lb, ub, opts)          (9-arg)
         *   ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, opts) (10-arg,
         *      the canonical full signature — nonlcon must be [] today). */
        if (Nm == "ga" &&
            (C.Args.size() == 5 || C.Args.size() == 9 || C.Args.size() == 10)) {
          mlir::Value Fn = lowerExpr(*C.Args[0]);
          if (Fn.getType() != PtrTy) Fn.setType(PtrTy);
          mlir::Value Nv = lowerExpr(*C.Args[1]);
          mlir::Value Lb, Ub, Opts;
          if (C.Args.size() == 5)      { Lb = lowerExpr(*C.Args[2]); Ub = lowerExpr(*C.Args[3]); Opts = loadObj(C.Args[4]); }
          else if (C.Args.size() == 9) { Lb = lowerExpr(*C.Args[6]); Ub = lowerExpr(*C.Args[7]); Opts = loadObj(C.Args[8]); }
          else                         { Lb = lowerExpr(*C.Args[6]); Ub = lowerExpr(*C.Args[7]); Opts = loadObj(C.Args[9]); }
          mlir::Value Hy = emitUnreg("matlab.const_float", {}, F64, L,
                                     {mlir::NamedAttribute(
                                         mlir::StringAttr::get(&MCtx, "value"),
                                         mlir::FloatAttr::get(F64, 1.0))});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_gads_ga_opts"));
          return emitUnreg("matlab.call_builtin", {Fn, Nv, Lb, Ub, Hy, Opts}, PtrTy, L, {Cal});
        }

        if (Nm == "ga" || Nm == "particleswarm" || Nm == "simulannealbnd") {
          auto f64const = [&](double v) -> mlir::Value {
            return emitUnreg("matlab.const_float", {}, F64, L,
                             {mlir::NamedAttribute(
                                 mlir::StringAttr::get(&MCtx, "value"),
                                 mlir::FloatAttr::get(F64, v))});
          };
          auto emptyMat = [&]() -> mlir::Value {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_empty_mat"));
            return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
          };
          mlir::Value Fn = lowerExpr(*C.Args[0]);   /* objective handle */
          if (Fn.getType() != PtrTy) Fn.setType(PtrTy);
          /* Matrix args (x0 / lb / ub) are lowered with plain lowerExpr
           * (NOT loadObj's setType) — inline matrix literals must keep
           * their tensor result type so the concat op lowers; the
           * pde_table loose-match then coerces tensor→ptr at the call. */
          /* Arg 1 is `nvars` (f64 scalar) for ga / particleswarm, but the
           * `x0` start-point matrix for simulannealbnd. */
          mlir::Value Arg1 = (C.Args.size() >= 2) ? lowerExpr(*C.Args[1])
                                                  : f64const(1.0);
          /* lb / ub extraction by call form. */
          mlir::Value Lb, Ub;
          if (Nm == "ga" && C.Args.size() == 8) {
            Lb = lowerExpr(*C.Args[6]); Ub = lowerExpr(*C.Args[7]);   /* (fun,nvars,A,b,Aeq,beq,lb,ub) */
          } else if (C.Args.size() == 4) {
            Lb = lowerExpr(*C.Args[2]); Ub = lowerExpr(*C.Args[3]);   /* (fun,nvars/x0,lb,ub) */
          } else {
            Lb = emptyMat(); Ub = emptyMat();                        /* unbounded fallback */
          }
          const char *rt = (Nm == "ga") ? "matlab_gads_ga"
                         : (Nm == "particleswarm") ? "matlab_gads_particleswarm"
                         : "matlab_gads_simulannealbnd";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin",
                           {Fn, Arg1, Lb, Ub, f64const(1.0)}, PtrTy, L, {Cal});
        }

        /* Global Optimization Tier-3 — patternsearch(fun, x0, A, b, Aeq,
         * beq, lb, ub).  Deterministic direct search; x0 is a matrix
         * start point (arg 1) and there is NO hybrid arg (the mesh
         * refinement is the convergence).  4-arg runtime. */
        if (Nm == "patternsearch") {
          auto emptyMat = [&]() -> mlir::Value {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_empty_mat"));
            return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
          };
          mlir::Value Fn = lowerExpr(*C.Args[0]);
          if (Fn.getType() != PtrTy) Fn.setType(PtrTy);
          mlir::Value X0 = (C.Args.size() >= 2) ? lowerExpr(*C.Args[1]) : emptyMat();
          mlir::Value Lb, Ub;
          if (C.Args.size() == 8) {            /* (fun,x0,A,b,Aeq,beq,lb,ub) */
            Lb = lowerExpr(*C.Args[6]); Ub = lowerExpr(*C.Args[7]);
          } else if (C.Args.size() == 4) {     /* (fun,x0,lb,ub) */
            Lb = lowerExpr(*C.Args[2]); Ub = lowerExpr(*C.Args[3]);
          } else {
            Lb = emptyMat(); Ub = emptyMat();
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_gads_patternsearch"));
          return emitUnreg("matlab.call_builtin", {Fn, X0, Lb, Ub}, PtrTy, L, {Cal});
        }

        /* Global Optimization Tier-5 — gamultiobj / paretosearch.  Same
         * arg shape as ga (nvars at arg 1; 8-arg `(fun,nvars,A,b,Aeq,beq,
         * lb,ub)` / 4-arg `(fun,nvars,lb,ub)`), but the objective returns
         * a vector (handled by the vector-objective retype) and there is
         * NO hybrid arg.  Returns the Pareto set (k×nvars). */
        if (Nm == "gamultiobj" || Nm == "paretosearch") {
          auto emptyMat = [&]() -> mlir::Value {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_empty_mat"));
            return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
          };
          mlir::Value Fn = lowerExpr(*C.Args[0]);
          if (Fn.getType() != PtrTy) Fn.setType(PtrTy);
          mlir::Value Nv = (C.Args.size() >= 2) ? lowerExpr(*C.Args[1])
                                                : emptyMat();   /* nvars (f64) */
          mlir::Value Lb, Ub;
          if (C.Args.size() == 8) { Lb = lowerExpr(*C.Args[6]); Ub = lowerExpr(*C.Args[7]); }
          else if (C.Args.size() == 4) { Lb = lowerExpr(*C.Args[2]); Ub = lowerExpr(*C.Args[3]); }
          else { Lb = emptyMat(); Ub = emptyMat(); }
          const char *rt = (Nm == "gamultiobj") ? "matlab_gads_gamultiobj"
                                                 : "matlab_gads_paretosearch";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Fn, Nv, Lb, Ub}, PtrTy, L, {Cal});
        }

        /* Global Optimization Tier-4 — surrogateopt(fun, lb, ub).  No
         * start point (samples within bounds); lb/ub are args 1,2.
         * Injects the hybrid flag (final fmincon polish). */
        if (Nm == "surrogateopt" && C.Args.size() == 3) {
          auto f64c = [&](double v) -> mlir::Value {
            return emitUnreg("matlab.const_float", {}, F64, L,
                             {mlir::NamedAttribute(
                                 mlir::StringAttr::get(&MCtx, "value"),
                                 mlir::FloatAttr::get(F64, v))});
          };
          mlir::Value Fn = lowerExpr(*C.Args[0]);
          if (Fn.getType() != PtrTy) Fn.setType(PtrTy);
          mlir::Value Lb = lowerExpr(*C.Args[1]);
          mlir::Value Ub = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_gads_surrogateopt"));
          return emitUnreg("matlab.call_builtin",
                           {Fn, Lb, Ub, f64c(1.0)}, PtrTy, L, {Cal});
        }

        /* Global Optimization Tier-2 — createOptimProblem('fmincon',
         * 'objective',@f,'x0',x0,'lb',lb,'ub',ub).  Scan the name-value
         * pairs, stash the objective handle + x0/lb/ub into the runtime
         * thread-local problem context (matlab_gads_make_problem), and
         * return its marker.  The objective handle is lowered as a ptr
         * (LowerAnonCalls retypes operand 0). */
        if (Nm == "createOptimProblem") {
          mlir::Value Obj_, X0_, Lb_, Ub_;
          auto emptyMat2 = [&]() -> mlir::Value {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_empty_mat"));
            return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
          };
          for (size_t i = 0; i + 1 < C.Args.size(); ++i) {
            std::string key;
            if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[i]))
              key = CL->Value;
            else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[i]))
              key = SL->Value;
            if (key.empty()) continue;
            const Expr *valE = C.Args[i + 1];
            if (key == "objective") { Obj_ = lowerExpr(*valE);
                                      if (Obj_.getType() != PtrTy) Obj_.setType(PtrTy); }
            else if (key == "x0")    X0_ = lowerExpr(*valE);
            else if (key == "lb")    Lb_ = lowerExpr(*valE);
            else if (key == "ub")    Ub_ = lowerExpr(*valE);
          }
          if (!Obj_) Obj_ = emptyMat2();
          if (!X0_)  X0_  = emptyMat2();
          if (!Lb_)  Lb_  = emptyMat2();
          if (!Ub_)  Ub_  = emptyMat2();
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_gads_make_problem"));
          return emitUnreg("matlab.call_builtin", {Obj_, X0_, Lb_, Ub_},
                           PtrTy, L, {Cal});
        }
        /* Global Optimization Tier-2 — run(solver, problem [, k]).  The
         * objective + bounds ride in the thread-local set by
         * createOptimProblem; the solver object is forwarded so the
         * runtime can read its class (MultiStart vs GlobalSearch) and
         * branch.  Dispatching at runtime — rather than on a Sema-pinned
         * type — makes `run` work in the line-by-line REPL too, where
         * cross-line class pinning is not retained.  Gated to a non-string
         * first arg so a future `run('script.m')` is not hijacked. */
        if (Nm == "run" && C.Args.size() >= 2 &&
            !dynamic_cast<const CharLiteral *>(C.Args[0]) &&
            !dynamic_cast<const StringLiteral *>(C.Args[0])) {
          auto f64c = [&](double v) -> mlir::Value {
            return emitUnreg("matlab.const_float", {}, F64, L,
                             {mlir::NamedAttribute(
                                 mlir::StringAttr::get(&MCtx, "value"),
                                 mlir::FloatAttr::get(F64, v))});
          };
          mlir::Value Solver = loadObj(C.Args[0]);
          mlir::Value K = (C.Args.size() >= 3) ? lowerExpr(*C.Args[2])
                                               : f64c(20.0);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_gads_run"));
          return emitUnreg("matlab.call_builtin", {Solver, K}, PtrTy, L, {Cal});
        }

        /* ===========================================================
         * Statistics Toolbox Tier-1 — distribution objects.
         * =========================================================== */
        /* Distribution name string -> code (1=Normal,2=Exponential,3=Uniform). */
        auto distCode = [&](const Expr *E) -> double {
          std::string s;
          if (auto *CL = dynamic_cast<const CharLiteral *>(E)) s = CL->Value;
          else if (auto *SL = dynamic_cast<const StringLiteral *>(E)) s = SL->Value;
          if (s == "Exponential" || s == "exponential") return 2.0;
          if (s == "Uniform" || s == "uniform") return 3.0;
          return 1.0;  /* Normal default */
        };
        auto f64lit = [&](double v) -> mlir::Value {
          return emitUnreg("matlab.const_float", {}, F64, L,
                           {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                                                 mlir::FloatAttr::get(F64, v))});
        };
        auto setObjF64 = [&](mlir::Value Obj, const char *fld, mlir::Value Val) {
          mlir::Value NameV = emitFieldNameChar(fld, L);
          mlir::NamedAttribute SetCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_obj_set_f64"));
          emitUnregOp("matlab.call_builtin", {Obj, NameV, Val},
                      {mlir::NoneType::get(&MCtx)}, L, {SetCal});
        };

        /* makedist('Normal','mu',M,'sigma',S) — alloc the ProbDistUnivParam
         * shell and write DistCode + the named params (lower/upper alias to
         * mu/sigma for the Uniform). */
        if (Nm == "makedist" && C.Args.size() >= 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ProbDistUnivParam__ProbDistUnivParam"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          setObjF64(Obj, "DistCode", f64lit(distCode(C.Args[0])));
          for (size_t i = 1; i + 1 < C.Args.size(); i += 2) {
            std::string key;
            if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[i])) key = CL->Value;
            else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[i])) key = SL->Value;
            if (key.empty()) continue;
            const char *fld = (key == "mu" || key == "lower")    ? "mu"
                            : (key == "sigma" || key == "upper") ? "sigma"
                            : nullptr;
            if (!fld) continue;
            setObjF64(Obj, fld, lowerExpr(*C.Args[i + 1]));
          }
          return Obj;
        }

        /* fitdist(x, 'Normal') — alloc the shell, then MLE-populate it via
         * the runtime (alloc-then-populate, like idss / EKF). */
        if (Nm == "fitdist" && C.Args.size() >= 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ProbDistUnivParam__ProbDistUnivParam"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::Value X = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_stats_fitdist_init"));
          emitUnregOp("matlab.call_builtin", {Obj, X, f64lit(distCode(C.Args[1]))},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* pdf/cdf/icdf(pd, x) and random(pd, m, n) — runtime-dispatched on
         * the distribution object's class (REPL-safe; gated to a non-string
         * first arg so the pdf('Normal',…) name form is not hijacked). */
        if ((Nm == "pdf" || Nm == "cdf" || Nm == "icdf" || Nm == "random") &&
            C.Args.size() >= 2 &&
            !dynamic_cast<const CharLiteral *>(C.Args[0]) &&
            !dynamic_cast<const StringLiteral *>(C.Args[0])) {
          mlir::Value Pd = loadObj(C.Args[0]);
          const char *rt = (Nm == "pdf")  ? "matlab_stats_pd_pdf"
                         : (Nm == "cdf")  ? "matlab_stats_pd_cdf"
                         : (Nm == "icdf") ? "matlab_stats_pd_icdf"
                                          : "matlab_stats_pd_random";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          if (Nm == "random") {
            mlir::Value Mm = lowerExpr(*C.Args[1]);
            mlir::Value Nn = (C.Args.size() >= 3) ? lowerExpr(*C.Args[2]) : f64lit(1.0);
            return emitUnreg("matlab.call_builtin", {Pd, Mm, Nn}, PtrTy, L, {Cal});
          }
          mlir::Value Xx = lowerExpr(*C.Args[1]);
          return emitUnreg("matlab.call_builtin", {Pd, Xx}, PtrTy, L, {Cal});
        }

        /* Stats Tier-3 — fitlm(X,y) / fitglm(X,y,…): alloc a LinearModel
         * shell and populate it via the runtime (OLS or logistic IRLS).
         * The fitglm 'Distribution' name-value args are accepted-and-
         * ignored for now (logistic/binomial is the wired link). */
        if ((Nm == "fitlm" || Nm == "fitglm") && C.Args.size() >= 2) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "LinearModel__LinearModel"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::Value Xd = lowerExpr(*C.Args[0]);   /* data: plain lower (inline-matrix safe) */
          mlir::Value Yd = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Nm == "fitglm" ? "matlab_stats_fitglm_init"
                                                          : "matlab_stats_fitlm_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* Stats Tier-5 — fitcknn/fitcnb/fitcdiscr/fitctree/fitcsvm/fitcecoc:
         * alloc a ClassificationModel shell and populate it via the runtime
         * (each `fitc*` maps to its matlab_stats_fit*_init). */
        if ((Nm == "fitcknn" || Nm == "fitcnb" || Nm == "fitcdiscr" ||
             Nm == "fitctree" || Nm == "fitcsvm" || Nm == "fitcecoc") &&
            C.Args.size() >= 2) {
          const char *initSym =
              (Nm == "fitcknn")   ? "matlab_stats_fitknn_init"
            : (Nm == "fitcnb")    ? "matlab_stats_fitnb_init"
            : (Nm == "fitcdiscr") ? "matlab_stats_fitlda_init"
            : (Nm == "fitctree")  ? "matlab_stats_fittree_init"
            : (Nm == "fitcsvm")   ? "matlab_stats_fitsvm_init"
                                  : "matlab_stats_fitecoc_init";
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ClassificationModel__ClassificationModel"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::Value Xd = lowerExpr(*C.Args[0]);
          mlir::Value Yd = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, initSym));
          emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* Stats Tier-6 — ensembles.  fitcensemble(X,y) = bagging (50 trees,
         * all features); TreeBagger(nTrees,X,y) = random forest (featsub<0
         * → √p in the runtime).  Both populate a ClassificationModel
         * (ModelType 7) via matlab_stats_fitensemble_init(obj,X,y,T,fs). */
        if ((Nm == "fitcensemble" || Nm == "TreeBagger") && C.Args.size() >= 2) {
          bool isBagger = (Nm == "TreeBagger");
          auto f64c = [&](double v) -> mlir::Value {
            return emitUnreg("matlab.const_float", {}, F64, L,
                             {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                                                   mlir::FloatAttr::get(F64, v))});
          };
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ClassificationModel__ClassificationModel"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          /* TreeBagger(nTrees, X, y): args shifted by one; featsub = -1 (√p).
           * fitcensemble(X, y): T=50, featsub=0 (all features). */
          mlir::Value Xd = lowerExpr(*C.Args[isBagger ? 1 : 0]);
          mlir::Value Yd = lowerExpr(*C.Args[isBagger ? 2 : 1]);
          mlir::Value Tn = isBagger ? lowerExpr(*C.Args[0]) : f64c(50.0);
          if (Tn.getType() != F64) Tn = f64c(50.0);
          mlir::Value Fs = f64c(isBagger ? -1.0 : 0.0);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_stats_fitensemble_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd, Tn, Fs},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* Stats Tier-6 — bayesopt(fun, lb, ub): GP + EI optimization over a
         * box.  Objective handle at operand 0 (retyped to ptr by
         * LowerAnonCalls); lb/ub plain-lowered (inline-matrix safe). */
        if (Nm == "bayesopt" && C.Args.size() == 3) {
          mlir::Value Fn = lowerExpr(*C.Args[0]);
          if (Fn.getType() != PtrTy) Fn.setType(PtrTy);
          mlir::Value Lb = lowerExpr(*C.Args[1]);
          mlir::Value Ub = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_stats_bayesopt"));
          return emitUnreg("matlab.call_builtin", {Fn, Lb, Ub}, PtrTy, L, {Cal});
        }

        /* Image Processing Tier-3 — fitgeotform2d(moving, fixed, type):
         * alloc an affine2d shell + least-squares populate via the runtime
         * (the type string is materialised by the pde_table coercion). */
        if (Nm == "fitgeotform2d" && C.Args.size() == 3) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "affine2d__affine2d"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::Value Mv = lowerExpr(*C.Args[0]);
          mlir::Value Fx = lowerExpr(*C.Args[1]);
          mlir::Value Ty = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_image_fitgeo_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Mv, Fx, Ty},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* Curve Fitting Tier-1 — fit(x, y, 'polyN'): alloc a cfit shell and
         * populate it via the runtime (center-and-scale + Vandermonde LS).
         * The model-tag string is materialised by the pde_table const_char
         * coercion.  The single-return form (`f = fit(...)` / bare call);
         * the [f,gof,output] multi-return form is handled in the assignment
         * lowering. */
        if (Nm == "fit" && C.Args.size() >= 3) {
          /* Surface fit: a 2-digit poly tag ('poly23') → sfit + bivariate LS. */
          std::string surfTag;
          if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[2])) surfTag = CL->Value;
          else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[2])) surfTag = SL->Value;
          bool isSurf = (surfTag.size() == 6 && surfTag.compare(0, 4, "poly") == 0 &&
                         isdigit(static_cast<unsigned char>(surfTag[4])) &&
                         isdigit(static_cast<unsigned char>(surfTag[5])));
          if (isSurf) {
            mlir::NamedAttribute SCtor(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "sfit__sfit"));
            mlir::Value SObj = emitUnreg("matlab.call", {}, PtrTy, L, {SCtor});
            mlir::Value XY = lowerExpr(*C.Args[0]);
            mlir::Value Zd = lowerExpr(*C.Args[1]);
            mlir::Value Tg = lowerExpr(*C.Args[2]);
            mlir::NamedAttribute SCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_curvefit_fit_surface"));
            emitUnregOp("matlab.call_builtin", {SObj, XY, Zd, Tg},
                        {mlir::NoneType::get(&MCtx)}, L, {SCal});
            return SObj;
          }
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "cfit__cfit"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::Value Xd = lowerExpr(*C.Args[0]);
          mlir::Value Yd = lowerExpr(*C.Args[1]);
          /* Custom equation: a fittype object as the model arg routes to the
           * finite-difference LM (matlab_curvefit_fit_custom). */
          const ClassDef *MdlCls = nullptr;
          if (auto *MN = dynamic_cast<const NameExpr *>(C.Args[2]))
            if (MN->Ref) MdlCls = MN->Ref->PinnedClass;
          if (MdlCls && llvm::StringRef(MdlCls->Name) == "fittype") {
            mlir::Value Ft = lowerExpr(*C.Args[2]);
            if (Ft.getType() != PtrTy) Ft.setType(PtrTy);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_curvefit_fit_custom"));
            emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd, Ft},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Obj;
          }
          mlir::Value Md = lowerExpr(*C.Args[2]);
          if (C.Args.size() >= 4) {                /* fit(x,y,model,opts) */
            mlir::Value Op = lowerExpr(*C.Args[3]);
            if (Op.getType() != PtrTy) Op.setType(PtrTy);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_curvefit_fit_opts"));
            emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd, Md, Op},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Obj;
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_curvefit_fit"));
          emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd, Md},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* Bioinformatics Tier-4 — seqlinkage / seqneighjoin: alloc a phytree
         * shell, then populate it from the distance matrix via the runtime
         * (UPGMA / neighbor-joining).  Arity selects the populate variant:
         * (D), (D, method), (D, method, names_cell).  Mirrors the `fit`
         * alloc-then-populate idiom. */
        if ((Nm == "seqlinkage" || Nm == "seqneighjoin") && C.Args.size() >= 1) {
          bool nj = (Nm == "seqneighjoin");
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "phytree__phytree"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::Value Dv = lowerExpr(*C.Args[0]);
          if (Dv.getType() != PtrTy) Dv.setType(PtrTy);
          llvm::SmallVector<mlir::Value, 4> Args{Obj, Dv};
          const char *Pop;
          if (C.Args.size() >= 3) {
            mlir::Value Mv = lowerExpr(*C.Args[1]);
            mlir::Value Nv = lowerExpr(*C.Args[2]);
            if (Mv.getType() != PtrTy) Mv.setType(PtrTy);
            if (Nv.getType() != PtrTy) Nv.setType(PtrTy);
            Args.push_back(Mv); Args.push_back(Nv);
            Pop = nj ? "matlab_bioinfo_seqneighjoin3" : "matlab_bioinfo_seqlinkage3";
          } else if (C.Args.size() == 2 && !nj) {
            mlir::Value Mv = lowerExpr(*C.Args[1]);
            if (Mv.getType() != PtrTy) Mv.setType(PtrTy);
            Args.push_back(Mv);
            Pop = "matlab_bioinfo_seqlinkage2";
          } else {
            Pop = nj ? "matlab_bioinfo_seqneighjoin1" : "matlab_bioinfo_seqlinkage1";
          }
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                   mlir::StringAttr::get(&MCtx, Pop));
          emitUnregOp("matlab.call_builtin", Args,
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* Curve Fitting Tier-6 — ppform constructors.  spline/pchip/ppmak
         * build a fresh ppform; fnder/fnint transform one.  Each allocs the
         * ppform shell then populates it via the runtime. */
        if ((Nm == "spline" || Nm == "pchip") && C.Args.size() == 2) {
          mlir::NamedAttribute Ctor(mlir::StringAttr::get(&MCtx, "callee"),
                                    mlir::StringAttr::get(&MCtx, "ppform__ppform"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {Ctor});
          mlir::Value Xd = lowerExpr(*C.Args[0]);
          mlir::Value Yd = lowerExpr(*C.Args[1]);
          mlir::Value Kind = emitUnreg("matlab.const_float", {}, F64, L,
              {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                                    mlir::FloatAttr::get(F64, Nm == "pchip" ? 1.0 : 0.0))});
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                   mlir::StringAttr::get(&MCtx, "matlab_curvefit_spline_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Xd, Yd, Kind},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (Nm == "ppmak" && C.Args.size() == 2) {
          mlir::NamedAttribute Ctor(mlir::StringAttr::get(&MCtx, "callee"),
                                    mlir::StringAttr::get(&MCtx, "ppform__ppform"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {Ctor});
          mlir::Value Br = lowerExpr(*C.Args[0]);
          mlir::Value Cf = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                   mlir::StringAttr::get(&MCtx, "matlab_curvefit_ppmak_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Br, Cf},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if ((Nm == "fnder" || Nm == "fnint") && C.Args.size() == 1) {
          mlir::NamedAttribute Ctor(mlir::StringAttr::get(&MCtx, "callee"),
                                    mlir::StringAttr::get(&MCtx, "ppform__ppform"));
          mlir::Value Obj = emitUnreg("matlab.call", {}, PtrTy, L, {Ctor});
          mlir::Value Pp = lowerExpr(*C.Args[0]);
          if (Pp.getType() != PtrTy) Pp.setType(PtrTy);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Nm == "fnint" ? "matlab_curvefit_fnint_init"
                                                         : "matlab_curvefit_fnder_init"));
          emitUnregOp("matlab.call_builtin", {Obj, Pp},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }

        /* cat(dim, A, B, …): dim 3 stacks 2-D planes into a matlab_mat3
         * (matlab_cat3_2 / matlab_cat3_3); dim 1/2 fold pairwise through
         * vertcat / horzcat.  `dim` must be an integer literal. */
        if (Nm == "cat" && C.Args.size() >= 2) {
          if (auto *DimL = dynamic_cast<const IntegerLiteral *>(C.Args[0])) {
            llvm::SmallVector<mlir::Value, 4> Ms;
            for (size_t a = 1; a < C.Args.size(); ++a) Ms.push_back(lowerExpr(*C.Args[a]));
            /* cat(dim, a) of a single operand returns it unchanged — a
             * trailing-singleton stays 2-D (no mat3 wrapper). */
            if (Ms.size() == 1) return Ms[0];
            if (DimL->Text == "3") {
              /* Fold N planes into a slice-major mat3 of depth N: cat3_2 of
               * the first two, then append each remaining plane. No arity
               * cap — cat(3, p1, …, pN) works for any N (Ms.size() >= 2,
               * guaranteed by C.Args.size() >= 3). */
              mlir::NamedAttribute Cal2(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_cat3_2"));
              mlir::Value Acc =
                  emitUnreg("matlab.call_builtin", {Ms[0], Ms[1]}, PtrTy, L, {Cal2});
              for (size_t a = 2; a < Ms.size(); ++a) {
                mlir::NamedAttribute CalN(
                    mlir::StringAttr::get(&MCtx, "callee"),
                    mlir::StringAttr::get(&MCtx, "matlab_cat3_append"));
                Acc = emitUnreg("matlab.call_builtin", {Acc, Ms[a]}, PtrTy, L, {CalN});
              }
              return Acc;
            }
            if (DimL->Text == "4") {
              /* cat(4, mat3_or_mat, ...): stack 3-D images (or 2-D
               * grayscale) into a rank-4 matN.  Fixed-arity entries
               * cover N = 2 / 3 / 4; tuck the implementation against
               * those for the common image-batch sizes.  Arities > 4
               * fall through to a sequence of cat4_2 + (carved) append. */
              const char *Sym = nullptr;
              if (Ms.size() == 2) Sym = "matlab_cat4_2";
              else if (Ms.size() == 3) Sym = "matlab_cat4_3";
              else if (Ms.size() == 4) Sym = "matlab_cat4_4";
              if (Sym) {
                mlir::NamedAttribute Cal(
                    mlir::StringAttr::get(&MCtx, "callee"),
                    mlir::StringAttr::get(&MCtx, Sym));
                return emitUnreg("matlab.call_builtin", Ms, PtrTy, L, {Cal});
              }
            }
            const char *Sym = (DimL->Text == "1") ? "vertcat" : "horzcat";
            mlir::Value Acc = Ms[0];
            for (size_t a = 1; a < Ms.size(); ++a) {
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Sym));
              Acc = emitUnreg("matlab.call_builtin", {Acc, Ms[a]}, PtrTy, L, {Cal});
            }
            return Acc;
          }
        }

        auto rebuildCall = [&](llvm::StringRef Callee,
                               llvm::ArrayRef<mlir::Value> Args,
                               mlir::Type ResTy) -> mlir::Value {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          return emitUnreg("matlab.call_builtin", Args, ResTy, L, {Cal});
        };

        /* pole(sys): for ss → eig(sys.A); for tf → roots(sys.Denominator).
         * Result is a (possibly complex) matrix. */
        if (Nm == "pole" && Cls0 && (Cn0 == "ss" || Cn0 == "tf")) {
          mlir::Value Obj = loadObj(C.Args[0]);
          if (Cn0 == "ss")
            return rebuildCall("eig", {getProp(Obj, "A")}, PtrTy);
          return rebuildCall("roots", {getProp(Obj, "Denominator")}, PtrTy);
        }

        /* dcgain(sys): ss → dcgain_ss(A,B,C,D); tf → dcgain_tf(num,den). */
        if (Nm == "dcgain" && Cls0 && (Cn0 == "ss" || Cn0 == "tf")) {
          mlir::Value Obj = loadObj(C.Args[0]);
          if (Cn0 == "tf")
            return rebuildCall("dcgain_tf",
                               {getProp(Obj, "Numerator"),
                                getProp(Obj, "Denominator")}, PtrTy);
          return rebuildCall("dcgain_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D")},
                             PtrTy);
        }

        /* bandwidth(sys): ss → bandwidth_ss; tf → bandwidth_tf. Returns
         * a scalar f64 (the −3 dB bandwidth in rad/s). */
        if (Nm == "bandwidth" && Cls0 && (Cn0 == "ss" || Cn0 == "tf")) {
          mlir::Value Obj = loadObj(C.Args[0]);
          if (Cn0 == "tf")
            return rebuildCall("bandwidth_tf",
                               {getProp(Obj, "Numerator"),
                                getProp(Obj, "Denominator")}, F64);
          return rebuildCall("bandwidth_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D")},
                             F64);
        }

        /* step(sys [, dt, N]): for ss → step_ss with defaults
         * dt=0.01, N=500 if not provided. Returns y as a column
         * matrix. */
        if (Nm == "step" && Cls0 && (Cn0 == "ss" || Cn0 == "tf")) {
          mlir::Value Obj = loadObj(C.Args[0]);
          /* step(model, t): 2-arg form with a supplied time vector — honour
           * it (derive dt / N from the grid in the runtime). step_*_t reads
           * the time vector and returns one row per sample. */
          if (C.Args.size() == 2 && C.Args[1]) {
            mlir::Value T = lowerExpr(*C.Args[1]);
            if (T.getType() != PtrTy) T.setType(PtrTy);
            if (Cn0 == "tf")
              return rebuildCall("step_tf_t",
                                 {getProp(Obj, "Numerator"),
                                  getProp(Obj, "Denominator"), T}, PtrTy);
            return rebuildCall("step_ss_t",
                               {getProp(Obj, "A"), getProp(Obj, "B"),
                                getProp(Obj, "C"), getProp(Obj, "D"), T}, PtrTy);
          }
          /* ss-only legacy forms: step(sys) defaults, step(sys, dt, N). */
          if (Cn0 == "ss") {
            mlir::Value Dt, Nval;
            if (C.Args.size() >= 3 && C.Args[1] && C.Args[2]) {
              Dt = lowerExpr(*C.Args[1]);
              Nval = lowerExpr(*C.Args[2]);
            } else {
              Dt = mlir::arith::ConstantOp::create(
                  B, L, F64, mlir::FloatAttr::get(F64, 0.01)).getResult();
              Nval = mlir::arith::ConstantOp::create(
                  B, L, F64, mlir::FloatAttr::get(F64, 500.0)).getResult();
            }
            return rebuildCall("step_ss",
                               {getProp(Obj, "A"), getProp(Obj, "B"),
                                getProp(Obj, "C"), getProp(Obj, "D"),
                                Dt, Nval},
                               PtrTy);
          }
        }

        /* lsim(sys, u, dt): for ss → lsim_ss(A, B, C, D, u, dt). */
        if (Nm == "lsim" && Cls0 && Cn0 == "ss" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value U  = lowerExpr(*C.Args[1]);
          mlir::Value Dt = lowerExpr(*C.Args[2]);
          if (U.getType() != PtrTy) U.setType(PtrTy);
          return rebuildCall("lsim_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D"),
                              U, Dt},
                             PtrTy);
        }

        /* bode(sys, w): ss → bode_ss(A, B, C, D, w); tf →
         * bode_tf(num, den, w). Returns the magnitude vector
         * (1-return form; 2-return [mag, phase] goes through the
         * dedicated splitter in LowerTensorOps). */
        if (Nm == "bode" && Cls0 && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value W = lowerExpr(*C.Args[1]);
          if (W.getType() != PtrTy) W.setType(PtrTy);
          if (Cn0 == "ss") {
            return rebuildCall("bode_ss",
                               {getProp(Obj, "A"), getProp(Obj, "B"),
                                getProp(Obj, "C"), getProp(Obj, "D"), W},
                               PtrTy);
          }
          if (Cn0 == "tf") {
            return rebuildCall("bode_tf",
                               {getProp(Obj, "Numerator"),
                                getProp(Obj, "Denominator"), W},
                               PtrTy);
          }
        }

        /* impulse(sys [, dt, N]): for ss → impulse_ss with defaults
         * dt=0.01, N=500 if not provided. Returns y as a column
         * matrix. Same arg-shape as step(sys). */
        if (Nm == "impulse" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Dt, Nval;
          if (C.Args.size() >= 3 && C.Args[1] && C.Args[2]) {
            Dt = lowerExpr(*C.Args[1]);
            Nval = lowerExpr(*C.Args[2]);
          } else {
            Dt = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, 0.01)).getResult();
            Nval = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, 500.0)).getResult();
          }
          return rebuildCall("impulse_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D"),
                              Dt, Nval},
                             PtrTy);
        }

        /* initial(sys, x0 [, dt, N]): for ss → initial_ss(A, B, C,
         * D, x0, dt, N). x0 is the initial state column vector. */
        if (Nm == "initial" && Cls0 && Cn0 == "ss" && C.Args.size() >= 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value X0 = lowerExpr(*C.Args[1]);
          if (X0.getType() != PtrTy) X0.setType(PtrTy);
          mlir::Value Dt, Nval;
          if (C.Args.size() >= 4 && C.Args[2] && C.Args[3]) {
            Dt = lowerExpr(*C.Args[2]);
            Nval = lowerExpr(*C.Args[3]);
          } else {
            Dt = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, 0.01)).getResult();
            Nval = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, 500.0)).getResult();
          }
          return rebuildCall("initial_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D"),
                              X0, Dt, Nval},
                             PtrTy);
        }

        /* freqresp(sys, w): ss → freqresp_ss; tf → freqresp_tf.
         * Returns matlab_mat_c (complex column). */
        if (Nm == "freqresp" && Cls0 && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value W = lowerExpr(*C.Args[1]);
          if (W.getType() != PtrTy) W.setType(PtrTy);
          if (Cn0 == "ss") {
            return rebuildCall("freqresp_ss",
                               {getProp(Obj, "A"), getProp(Obj, "B"),
                                getProp(Obj, "C"), getProp(Obj, "D"), W},
                               PtrTy);
          }
          if (Cn0 == "tf") {
            return rebuildCall("freqresp_tf",
                               {getProp(Obj, "Numerator"),
                                getProp(Obj, "Denominator"), W},
                               PtrTy);
          }
        }

        /* nyquist(sys, w): ss → nyquist_ss; tf → nyquist_tf. Returns
         * a real N×2 matrix with columns [re, im]. */
        if (Nm == "nyquist" && Cls0 && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value W = lowerExpr(*C.Args[1]);
          if (W.getType() != PtrTy) W.setType(PtrTy);
          if (Cn0 == "ss") {
            return rebuildCall("nyquist_ss",
                               {getProp(Obj, "A"), getProp(Obj, "B"),
                                getProp(Obj, "C"), getProp(Obj, "D"), W},
                               PtrTy);
          }
          if (Cn0 == "tf") {
            return rebuildCall("nyquist_tf",
                               {getProp(Obj, "Numerator"),
                                getProp(Obj, "Denominator"), W},
                               PtrTy);
          }
        }

        /* allmargin(sys, w): ss → 1×4 row [Gm, Pm, Wcg, Wcp]. */
        if (Nm == "allmargin" && Cls0 && Cn0 == "ss" &&
            C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value W = lowerExpr(*C.Args[1]);
          if (W.getType() != PtrTy) W.setType(PtrTy);
          return rebuildCall("allmargin_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D"), W},
                             PtrTy);
        }

        /* damp(sys): ss → damp(sys.A). Returns the 2-column
         * [wn, zeta] matrix. tf form would need companion-matrix or
         * roots-then-damp wiring; deferred. */
        if (Nm == "damp" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("damp", {getProp(Obj, "A")}, PtrTy);
        }

        /* isstable(sys): ss → isstable(sys.A). Returns f64 (boolean). */
        if (Nm == "isstable" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("isstable", {getProp(Obj, "A")}, F64);
        }

        /* §4.4 — controllability / observability matrices on a
         * model object. ctrb(sys) → ctrb(sys.A, sys.B);
         * obsv(sys) → obsv(sys.A, sys.C). Returns a matlab_mat. */
        if (Nm == "ctrb" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("ctrb",
                             {getProp(Obj, "A"), getProp(Obj, "B")},
                             PtrTy);
        }
        if (Nm == "obsv" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("obsv",
                             {getProp(Obj, "A"), getProp(Obj, "C")},
                             PtrTy);
        }

        /* gram(sys, 'c') / gram(sys, 'o') — controllability /
         * observability gramian. Selects between gram_c and gram_o
         * based on the second arg's char literal. */
        if (Nm == "gram" && Cls0 && Cn0 == "ss" &&
            C.Args.size() == 2 && C.Args[1]) {
          const CharLiteral *CL =
              dynamic_cast<const CharLiteral *>(C.Args[1]);
          const StringLiteral *SL =
              CL ? nullptr
                 : dynamic_cast<const StringLiteral *>(C.Args[1]);
          llvm::StringRef Tok = CL ? CL->Value : (SL ? SL->Value : "");
          if (Tok == "c") {
            mlir::Value Obj = loadObj(C.Args[0]);
            return rebuildCall("gram_c",
                               {getProp(Obj, "A"), getProp(Obj, "B")},
                               PtrTy);
          }
          if (Tok == "o") {
            mlir::Value Obj = loadObj(C.Args[0]);
            return rebuildCall("gram_o",
                               {getProp(Obj, "A"), getProp(Obj, "C")},
                               PtrTy);
          }
        }

        /* norm(sys) / norm(sys, 2) — H₂ system norm via the
         * Lyapunov-derived formula sqrt(trace(C·Wc·C')). Returns
         * a scalar f64. norm(sys, Inf) (H∞) is a follow-on. */
        if (Nm == "norm" && Cls0 && Cn0 == "ss" &&
            (C.Args.size() == 1 || C.Args.size() == 2)) {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("norm_h2",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C")},
                             F64);
        }

        /* lqry(sys, Q, R) — output-weighted LQR. Routes to
         * matlab_lqry_ss which does the C'QC / R+D'QD / C'QD algebra
         * in C++ (cleaner than chaining matmul ops at the lowering
         * site and gets scalar f64 args auto-boxed via the dispatch
         * table's `lqry_ss` AutoBoxNames entry). */
        if (Nm == "lqry" && Cls0 && Cn0 == "ss" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Q = lowerExpr(*C.Args[1]);
          mlir::Value R = lowerExpr(*C.Args[2]);
          return rebuildCall("lqry_ss",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C"), getProp(Obj, "D"),
                              Q, R},
                             PtrTy);
        }

        /* §5.1 — model-reduction short forms. hsvd(sys) returns the
         * Hankel singular values vector; balreal_T(sys) returns the
         * balancing similarity transform T. Both route to existing
         * matrix-arg runtime entries by unpacking sys.A/B/C. The
         * full 4-return [Ar, Br, Cr, hsv] = balreal/balred forms
         * remain matrix-arg-only (need multi-return splitter on
         * model-object call sites; deferred). */
        if (Nm == "hsvd" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("hsvd",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C")},
                             PtrTy);
        }
        if (Nm == "balreal_T" && Cls0 && Cn0 == "ss") {
          mlir::Value Obj = loadObj(C.Args[0]);
          return rebuildCall("balreal_T",
                             {getProp(Obj, "A"), getProp(Obj, "B"),
                              getProp(Obj, "C")},
                             PtrTy);
        }

        /* §3.2 c2d(sys, Ts [, method]) — discretise an ss model.
         * Result is a fresh ss instance with (Ad, Bd, sys.C, sys.D)
         * where the (Ad, Bd) pair is picked per method:
         *   'zoh' (default) — matlab_c2d_Ad / matlab_c2d_Bd (expm-based)
         *   'tustin'        — matlab_c2d_tustin_Ad / _Bd (bilinear)
         * The 3-arg form with method='zoh' matches the 2-arg form so
         * `c2d(sys, Ts)` and `c2d(sys, Ts, 'zoh')` produce identical IR.
         * Result slot is class-pinned by Resolver.cpp's pinnedOfRhs. */
        {
          const CharLiteral *C2dMethod =
              (Nm == "c2d" && C.Args.size() == 3)
                  ? dynamic_cast<const CharLiteral *>(C.Args[2])
                  : nullptr;
          bool C2dArityOk = C.Args.size() == 2 ||
              (C.Args.size() == 3 && C2dMethod &&
               (C2dMethod->Value == "zoh" || C2dMethod->Value == "tustin"));
          if (Nm == "c2d" && Cls0 && Cn0 == "ss" && C2dArityOk) {
          bool IsTustin = C2dMethod && C2dMethod->Value == "tustin";
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Ts  = lowerExpr(*C.Args[1]);
          mlir::Value AVal = getProp(Obj, "A");
          mlir::Value BVal = getProp(Obj, "B");
          mlir::NamedAttribute CalAd(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx,
                  IsTustin ? "matlab_c2d_tustin_Ad" : "matlab_c2d_Ad"));
          mlir::NamedAttribute CalBd(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx,
                  IsTustin ? "matlab_c2d_tustin_Bd" : "matlab_c2d_Bd"));
          mlir::Value Ad = emitUnreg("matlab.call_builtin",
                                       {AVal, BVal, Ts}, PtrTy, L, {CalAd});
          mlir::Value Bd = emitUnreg("matlab.call_builtin",
                                       {AVal, BVal, Ts}, PtrTy, L, {CalBd});
          mlir::Value CVal = getProp(Obj, "C");
          mlir::Value DVal = getProp(Obj, "D");
          // The 5-arg constructor stamps Ts onto the result so the
          // returned model is correctly tagged as discrete-time.
          // Gates MPC bring-up (`mpc(plant, Ts, …)` reads sys.Ts to
          // decide whether to discretize internally).
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ss__ss"));
          return emitUnreg("matlab.call",
                           {Ad, Bd, CVal, DVal, Ts},
                           PtrTy, L, {CtorCal});
          }
        }

        /* §3.2 c2d(tf, Ts [, method]) — discretise a transfer-function
         * model. The tf carries (Numerator, Denominator); route through the
         * runtime tf-level discretiser (tf2ss → c2d → ss2tf for 'zoh', exact
         * bilinear substitution for 'tustin') and rebuild a discrete
         * tf(num_d, den_d, Ts). Mirrors the ss branch above; the result slot
         * is tf-pinned by Resolver.cpp's CST short-form block. (#27) */
        {
          const CharLiteral *TfC2dMethod =
              (Nm == "c2d" && C.Args.size() == 3)
                  ? dynamic_cast<const CharLiteral *>(C.Args[2])
                  : nullptr;
          bool TfC2dArityOk = C.Args.size() == 2 ||
              (C.Args.size() == 3 && TfC2dMethod &&
               (TfC2dMethod->Value == "zoh" || TfC2dMethod->Value == "tustin"));
          if (Nm == "c2d" && Cls0 && Cn0 == "tf" && TfC2dArityOk) {
            bool IsTustin = TfC2dMethod && TfC2dMethod->Value == "tustin";
            mlir::Value Obj  = loadObj(C.Args[0]);
            mlir::Value Ts   = lowerExpr(*C.Args[1]);
            mlir::Value NumV = getProp(Obj, "Numerator");
            mlir::Value DenV = getProp(Obj, "Denominator");
            mlir::NamedAttribute CalNum(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx,
                    IsTustin ? "matlab_c2d_tf_tustin_num" : "matlab_c2d_tf_num"));
            mlir::NamedAttribute CalDen(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx,
                    IsTustin ? "matlab_c2d_tf_tustin_den" : "matlab_c2d_tf_den"));
            mlir::Value NumD = emitUnreg("matlab.call_builtin",
                                         {NumV, DenV, Ts}, PtrTy, L, {CalNum});
            mlir::Value DenD = emitUnreg("matlab.call_builtin",
                                         {NumV, DenV, Ts}, PtrTy, L, {CalDen});
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "tf__tf"));
            // 3-arg constructor stamps Ts so the result is tagged discrete.
            return emitUnreg("matlab.call",
                             {NumD, DenD, Ts}, PtrTy, L, {CtorCal});
          }
        }

        /* §3.2 d2c(tf [, method]) — continuous-ise a discrete tf. The sample
         * time comes from the model's own Ts property (read boxed, unboxed
         * runtime-side). Routes through matlab_d2c_tf_num/den; the result is
         * a continuous tf (Ts = 0 via the 2-arg constructor). (#27) */
        {
          const CharLiteral *TfD2cMethod =
              (Nm == "d2c" && C.Args.size() == 2)
                  ? dynamic_cast<const CharLiteral *>(C.Args[1])
                  : nullptr;
          bool TfD2cArityOk = C.Args.size() == 1 ||
              (C.Args.size() == 2 && TfD2cMethod &&
               TfD2cMethod->Value == "zoh");
          if (Nm == "d2c" && Cls0 && Cn0 == "tf" && TfD2cArityOk) {
            mlir::Value Obj  = loadObj(C.Args[0]);
            mlir::Value NumV = getProp(Obj, "Numerator");
            mlir::Value DenV = getProp(Obj, "Denominator");
            mlir::Value TsV  = getProp(Obj, "Ts");
            mlir::NamedAttribute CalNum(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_d2c_tf_num"));
            mlir::NamedAttribute CalDen(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_d2c_tf_den"));
            mlir::Value NumC = emitUnreg("matlab.call_builtin",
                                         {NumV, DenV, TsV}, PtrTy, L, {CalNum});
            mlir::Value DenC = emitUnreg("matlab.call_builtin",
                                         {NumV, DenV, TsV}, PtrTy, L, {CalDen});
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "tf__tf"));
            return emitUnreg("matlab.call",
                             {NumC, DenC}, PtrTy, L, {CtorCal});
          }
        }

        /* MPC Tier-1 — `mpcmove(obj, st, ym, r)` and `sim(obj, T, r)`
         * class-pinned-first-arg routes.  Bypass the classdef-method
         * function call (which suffers from `none`-typed formal-
         * parameter slots inside the body); emit the runtime entry
         * directly with the user-supplied operand types in scope so
         * the pde_table dispatch can resolve them. */
        if (Nm == "mpcmove" && Cls0 && Cn0 == "mpc" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = loadObj(C.Args[1]);
          mlir::Value Ym  = lowerExpr(*C.Args[2]);
          mlir::Value R   = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_move"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, St, Ym, R}, PtrTy, L, {Cal});
        }
        /* MPC Tier-2 §3.7 — 5-arg form `mpcmove(obj, st, ym, r, opt)`
         * routes to matlab_mpc_move_opt for run-time bound overrides. */
        if (Nm == "mpcmove" && Cls0 && Cn0 == "mpc" && C.Args.size() == 5) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = loadObj(C.Args[1]);
          mlir::Value Ym  = lowerExpr(*C.Args[2]);
          mlir::Value R   = lowerExpr(*C.Args[3]);
          mlir::Value Opt = loadObj(C.Args[4]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_move_opt"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, St, Ym, R, Opt}, PtrTy, L, {Cal});
        }
        /* MPC Tier-3 §4.1 — adaptive MPC.  7-arg form
         * `mpcmoveAdaptive(obj, st, A, B, C, ym, r)` rebuilds the
         * cached prediction matrices from per-tick (A, B, C) before
         * solving the QP.  Routes to matlab_mpc_move_adaptive. */
        if (Nm == "mpcmoveAdaptive" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 7) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = loadObj(C.Args[1]);
          mlir::Value Aa  = lowerExpr(*C.Args[2]);
          mlir::Value Bb  = lowerExpr(*C.Args[3]);
          mlir::Value Cc  = lowerExpr(*C.Args[4]);
          mlir::Value Ym  = lowerExpr(*C.Args[5]);
          mlir::Value R   = lowerExpr(*C.Args[6]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_move_adaptive"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, St, Aa, Bb, Cc, Ym, R}, PtrTy, L, {Cal});
        }
        /* MPC Tier-4 §5.1 — `generateExplicitMPC(mpc, x_lo, x_hi,
         * n_grid, r)`.  Bypass the MATLAB wrapper (whose typed
         * formal-params leak into the inner runtime call) and emit
         * the runtime call directly.  Need to default-construct an
         * `explicitMPC` instance first to write fields onto.  We
         * synthesise that via the existing constructor call site. */
        if (Nm == "generateExplicitMPC" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 5) {
          mlir::Value Mpc   = loadObj(C.Args[0]);
          mlir::Value X_lo  = lowerExpr(*C.Args[1]);
          mlir::Value X_hi  = lowerExpr(*C.Args[2]);
          mlir::Value Ng    = lowerExpr(*C.Args[3]);
          mlir::Value R     = lowerExpr(*C.Args[4]);
          /* Allocate the explicitMPC obj. */
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "explicitMPC__explicitMPC"));
          mlir::Value Eobj = emitUnreg("matlab.call",
                                         {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute GenCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_generate_explicit"));
          emitUnregOp("matlab.call_builtin",
                      {Eobj, Mpc, X_lo, X_hi, Ng, R},
                      {mlir::NoneType::get(&MCtx)}, L, {GenCal});
          return Eobj;
        }
        /* MPC Tier-4 §5.2 — `mpcmoveExplicit(eobj, xc)`.  Bypass
         * the wrapper for type-flow reasons. */
        if (Nm == "mpcmoveExplicit" && Cls0 && Cn0 == "explicitMPC" &&
            C.Args.size() == 2) {
          mlir::Value Eobj = loadObj(C.Args[0]);
          mlir::Value Xc   = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_move_explicit"));
          return emitUnreg("matlab.call_builtin",
                           {Eobj, Xc}, PtrTy, L, {Cal});
        }
        /* MPC Tier-4 §5.7 — `mpcmoveFinite(obj, st, ym, r)`.  Routes
         * to matlab_mpc_move_finite which enumerates over the single
         * binary MV's two values and keeps the lower-cost branch. */
        if (Nm == "mpcmoveFinite" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = loadObj(C.Args[1]);
          mlir::Value Ym  = lowerExpr(*C.Args[2]);
          mlir::Value Rr  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_move_finite"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, St, Ym, Rr}, PtrTy, L, {Cal});
        }
        /* MPC Tier-6 §7.4 — `setEstimator(obj, L)` writes obj.L =
         * L; `getEstimator(obj)` returns obj.L.  Sugar over the
         * direct property access pattern. */
        if (Nm == "setEstimator" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Lv  = lowerExpr(*C.Args[1]);
          mlir::Value NameV = emitFieldNameChar("L", L);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_obj_set_mat"));
          emitUnregOp("matlab.call_builtin", {Obj, NameV, Lv},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Obj;
        }
        if (Nm == "getEstimator" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          return getProp(Obj, "L");
        }
        /* MPC Tier-6 §7.5 — `review(obj)` sanity diagnostic. */
        if (Nm == "review" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_review"));
          return emitUnreg("matlab.call_builtin",
                           {Obj}, PtrTy, L, {Cal});
        }

        /* MPC Tier-5 — `nlmpcmove(nlobj, x, lastu, r, @stateFn)`.
         * 5th arg is the StateFcn handle.  Routes to matlab_nlmpc_move
         * which sets up thread-local context and calls fmincon over
         * the u-trajectory decision variable. */
        if (Nm == "nlmpcmove" && Cls0 && Cn0 == "nlmpc" &&
            C.Args.size() == 5) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Xi  = lowerExpr(*C.Args[1]);
          mlir::Value Up  = lowerExpr(*C.Args[2]);
          mlir::Value Rr  = lowerExpr(*C.Args[3]);
          mlir::Value Fn  = lowerExpr(*C.Args[4]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nlmpc_move"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, Xi, Up, Rr, Fn}, PtrTy, L, {Cal});
        }
        /* MPC Tier-3 §4.2 — time-varying MPC.  7-arg form
         * `mpcmoveTV(obj, st, A_stack, B_stack, C_stack, ym, r)`
         * builds time-varying Sx / Su / Su1 from stacked plants
         * (block i of each stack is A_i / B_i / C_i for prediction
         * step i transitioning to step i+1). */
        if (Nm == "mpcmoveTV" && Cls0 && Cn0 == "mpc" &&
            C.Args.size() == 7) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = loadObj(C.Args[1]);
          mlir::Value Aa  = lowerExpr(*C.Args[2]);
          mlir::Value Bb  = lowerExpr(*C.Args[3]);
          mlir::Value Cc  = lowerExpr(*C.Args[4]);
          mlir::Value Ym  = lowerExpr(*C.Args[5]);
          mlir::Value R   = lowerExpr(*C.Args[6]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_move_tv"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, St, Aa, Bb, Cc, Ym, R}, PtrTy, L, {Cal});
        }
        if (Nm == "sim" && Cls0 && Cn0 == "mpc" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value T   = lowerExpr(*C.Args[1]);
          mlir::Value R   = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_sim"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, T, R}, PtrTy, L, {Cal});
        }
        /* MPC Tier-6 §7.6 — 4-arg `sim(obj, T, r, opt)` routes to
         * matlab_mpc_sim_opt for sim-time overrides (PlantInitialState). */
        if (Nm == "sim" && Cls0 && Cn0 == "mpc" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value T   = lowerExpr(*C.Args[1]);
          mlir::Value R   = lowerExpr(*C.Args[2]);
          mlir::Value Opt = loadObj(C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_mpc_sim_opt"));
          return emitUnreg("matlab.call_builtin",
                           {Obj, T, R, Opt}, PtrTy, L, {Cal});
        }

        /* ============================================================
         * Econometrics Toolbox — class-pinned-first-arg dispatch.  The
         * generic method names estimate/forecast/infer/simulate/filter
         * are routed by the receiver's class name (arima/garch/...) to
         * the matlab_econ_* kernels, so they coexist with the System-
         * Identification idpoly routes below.
         * ============================================================ */
        {
          /* Is the receiver an Econometrics model class?  Pick the runtime
           * family by class name: arima -> matlab_econ_arima_*; the
           * conditional-variance trio (garch/egarch/gjr) shares
           * matlab_econ_garch_* (which dispatches internally on ModelKind);
           * the model constructor name is `<class>__<class>`. */
          const char *fam = nullptr;        /* runtime prefix */
          const char *ctor = nullptr;       /* fresh-object constructor */
          if (Cls0 && Cn0 == "arima") { fam = "arima"; ctor = "arima__arima"; }
          else if (Cls0 && (Cn0 == "garch" || Cn0 == "egarch" || Cn0 == "gjr")) {
            fam = "garch";
            ctor = (Cn0 == "garch") ? "garch__garch"
                 : (Cn0 == "egarch") ? "egarch__egarch" : "gjr__gjr";
          }
          else if (Cls0 && Cn0 == "varm") { fam = "varm"; ctor = "varm__varm"; }
          else if (Cls0 && (Cn0 == "ssm" || Cn0 == "dssm")) {
            fam = "ssm";
            ctor = (Cn0 == "ssm") ? "ssm__ssm" : "dssm__dssm";
          }
          /* irf(Mdl, numObs) — VAR impulse responses (varm only). */
          if (Cls0 && Cn0 == "varm" && Nm == "irf" && C.Args.size() == 2) {
            mlir::Value Mdl = loadObj(C.Args[0]);
            mlir::Value No  = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_econ_varm_irf"));
            return emitUnreg("matlab.call_builtin", {Mdl, No}, PtrTy, L, {Cal});
          }
          /* filter(Mdl, Y) / smooth(Mdl, Y) — state-space Kalman (ssm/dssm). */
          if (Cls0 && (Cn0 == "ssm" || Cn0 == "dssm") &&
              (Nm == "filter" || Nm == "smooth") && C.Args.size() == 2) {
            mlir::Value Mdl = loadObj(C.Args[0]);
            mlir::Value Y   = lowerExpr(*C.Args[1]);
            std::string rt = std::string("matlab_econ_ssm_") +
                             (Nm == "filter" ? "filter" : "smooth");
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, rt));
            return emitUnreg("matlab.call_builtin", {Mdl, Y}, PtrTy, L, {Cal});
          }
          /* bayeslm: estimate(Mdl, X, y) [3-arg] mutates + returns receiver;
           * forecast(Mdl, XNew) [2-arg] is the posterior-mean prediction. */
          if (Cls0 && Cn0 == "bayeslm" && Nm == "estimate" &&
              C.Args.size() == 3) {
            mlir::Value Mdl = loadObj(C.Args[0]);
            mlir::Value X   = lowerExpr(*C.Args[1]);
            mlir::Value Y   = lowerExpr(*C.Args[2]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_econ_bayeslm_estimate"));
            emitUnregOp("matlab.call_builtin", {Mdl, X, Y},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Mdl;
          }
          if (Cls0 && Cn0 == "bayeslm" && Nm == "forecast" &&
              C.Args.size() == 2) {
            mlir::Value Mdl  = loadObj(C.Args[0]);
            mlir::Value XNew = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_econ_bayeslm_forecast"));
            return emitUnreg("matlab.call_builtin", {Mdl, XNew}, PtrTy, L, {Cal});
          }
          /* dtmc: asymptotics(mc) stationary dist; simulate(mc, n) path. */
          if (Cls0 && Cn0 == "dtmc" && Nm == "asymptotics" &&
              C.Args.size() == 1) {
            mlir::Value Mc = loadObj(C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_econ_dtmc_asymptotics"));
            return emitUnreg("matlab.call_builtin", {Mc}, PtrTy, L, {Cal});
          }
          if (Cls0 && Cn0 == "dtmc" && Nm == "simulate" &&
              C.Args.size() == 2) {
            mlir::Value Mc = loadObj(C.Args[0]);
            mlir::Value Ns = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_econ_dtmc_simulate"));
            return emitUnreg("matlab.call_builtin", {Mc, Ns}, PtrTy, L, {Cal});
          }
          if (fam && Nm == "estimate" && C.Args.size() == 2) {
            mlir::Value Tmpl = loadObj(C.Args[0]);
            mlir::Value Y    = lowerExpr(*C.Args[1]);
            std::string rt = std::string("matlab_econ_") + fam + "_estimate";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, rt));
            /* ssm/dssm: mutate the template in place and return it (the
             * matrix-typed system matrices make the zero-arg fresh-ctor
             * path hit a param-slot typing limit; the receiver already
             * carries the ssm class so the result propagates correctly). */
            if (std::string(fam) == "ssm") {
              emitUnregOp("matlab.call_builtin", {Tmpl, Y},
                          {mlir::NoneType::get(&MCtx)}, L, {Cal});
              return Tmpl;
            }
            /* Other families: allocate a FRESH model via the zero-arg ctor
             * (so the result carries the model class, exactly like armax
             * returns a fresh idpoly), then populate it in place from the
             * template orders + data. */
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, ctor));
            mlir::Value Model =
                emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
            emitUnregOp("matlab.call_builtin", {Model, Tmpl, Y},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Model;
          }
          if (fam && Nm == "forecast" && C.Args.size() == 3) {
            /* yF = forecast(Mdl, numPeriods, Y0) */
            mlir::Value Mdl = loadObj(C.Args[0]);
            mlir::Value H   = lowerExpr(*C.Args[1]);
            mlir::Value Y0  = lowerExpr(*C.Args[2]);
            std::string rt = std::string("matlab_econ_") + fam + "_forecast";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, rt));
            return emitUnreg("matlab.call_builtin", {Mdl, H, Y0}, PtrTy, L,
                             {Cal});
          }
          if (fam && Nm == "infer" && C.Args.size() == 2) {
            /* E = infer(Mdl, Y) — residuals (arima) or conditional
             * variances (garch family). */
            mlir::Value Mdl = loadObj(C.Args[0]);
            mlir::Value Y   = lowerExpr(*C.Args[1]);
            std::string rt = std::string("matlab_econ_") + fam + "_infer";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, rt));
            return emitUnreg("matlab.call_builtin", {Mdl, Y}, PtrTy, L, {Cal});
          }
          if (fam && Nm == "simulate" && C.Args.size() == 2) {
            /* Y = simulate(Mdl, numObs) */
            mlir::Value Mdl = loadObj(C.Args[0]);
            mlir::Value N   = lowerExpr(*C.Args[1]);
            std::string rt = std::string("matlab_econ_") + fam + "_simulate";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, rt));
            return emitUnreg("matlab.call_builtin", {Mdl, N}, PtrTy, L, {Cal});
          }
        }

        /* ============================================================
         * System Identification Toolbox Tier-1 — class-pinned-first-arg
         * dispatch.  arx / ar return a fresh idpoly (allocated via the
         * zero-arg ctor, then populated by the runtime in place — same
         * pattern as MPC's generateExplicitMPC).  sim / predict /
         * compare / fpe / aic key on the model class so they coexist
         * with the identically-named MPC + CST routes above.  (tf/ss
         * conversion of an idpoly is handled in the constructor-call
         * path below, since `tf`/`ss` resolve as classes, not builtins,
         * and never reach this BindingKind::Builtin-gated block.)
         * ============================================================ */
        /* arx(data, [na nb nk]) — QR least-squares ARX. */
        if (Nm == "arx" && Cls0 && Cn0 == "iddata" && C.Args.size() == 2) {
          mlir::Value Data   = loadObj(C.Args[0]);
          mlir::Value Orders = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idpoly__idpoly"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_arx"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Orders},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* arx(data, orders, opt) — Tier-6 regularized ARX. `opt` is an
         * arxOptions; we read opt.Regularization off it and call the
         * ridge variant. */
        if (Nm == "arx" && Cls0 && Cn0 == "iddata" && C.Args.size() == 3) {
          mlir::Value Data   = loadObj(C.Args[0]);
          mlir::Value Orders = lowerExpr(*C.Args[1]);
          mlir::Value Opt    = loadObj(C.Args[2]);
          /* opt.Regularization → scalar λ. */
          mlir::Value NameReg = emitFieldNameChar("Regularization", L);
          mlir::NamedAttribute RegCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_obj_get_f64"));
          mlir::Value Lam = emitUnreg("matlab.call_builtin", {Opt, NameReg},
                                      F64, L, {RegCal});
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idpoly__idpoly"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_arx_reg"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Orders, Lam},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* getcov(model) — parameter covariance from cached Gram. */
        if (Nm == "getcov" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 1) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_getcov"));
          return emitUnreg("matlab.call_builtin", {Model}, PtrTy, L, {Cal});
        }
        /* getpvec(model) — packed parameter vector. */
        if (Nm == "getpvec" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 1) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_getpvec"));
          return emitUnreg("matlab.call_builtin", {Model}, PtrTy, L, {Cal});
        }
        /* setpvec(model, theta) — write parameter vector back. */
        if (Nm == "setpvec" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 2) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::Value Th    = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_setpvec"));
          emitUnregOp("matlab.call_builtin", {Model, Th},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* ar(data, na) — Yule-Walker AR time-series estimation. */
        if (Nm == "ar" && Cls0 && Cn0 == "iddata" && C.Args.size() == 2) {
          mlir::Value Data = loadObj(C.Args[0]);
          mlir::Value Na   = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idpoly__idpoly"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_ar"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Na},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* armax/oe/bj(data, orders) — PEM estimators; same alloc-then-
         * populate shape as arx (return a fresh idpoly). */
        {
          const char *pemRt = nullptr;
          if (Nm == "armax") pemRt = "matlab_ident_armax";
          else if (Nm == "oe") pemRt = "matlab_ident_oe";
          else if (Nm == "bj") pemRt = "matlab_ident_bj";
          else if (Nm == "iv4") pemRt = "matlab_ident_iv4";
          if (pemRt && Cls0 && Cn0 == "iddata" && C.Args.size() == 2) {
            mlir::Value Data   = loadObj(C.Args[0]);
            mlir::Value Orders = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "idpoly__idpoly"));
            mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, pemRt));
            emitUnregOp("matlab.call_builtin", {Model, Data, Orders},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Model;
          }
        }
        /* etfe(data) / spa(data) — non-parametric frequency response;
         * returns a fresh idfrd. */
        {
          const char *frRt = nullptr;
          if (Nm == "etfe") frRt = "matlab_ident_etfe";
          else if (Nm == "spa") frRt = "matlab_ident_spa";
          if (frRt && Cls0 && Cn0 == "iddata" && C.Args.size() == 1) {
            mlir::Value Data = loadObj(C.Args[0]);
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "idfrd__idfrd"));
            mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, frRt));
            emitUnregOp("matlab.call_builtin", {Model, Data},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Model;
          }
        }
        /* impulseest(data, N) — non-parametric impulse response (FIR);
         * returns a fresh idpoly (A = 1, B = Markov params). */
        if (Nm == "impulseest" && Cls0 && Cn0 == "iddata" && C.Args.size() == 2) {
          mlir::Value Data = loadObj(C.Args[0]);
          mlir::Value Nl   = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idpoly__idpoly"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_impulseest"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Nl},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* forecast(model, data, K) — K-step time-series forecast. */
        if (Nm == "forecast" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 3) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::Value Data  = loadObj(C.Args[1]);
          mlir::Value Kk    = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_forecast"));
          return emitUnreg("matlab.call_builtin", {Model, Data, Kk}, PtrTy, L, {Cal});
        }
        /* tfest(data, np, nz) — transfer-function estimation (OE form);
         * returns a fresh idpoly (B = num, F = den). */
        if (Nm == "tfest" && Cls0 && Cn0 == "iddata" && C.Args.size() == 3) {
          mlir::Value Data = loadObj(C.Args[0]);
          mlir::Value Np   = lowerExpr(*C.Args[1]);
          mlir::Value Nz   = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idpoly__idpoly"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_tfest"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Np, Nz},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* n4sid/ssest(data, nx) — subspace state-space estimation;
         * returns a fresh idss (alloc-then-populate, like arx, but the
         * 2nd arg is a scalar model order). */
        {
          const char *ssRt = nullptr;
          if (Nm == "n4sid") ssRt = "matlab_ident_n4sid";
          else if (Nm == "ssest") ssRt = "matlab_ident_ssest";
          if (ssRt && Cls0 && Cn0 == "iddata" && C.Args.size() == 2) {
            mlir::Value Data = loadObj(C.Args[0]);
            mlir::Value Nx   = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "idss__idss"));
            mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, ssRt));
            emitUnregOp("matlab.call_builtin", {Model, Data, Nx},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
            return Model;
          }
        }
        /* nlgreyest(data, par0, @statefn, nx) — nonlinear grey-box; the
         * 3rd arg is the ODE-rhs handle.  Returns idnlgrey. */
        if (Nm == "nlgreyest" && Cls0 && Cn0 == "iddata" && C.Args.size() == 4) {
          mlir::Value Data = loadObj(C.Args[0]);
          mlir::Value Par0 = lowerExpr(*C.Args[1]);
          mlir::Value Fn   = lowerExpr(*C.Args[2]);
          mlir::Value Nx   = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idnlgrey__idnlgrey"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_nlgreyest"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Par0, Fn, Nx},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* greyest(data, par0, @structfn, nx) — linear grey-box; the
         * 3rd arg is the structure-function handle.  Returns idgrey. */
        if (Nm == "greyest" && Cls0 && Cn0 == "iddata" && C.Args.size() == 4) {
          mlir::Value Data = loadObj(C.Args[0]);
          mlir::Value Par0 = lowerExpr(*C.Args[1]);
          mlir::Value Fn   = lowerExpr(*C.Args[2]);
          mlir::Value Nx   = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "idgrey__idgrey"));
          mlir::Value Model = emitUnreg("matlab.call", {}, PtrTy, L, {CtorCal});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_greyest"));
          emitUnregOp("matlab.call_builtin", {Model, Data, Par0, Fn, Nx},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Model;
        }
        /* sim(model, u) for an idss / idgrey — state-space simulation. */
        if (Nm == "sim" && Cls0 && (Cn0 == "idss" || Cn0 == "idgrey") &&
            C.Args.size() == 2) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::Value U     = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_sim_ss"));
          return emitUnreg("matlab.call_builtin", {Model, U}, PtrTy, L, {Cal});
        }
        /* pe(model, data) / resid(model, data) — return matrices
         * (prediction errors / whiteness diagnostic). */
        if ((Nm == "pe" || Nm == "resid") && Cls0 && Cn0 == "idpoly" &&
            C.Args.size() == 2) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::Value Data  = loadObj(C.Args[1]);
          const char *peRt = (Nm == "pe") ? "matlab_ident_pe"
                                           : "matlab_ident_resid";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, peRt));
          return emitUnreg("matlab.call_builtin", {Model, Data}, PtrTy, L, {Cal});
        }
        /* sim(model, u) — deterministic simulation B(q)/A(q)·u. */
        if (Nm == "sim" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 2) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::Value U     = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_sim"));
          return emitUnreg("matlab.call_builtin", {Model, U}, PtrTy, L, {Cal});
        }
        /* EKF/UKF predict(obj, @StateFcn) — one filter prediction step. */
        if (Nm == "predict" && Cls0 &&
            (Cn0 == "extendedKalmanFilter" || Cn0 == "unscentedKalmanFilter") &&
            C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Fn  = lowerExpr(*C.Args[1]);
          const char *rt = (Cn0 == "extendedKalmanFilter")
                               ? "matlab_ident_ekf_predict"
                               : "matlab_ident_ukf_predict";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Obj, Fn}, PtrTy, L, {Cal});
        }
        /* EKF/UKF correct(obj, @MeasFcn, y) — one filter correction step. */
        if (Nm == "correct" && Cls0 &&
            (Cn0 == "extendedKalmanFilter" || Cn0 == "unscentedKalmanFilter") &&
            C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Fn  = lowerExpr(*C.Args[1]);
          mlir::Value Yv  = lowerExpr(*C.Args[2]);
          const char *rt = (Cn0 == "extendedKalmanFilter")
                               ? "matlab_ident_ekf_correct"
                               : "matlab_ident_ukf_correct";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Obj, Fn, Yv}, PtrTy, L, {Cal});
        }
        /* ===== Sensor Fusion Tier-2 — tracking-filter methods =================
         * predict(trackingKF) / correct(trackingKF, y) — no handles, linear KF. */
        if (Nm == "predict" && Cls0 && Cn0 == "trackingKF" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_trackingkf_predict"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        if (Nm == "correct" && Cls0 && Cn0 == "trackingKF" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Yv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_trackingkf_correct"));
          return emitUnreg("matlab.call_builtin", {Obj, Yv}, PtrTy, L, {Cal});
        }
        /* predict(trackingEKF/UKF, @f) / correct(trackingEKF/UKF, @h, y_vec). */
        if (Nm == "predict" && Cls0 &&
            (Cn0 == "trackingEKF" || Cn0 == "trackingUKF") &&
            C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Fn  = lowerExpr(*C.Args[1]);
          const char *rt = (Cn0 == "trackingEKF")
                               ? "matlab_fusion_trackingekf_predict"
                               : "matlab_fusion_trackingukf_predict";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Obj, Fn}, PtrTy, L, {Cal});
        }
        if (Nm == "correct" && Cls0 &&
            (Cn0 == "trackingEKF" || Cn0 == "trackingUKF") &&
            C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Fn  = lowerExpr(*C.Args[1]);
          mlir::Value Yv  = lowerExpr(*C.Args[2]);
          const char *rt = (Cn0 == "trackingEKF")
                               ? "matlab_fusion_trackingekf_correct"
                               : "matlab_fusion_trackingukf_correct";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Obj, Fn, Yv}, PtrTy, L, {Cal});
        }
        /* ===== Sensor Fusion Tier-3 — sensor / orientation-filter step ========
         * step(imuSensor, acc_true, gyro_true) / step(imuSensor, acc, gyro, mag). */
        if (Nm == "step" && Cls0 && Cn0 == "imuSensor" &&
            (C.Args.size() == 3 || C.Args.size() == 4)) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value A   = lowerExpr(*C.Args[1]);
          mlir::Value G   = lowerExpr(*C.Args[2]);
          mlir::Value M   = (C.Args.size() == 4) ? lowerExpr(*C.Args[3]) : A;
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_imu_step"));
          return emitUnreg("matlab.call_builtin", {Obj, A, G, M}, PtrTy, L, {Cal});
        }
        if (Nm == "step" && Cls0 && Cn0 == "gpsSensor" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value P   = lowerExpr(*C.Args[1]);
          mlir::Value V   = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_gps_step"));
          return emitUnreg("matlab.call_builtin", {Obj, P, V}, PtrTy, L, {Cal});
        }
        /* step(ahrsfilter/imufilter/complementaryFilter, accel, gyro [,mag]) */
        if (Nm == "step" && Cls0 &&
            (Cn0 == "ahrsfilter" || Cn0 == "imufilter" ||
             Cn0 == "complementaryFilter") &&
            (C.Args.size() == 3 || C.Args.size() == 4)) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value A   = lowerExpr(*C.Args[1]);
          mlir::Value G   = lowerExpr(*C.Args[2]);
          mlir::Value M;
          if (C.Args.size() == 4) M = lowerExpr(*C.Args[3]);
          const char *rt;
          if      (Cn0 == "ahrsfilter")          rt = "matlab_fusion_ahrs_step";
          else if (Cn0 == "imufilter")           rt = "matlab_fusion_imufilter_step";
          else                                   rt = "matlab_fusion_compfilter_step";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          if (C.Args.size() == 4)
            return emitUnreg("matlab.call_builtin", {Obj, A, G, M}, PtrTy, L, {Cal});
          return emitUnreg("matlab.call_builtin", {Obj, A, G}, PtrTy, L, {Cal});
        }
        /* predict(insfilterMARG, acc, gyro, dt) / fuseaccel(...) / fusegps(...). */
        if (Nm == "predict" && Cls0 && Cn0 == "insfilterMARG" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value A   = lowerExpr(*C.Args[1]);
          mlir::Value G   = lowerExpr(*C.Args[2]);
          mlir::Value Dt  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_insmarg_predict"));
          return emitUnreg("matlab.call_builtin", {Obj, A, G, Dt}, PtrTy, L, {Cal});
        }
        if (Nm == "fuseaccel" && Cls0 && Cn0 == "insfilterMARG" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value A   = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_insmarg_fuse_accel"));
          return emitUnreg("matlab.call_builtin", {Obj, A}, PtrTy, L, {Cal});
        }
        if (Nm == "fusegps" && Cls0 && Cn0 == "insfilterMARG" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value P   = lowerExpr(*C.Args[1]);
          mlir::Value V   = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_insmarg_fuse_gps"));
          return emitUnreg("matlab.call_builtin", {Obj, P, V}, PtrTy, L, {Cal});
        }
        /* ===== Sensor Fusion Tier-4 / Tier-5 — trajectory + tracker methods ==
         * lookupPose(waypointTrajectory, t) → 1×3 position. */
        if (Nm == "lookupPose" && Cls0 && Cn0 == "waypointTrajectory" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Tv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_waypoint_lookup"));
          return emitUnreg("matlab.call_builtin", {Obj, Tv}, PtrTy, L, {Cal});
        }
        /* step(trackerGNN, detections_Nx2, dt). */
        if (Nm == "step" && Cls0 && Cn0 == "trackerGNN" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Dt  = lowerExpr(*C.Args[1]);
          mlir::Value DtS = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_gnn_step"));
          return emitUnreg("matlab.call_builtin", {Obj, Dt, DtS}, PtrTy, L, {Cal});
        }
        /* numConfirmed(trackerGNN) — confirmed-track count. */
        if (Nm == "numConfirmed" && Cls0 && Cn0 == "trackerGNN" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_fusion_gnn_numconfirmed"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        /* ===== Robotics Tier-2 — rigidBodyTree methods ====================== */
        if (Nm == "addBody" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 5) {
          // addBody(tree, dh_row, jt_code, lo, hi)
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value DH  = lowerExpr(*C.Args[1]);
          mlir::Value JT  = lowerExpr(*C.Args[2]);
          mlir::Value Lo  = lowerExpr(*C.Args[3]);
          mlir::Value Hi  = lowerExpr(*C.Args[4]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_tree_addbody"));
          return emitUnreg("matlab.call_builtin", {Obj, DH, JT, Lo, Hi}, PtrTy, L, {Cal});
        }
        if (Nm == "loadrobot" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          // loadrobot(tree, name_string)
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Nm2 = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_loadrobot"));
          return emitUnreg("matlab.call_builtin", {Obj, Nm2}, PtrTy, L, {Cal});
        }
        if (Nm == "importrobot" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          // importrobot(tree, filename) — populate an existing tree in place
          // (keeps the LHS class-pinned for downstream method dispatch).
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Fn  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_importrobot"));
          return emitUnreg("matlab.call_builtin", {Obj, Fn}, PtrTy, L, {Cal});
        }
        if (Nm == "getTransform" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_getTransform"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv}, PtrTy, L, {Cal});
        }
        if (Nm == "geometricJacobian" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_geometricJacobian"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv}, PtrTy, L, {Cal});
        }
        if (Nm == "homeConfiguration" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_homeConfiguration"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        if (Nm == "randomConfiguration" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_randomConfiguration"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        if (Nm == "massMatrix" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_massMatrix"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv}, PtrTy, L, {Cal});
        }
        if (Nm == "inverseDynamics" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::Value Qd  = lowerExpr(*C.Args[2]);
          mlir::Value Qdd = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_inverseDynamics"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv, Qd, Qdd}, PtrTy, L, {Cal});
        }
        if (Nm == "forwardDynamics" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::Value Qd  = lowerExpr(*C.Args[2]);
          mlir::Value Tau = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_forwardDynamics"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv, Qd, Tau}, PtrTy, L, {Cal});
        }
        if (Nm == "gravityTorque" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_gravityTorque"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv}, PtrTy, L, {Cal});
        }
        if (Nm == "velocityProduct" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::Value Qd  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_velocityProduct"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv, Qd}, PtrTy, L, {Cal});
        }
        if (Nm == "centerOfMass" && Cls0 && Cn0 == "rigidBodyTree" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_centerOfMass"));
          return emitUnreg("matlab.call_builtin", {Obj, Qv}, PtrTy, L, {Cal});
        }
        /* ===== Robotics Tier-3 — inverseKinematics solve ===================== */
        // ik(ik_obj, target_tform, q0, w_pos, w_ori) — call-syntax sugar on
        // the inverseKinematics instance.  The MATLAB form is
        // `[q,info] = ik(target, weights, q0)` — we expose the 5-arg flat form.
        if (Nm == "solveik" && Cls0 && Cn0 == "inverseKinematics" && C.Args.size() == 5) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Tg  = lowerExpr(*C.Args[1]);
          mlir::Value Q0  = lowerExpr(*C.Args[2]);
          mlir::Value Wp  = lowerExpr(*C.Args[3]);
          mlir::Value Wo  = lowerExpr(*C.Args[4]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_ik_solve"));
          return emitUnreg("matlab.call_builtin", {Obj, Tg, Q0, Wp, Wo}, PtrTy, L, {Cal});
        }
        /* ===== Robotics Tier-5 — diffdrive derivative / occmap methods / PRM / pursuit == */
        if (Nm == "derivative" && Cls0 &&
            (Cn0 == "differentialDriveKinematics" || Cn0 == "unicycleKinematics" ||
             Cn0 == "bicycleKinematics" || Cn0 == "ackermannKinematics") &&
            C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = lowerExpr(*C.Args[1]);
          mlir::Value Cm  = lowerExpr(*C.Args[2]);
          const char *rt;
          if      (Cn0 == "unicycleKinematics")  rt = "matlab_robotics_unicycle_derivative";
          else if (Cn0 == "bicycleKinematics")   rt = "matlab_robotics_bicycle_derivative";
          else if (Cn0 == "ackermannKinematics") rt = "matlab_robotics_ackermann_derivative";
          else                                   rt = "matlab_robotics_diffdrive_derivative";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Obj, St, Cm}, PtrTy, L, {Cal});
        }
        if (Nm == "setOccupancy" && Cls0 && Cn0 == "binaryOccupancyMap" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value XY  = lowerExpr(*C.Args[1]);
          mlir::Value V   = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_occmap_set"));
          return emitUnreg("matlab.call_builtin", {Obj, XY, V}, PtrTy, L, {Cal});
        }
        if (Nm == "getOccupancy" && Cls0 && Cn0 == "binaryOccupancyMap" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value XY  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_occmap_get"));
          return emitUnreg("matlab.call_builtin", {Obj, XY}, PtrTy, L, {Cal});
        }
        if (Nm == "checkOccupancy" && Cls0 && Cn0 == "binaryOccupancyMap" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value XY  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_occmap_check"));
          return emitUnreg("matlab.call_builtin", {Obj, XY}, PtrTy, L, {Cal});
        }
        if (Nm == "findpath" && Cls0 && Cn0 == "mobileRobotPRM" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Sxy = lowerExpr(*C.Args[1]);
          mlir::Value Gxy = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_prm_findpath"));
          return emitUnreg("matlab.call_builtin", {Obj, Sxy, Gxy}, PtrTy, L, {Cal});
        }
        if (Nm == "step" && Cls0 && Cn0 == "controllerPurePursuit" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Po  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_pursuit_step"));
          return emitUnreg("matlab.call_builtin", {Obj, Po}, PtrTy, L, {Cal});
        }
        /* ===== Robotics Tier-6 — collision check + manipulatorRRT plan ===== */
        if (Nm == "checkCollision" && Cls0 &&
            (Cn0 == "collisionBox" || Cn0 == "collisionSphere" ||
             Cn0 == "collisionCylinder" || Cn0 == "collisionCapsule") &&
            C.Args.size() == 2) {
          mlir::Value A = loadObj(C.Args[0]);
          mlir::Value B = loadObj(C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_checkCollision"));
          return emitUnreg("matlab.call_builtin", {A, B}, PtrTy, L, {Cal});
        }
        if (Nm == "plan" && Cls0 && Cn0 == "manipulatorRRT" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qs  = lowerExpr(*C.Args[1]);
          mlir::Value Qg  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_robotics_rrt_plan"));
          return emitUnreg("matlab.call_builtin", {Obj, Qs, Qg}, PtrTy, L, {Cal});
        }
        /* ===== Reinforcement Learning Tier 1 — method dispatch ===========
         * Keyed on arg-0's pinned class.  getObservationInfo/getActionInfo
         * build a fresh rlFiniteSetSpec; train/sim/getCritic operate on the
         * agent; getLearnableParameters reads the critic's Q table. */
        if ((Nm == "getObservationInfo" || Nm == "getActionInfo") &&
            Cls0 && Cn0 == "rlMDPEnv" && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rlFiniteSetSpec__rlFiniteSetSpec"));
          mlir::Value Spec = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Env  = loadObj(C.Args[0]);
          const char *rt = (Nm == "getObservationInfo") ? "matlab_rl_obs_info"
                                                         : "matlab_rl_act_info";
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          emitUnregOp("matlab.call_builtin", {Spec, Env},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Spec;
        }
        if (Nm == "train" && Cls0 &&
            (Cn0 == "rlQAgent" || Cn0 == "rlSARSAAgent" || Cn0 == "rlDQNAgent" ||
             Cn0 == "rlPGAgent" || Cn0 == "rlDDPGAgent" || Cn0 == "rlTD3Agent" ||
             Cn0 == "rlPPOAgent" || Cn0 == "rlSACAgent" ||
             Cn0 == "rlGRPOAgent" || Cn0 == "rlTRPOAgent") && C.Args.size() == 3) {
          mlir::Value Ag = loadObj(C.Args[0]);
          mlir::Value En = loadObj(C.Args[1]);
          mlir::Value Op = loadObj(C.Args[2]);
          const char *rt = (Cn0 == "rlDQNAgent")  ? "matlab_rl_dqn_train"
                         : (Cn0 == "rlPGAgent")   ? "matlab_rl_pg_train"
                         : (Cn0 == "rlDDPGAgent") ? "matlab_rl_ddpg_train"
                         : (Cn0 == "rlTD3Agent")  ? "matlab_rl_td3_train"
                         : (Cn0 == "rlPPOAgent")  ? "matlab_rl_ppo_train"
                         : (Cn0 == "rlSACAgent")  ? "matlab_rl_sac_train"
                         : (Cn0 == "rlGRPOAgent") ? "matlab_rl_grpo_train"
                         : (Cn0 == "rlTRPOAgent") ? "matlab_rl_trpo_train"
                                                  : "matlab_rl_train";
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Ag, En, Op}, PtrTy, L, {Cal});
        }
        if (Nm == "sim" && Cls0 &&
            (Cn0 == "rlQAgent" || Cn0 == "rlSARSAAgent" || Cn0 == "rlDQNAgent" ||
             Cn0 == "rlPGAgent" || Cn0 == "rlDDPGAgent" || Cn0 == "rlTD3Agent" ||
             Cn0 == "rlPPOAgent" || Cn0 == "rlSACAgent" ||
             Cn0 == "rlGRPOAgent" || Cn0 == "rlTRPOAgent") && C.Args.size() == 2) {
          mlir::Value Ag = loadObj(C.Args[0]);
          mlir::Value En = loadObj(C.Args[1]);
          const char *rt = (Cn0 == "rlDQNAgent")  ? "matlab_rl_dqn_sim"
                         : (Cn0 == "rlPGAgent")   ? "matlab_rl_pg_sim"
                         : (Cn0 == "rlDDPGAgent") ? "matlab_rl_ddpg_sim"
                         : (Cn0 == "rlTD3Agent")  ? "matlab_rl_td3_sim"
                         : (Cn0 == "rlPPOAgent")  ? "matlab_rl_ppo_sim"
                         : (Cn0 == "rlSACAgent")  ? "matlab_rl_sac_sim"
                         : (Cn0 == "rlGRPOAgent") ? "matlab_rl_grpo_sim"
                         : (Cn0 == "rlTRPOAgent") ? "matlab_rl_trpo_sim"
                                                  : "matlab_rl_sim";
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Ag, En}, PtrTy, L, {Cal});
        }
        if (Nm == "getCritic" && Cls0 &&
            (Cn0 == "rlQAgent" || Cn0 == "rlSARSAAgent") && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rlQValueFunction__rlQValueFunction"));
          mlir::Value Qvf = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Ag  = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_get_critic"));
          emitUnregOp("matlab.call_builtin", {Qvf, Ag},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Qvf;
        }
        if (Nm == "getLearnableParameters" && Cls0 &&
            Cn0 == "rlQValueFunction" && C.Args.size() == 1) {
          mlir::Value Cr = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_get_params"));
          return emitUnreg("matlab.call_builtin", {Cr}, PtrTy, L, {Cal});
        }
        /* getAction / getMaxQValue (agent|policy, obs) — query the greedy
         * policy.  Dispatch on any value-based agent or extracted policy. */
        if ((Nm == "getAction" || Nm == "getMaxQValue") && Cls0 &&
            (Cn0 == "rlDQNAgent" || Cn0 == "rlPGAgent" || Cn0 == "rlQAgent" ||
             Cn0 == "rlSARSAAgent" || Cn0 == "rlMaxQPolicy") &&
            C.Args.size() == 2) {
          mlir::Value Ag = loadObj(C.Args[0]);
          mlir::Value Ob = lowerExpr(*C.Args[1]);
          const char *rt = (Nm == "getAction") ? "matlab_rl_get_action"
                                                : "matlab_rl_get_maxq";
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, rt));
          return emitUnreg("matlab.call_builtin", {Ag, Ob}, PtrTy, L, {Cal});
        }
        /* getGreedyPolicy(agent) → rlMaxQPolicy carrying a copy of the net/Q. */
        if (Nm == "getGreedyPolicy" && Cls0 &&
            (Cn0 == "rlDQNAgent" || Cn0 == "rlPGAgent" || Cn0 == "rlQAgent" ||
             Cn0 == "rlSARSAAgent") && C.Args.size() == 1) {
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "rlMaxQPolicy__rlMaxQPolicy"));
          mlir::Value Pol = emitUnreg("matlab.call", {}, PtrTyConst, L, {CtorCal});
          mlir::Value Ag  = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_rl_greedy_policy"));
          emitUnregOp("matlab.call_builtin", {Pol, Ag},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
          return Pol;
        }
        /* ===== Navigation Toolbox — method + free-function dispatch =======
         * All keyed on arg-0's pinned class (the Robotics precedent). */
        /* occupancyMap: setOccupancy / getOccupancy / checkOccupancy / inflate. */
        if (Nm == "setOccupancy" && Cls0 && Cn0 == "occupancyMap" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value XY  = lowerExpr(*C.Args[1]);
          mlir::Value V   = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_occmap_set"));
          return emitUnreg("matlab.call_builtin", {Obj, XY, V}, PtrTy, L, {Cal});
        }
        if (Nm == "getOccupancy" && Cls0 && Cn0 == "occupancyMap" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value XY  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_occmap_get"));
          return emitUnreg("matlab.call_builtin", {Obj, XY}, PtrTy, L, {Cal});
        }
        if (Nm == "checkOccupancy" && Cls0 && Cn0 == "occupancyMap" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value XY  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_occmap_check"));
          return emitUnreg("matlab.call_builtin", {Obj, XY}, PtrTy, L, {Cal});
        }
        if (Nm == "inflate" && Cls0 && Cn0 == "occupancyMap" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Rd  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_occmap_inflate"));
          return emitUnreg("matlab.call_builtin", {Obj, Rd}, PtrTy, L, {Cal});
        }
        /* state space: distance / interpolate / sampleUniform. */
        if (Nm == "distance" && Cls0 &&
            (Cn0 == "stateSpaceSE2" || Cn0 == "stateSpaceDubins") && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value S1  = lowerExpr(*C.Args[1]);
          mlir::Value S2  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_ss_distance"));
          return emitUnreg("matlab.call_builtin", {Obj, S1, S2}, PtrTy, L, {Cal});
        }
        if (Nm == "interpolate" && Cls0 &&
            (Cn0 == "stateSpaceSE2" || Cn0 == "stateSpaceDubins") && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value S1  = lowerExpr(*C.Args[1]);
          mlir::Value S2  = lowerExpr(*C.Args[2]);
          mlir::Value Rt  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_ss_interpolate"));
          return emitUnreg("matlab.call_builtin", {Obj, S1, S2, Rt}, PtrTy, L, {Cal});
        }
        if (Nm == "sampleUniform" && Cls0 &&
            (Cn0 == "stateSpaceSE2" || Cn0 == "stateSpaceDubins") && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_ss_sample"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        /* validator: isStateValid / isMotionValid. */
        if (Nm == "isStateValid" && Cls0 && Cn0 == "validatorOccupancyMap" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value S   = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_validator_isstate"));
          return emitUnreg("matlab.call_builtin", {Obj, S}, PtrTy, L, {Cal});
        }
        if (Nm == "isMotionValid" && Cls0 && Cn0 == "validatorOccupancyMap" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value S1  = lowerExpr(*C.Args[1]);
          mlir::Value S2  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_validator_ismotion"));
          return emitUnreg("matlab.call_builtin", {Obj, S1, S2}, PtrTy, L, {Cal});
        }
        /* navPath: pathLength + shortenpath(np, validator). */
        if (Nm == "pathLength" && Cls0 && Cn0 == "navPath" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_path_length"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        if (Nm == "shortenpath" && Cls0 && Cn0 == "navPath" && C.Args.size() == 2) {
          mlir::Value Np = loadObj(C.Args[0]);
          mlir::Value Sv = loadObj(C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_shortenpath"));
          return emitUnreg("matlab.call_builtin", {Np, Sv}, PtrTy, L, {Cal});
        }
        /* planners: plan(planner, start, goal). */
        if (Nm == "plan" && Cls0 &&
            (Cn0 == "plannerRRT" || Cn0 == "plannerRRTStar") && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = lowerExpr(*C.Args[1]);
          mlir::Value Go  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_planner_plan"));
          return emitUnreg("matlab.call_builtin", {Obj, St, Go}, PtrTy, L, {Cal});
        }
        if (Nm == "plan" && Cls0 && Cn0 == "plannerAStarGrid" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value St  = lowerExpr(*C.Args[1]);
          mlir::Value Go  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_astar_plan"));
          return emitUnreg("matlab.call_builtin", {Obj, St, Go}, PtrTy, L, {Cal});
        }
        /* lidar: matchScans(ref, cur) + addScan(slam, scan). */
        if (Nm == "matchScans" && Cls0 && Cn0 == "lidarScan" && C.Args.size() == 2) {
          mlir::Value Rf = loadObj(C.Args[0]);
          mlir::Value Cu = loadObj(C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_matchscans"));
          return emitUnreg("matlab.call_builtin", {Rf, Cu}, PtrTy, L, {Cal});
        }
        if (Nm == "addScan" && Cls0 && Cn0 == "lidarSLAM" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Sc  = loadObj(C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_slam_addscan"));
          return emitUnreg("matlab.call_builtin", {Obj, Sc}, PtrTy, L, {Cal});
        }
        /* poseGraph: addRelativePose + optimizePoseGraph. */
        if (Nm == "addRelativePose" && Cls0 && Cn0 == "poseGraph" &&
            (C.Args.size() == 2 || C.Args.size() == 4)) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Rel = lowerExpr(*C.Args[1]);
          mlir::Value Fr, To;
          if (C.Args.size() == 4) { Fr = lowerExpr(*C.Args[2]); To = lowerExpr(*C.Args[3]); }
          else {
            Fr = emitUnreg("matlab.const_float", {}, mlir::Float64Type::get(&MCtx), L,
                {mlir::NamedAttribute(mlir::StringAttr::get(&MCtx, "value"),
                     mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx), 0.0))});
            To = Fr;
          }
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_posegraph_addrel"));
          return emitUnreg("matlab.call_builtin", {Obj, Rel, Fr, To}, PtrTy, L, {Cal});
        }
        if (Nm == "optimizePoseGraph" && Cls0 && Cn0 == "poseGraph" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_posegraph_optimize"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        /* ===== Navigation Tier-5/6 — method + free-fn dispatch =========== */
        // controllerVFH: step(vfh, ranges, angles, targetDir) -> steering.
        if (Nm == "step" && Cls0 && Cn0 == "controllerVFH" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Rg  = lowerExpr(*C.Args[1]);
          mlir::Value An  = lowerExpr(*C.Args[2]);
          mlir::Value Td  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_vfh_step"));
          return emitUnreg("matlab.call_builtin", {Obj, Rg, An, Td}, PtrTy, L, {Cal});
        }
        // monteCarloLocalization: step(mcl, odom, ranges, angles) -> pose.
        if (Nm == "step" && Cls0 && Cn0 == "monteCarloLocalization" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Od  = lowerExpr(*C.Args[1]);
          mlir::Value Rg  = lowerExpr(*C.Args[2]);
          mlir::Value An  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_mcl_step"));
          return emitUnreg("matlab.call_builtin", {Obj, Od, Rg, An}, PtrTy, L, {Cal});
        }
        // stateEstimatorPF: initialize / predict / correct / getStateEstimate.
        if (Nm == "initialize" && Cls0 && Cn0 == "stateEstimatorPF" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Nn  = lowerExpr(*C.Args[1]);
          mlir::Value Mn  = lowerExpr(*C.Args[2]);
          mlir::Value Cv  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_pf_initialize"));
          return emitUnreg("matlab.call_builtin", {Obj, Nn, Mn, Cv}, PtrTy, L, {Cal});
        }
        if (Nm == "predict" && Cls0 && Cn0 == "stateEstimatorPF" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Av  = lowerExpr(*C.Args[1]);
          mlir::Value Qv  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_pf_predict"));
          return emitUnreg("matlab.call_builtin", {Obj, Av, Qv}, PtrTy, L, {Cal});
        }
        if (Nm == "correct" && Cls0 && Cn0 == "stateEstimatorPF" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Zv  = lowerExpr(*C.Args[1]);
          mlir::Value Hv  = lowerExpr(*C.Args[2]);
          mlir::Value Rv  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_pf_correct"));
          return emitUnreg("matlab.call_builtin", {Obj, Zv, Hv, Rv}, PtrTy, L, {Cal});
        }
        if (Nm == "getStateEstimate" && Cls0 && Cn0 == "stateEstimatorPF" && C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_pf_estimate"));
          return emitUnreg("matlab.call_builtin", {Obj}, PtrTy, L, {Cal});
        }
        // gnssSensor: step(gnss, lla, vel) -> noisy [lla vel].
        if (Nm == "step" && Cls0 && Cn0 == "gnssSensor" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value La  = lowerExpr(*C.Args[1]);
          mlir::Value Ve  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_gnss_step"));
          return emitUnreg("matlab.call_builtin", {Obj, La, Ve}, PtrTy, L, {Cal});
        }
        // referencePathFrenet: global2frenet / frenet2global.
        if (Nm == "global2frenet" && Cls0 && Cn0 == "referencePathFrenet" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Gv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_frenet_g2f"));
          return emitUnreg("matlab.call_builtin", {Obj, Gv}, PtrTy, L, {Cal});
        }
        if (Nm == "frenet2global" && Cls0 && Cn0 == "referencePathFrenet" && C.Args.size() == 2) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Fv  = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_frenet_f2g"));
          return emitUnreg("matlab.call_builtin", {Obj, Fv}, PtrTy, L, {Cal});
        }
        // trajectoryGeneratorFrenet: connect(trajgen, init, term, T) -> traj.
        if (Nm == "connect" && Cls0 && Cn0 == "trajectoryGeneratorFrenet" && C.Args.size() == 4) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value In  = lowerExpr(*C.Args[1]);
          mlir::Value Tm  = lowerExpr(*C.Args[2]);
          mlir::Value Tt  = lowerExpr(*C.Args[3]);
          mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_nav_trajgen_connect"));
          return emitUnreg("matlab.call_builtin", {Obj, In, Tm, Tt}, PtrTy, L, {Cal});
        }
        /* recursiveLS step(obj, y, H) — RLS update with user regressor H. */
        if (Nm == "step" && Cls0 && Cn0 == "recursiveLS" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Yv  = lowerExpr(*C.Args[1]);
          mlir::Value Hv  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_rls_step"));
          return emitUnreg("matlab.call_builtin", {Obj, Yv, Hv}, PtrTy, L, {Cal});
        }
        /* recursiveARX step(obj, y, u) — RLS update from buffered I/O. */
        if (Nm == "step" && Cls0 && Cn0 == "recursiveARX" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Yv  = lowerExpr(*C.Args[1]);
          mlir::Value Uv  = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_rarx_step"));
          return emitUnreg("matlab.call_builtin", {Obj, Yv, Uv}, PtrTy, L, {Cal});
        }
        /* predict(model, data [, K]) — K-step predictor (default K=1). */
        /* Stats Tier-3 — predict(LinearModel, Xnew) on a fitlm/fitglm model. */
        if (Nm == "predict" && Cls0 && Cn0 == "LinearModel" && C.Args.size() == 2) {
          mlir::Value Mdl  = loadObj(C.Args[0]);
          mlir::Value Xnew = lowerExpr(*C.Args[1]);   /* data: plain lower (inline-matrix safe) */
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_stats_lm_predict"));
          return emitUnreg("matlab.call_builtin", {Mdl, Xnew}, PtrTy, L, {Cal});
        }
        /* Stats Tier-5 — predict(ClassificationModel, Xnew). */
        if (Nm == "predict" && Cls0 && Cn0 == "ClassificationModel" && C.Args.size() == 2) {
          mlir::Value Mdl  = loadObj(C.Args[0]);
          mlir::Value Xnew = lowerExpr(*C.Args[1]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_stats_clf_predict"));
          return emitUnreg("matlab.call_builtin", {Mdl, Xnew}, PtrTy, L, {Cal});
        }
        if (Nm == "predict" && Cls0 && Cn0 == "idpoly" &&
            (C.Args.size() == 2 || C.Args.size() == 3)) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::Value Data  = loadObj(C.Args[1]);
          mlir::Value K = (C.Args.size() == 3)
                              ? lowerExpr(*C.Args[2])
                              : emitUnreg("matlab.const_float", {}, F64, L,
                                          {mlir::NamedAttribute(
                                              mlir::StringAttr::get(&MCtx, "value"),
                                              mlir::FloatAttr::get(F64, 1.0))});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_predict"));
          return emitUnreg("matlab.call_builtin", {Model, Data, K}, PtrTy, L, {Cal});
        }
        /* compare(data, model) — NRMSE fit % (scalar).  Routes to the
         * state-space comparator when the model arg is an idss, else the
         * polynomial one. */
        if (Nm == "compare" && Cls0 && Cn0 == "iddata" && C.Args.size() == 2) {
          mlir::Value Data  = loadObj(C.Args[0]);
          mlir::Value Model = loadObj(C.Args[1]);
          bool modelIsSS = false;
          if (auto *AN1 = dynamic_cast<const NameExpr *>(C.Args[1]))
            if (AN1->Ref && AN1->Ref->PinnedClass &&
                (AN1->Ref->PinnedClass->Name == "idss" ||
                 AN1->Ref->PinnedClass->Name == "idgrey"))
              modelIsSS = true;
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, modelIsSS ? "matlab_ident_compare_ss"
                                                     : "matlab_ident_compare"));
          return emitUnreg("matlab.call_builtin", {Data, Model}, F64, L, {Cal});
        }
        /* delayest(data) — estimate the input transport delay (scalar). */
        if (Nm == "delayest" && Cls0 && Cn0 == "iddata" && C.Args.size() == 1) {
          mlir::Value Data = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_delayest"));
          return emitUnreg("matlab.call_builtin", {Data}, F64, L, {Cal});
        }
        /* fpe(model) / aic(model) — quality metrics (scalar). */
        if (Nm == "fpe" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 1) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_fpe"));
          return emitUnreg("matlab.call_builtin", {Model}, F64, L, {Cal});
        }
        if (Nm == "aic" && Cls0 && Cn0 == "idpoly" && C.Args.size() == 1) {
          mlir::Value Model = loadObj(C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ident_aic"));
          return emitUnreg("matlab.call_builtin", {Model}, F64, L, {Cal});
        }

        /* MPC Tier-0 — kalman(sys, Qn, Rn) sys-form.  When the first
         * arg is class-pinned to `ss`, extract A / B / C / Ts and
         * route to matlab_kalman_sys_L, which picks the continuous-
         * or discrete-time kernel based on Ts.  B reused as the
         * noise-input matrix G (MPC User's Guide §1.4 canonical
         * input-channel-noise assumption). */
        if (Nm == "kalman" && Cls0 && Cn0 == "ss" && C.Args.size() == 3) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value Qn  = lowerExpr(*C.Args[1]);
          mlir::Value Rn  = lowerExpr(*C.Args[2]);
          mlir::Value AVal = getProp(Obj, "A");
          mlir::Value BVal = getProp(Obj, "B");
          mlir::Value CVal = getProp(Obj, "C");
          // Ts read as a boxed 1×1 matrix (matches matrix-storage of
          // class scalar properties); matlab_kalman_sys_L unboxes
          // internally to pick the continuous vs discrete kernel.
          mlir::Value TsVal = getProp(Obj, "Ts");
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_kalman_sys_L"));
          return emitUnreg("matlab.call_builtin",
                           {AVal, BVal, CVal, Qn, Rn, TsVal},
                           PtrTy, L, {Cal});
        }

        /* §3.6 feedback(sys1, sys2) / series(sys1, sys2) /
         * parallel(sys1, sys2) — strictly-proper closed-loop
         * assembly. Result is a fresh ss(Acl, Bcl, Ccl, sys1.D)
         * where (Acl, Bcl, Ccl) come from the matching
         * matlab_<name>_ss_{A,B,C} splitter. */
        if ((Nm == "feedback" || Nm == "series" || Nm == "parallel") &&
            Cls0 && Cn0 == "ss" && C.Args.size() == 2 && C.Args[1]) {
          auto *AN1 = dynamic_cast<const NameExpr *>(C.Args[1]);
          if (AN1 && AN1->Ref && AN1->Ref->PinnedClass &&
              AN1->Ref->PinnedClass->Name == "ss") {
            mlir::Value O1 = loadObj(C.Args[0]);
            mlir::Value O2 = loadObj(C.Args[1]);
            mlir::Value A1 = getProp(O1, "A");
            mlir::Value B1 = getProp(O1, "B");
            mlir::Value C1 = getProp(O1, "C");
            mlir::Value A2 = getProp(O2, "A");
            mlir::Value B2 = getProp(O2, "B");
            mlir::Value C2 = getProp(O2, "C");
            llvm::SmallVector<mlir::Value, 6> Ssa{A1, B1, C1, A2, B2, C2};
            std::string PfA = "matlab_" + std::string(Nm) + "_ss_A";
            std::string PfB = "matlab_" + std::string(Nm) + "_ss_B";
            std::string PfC = "matlab_" + std::string(Nm) + "_ss_C";
            mlir::NamedAttribute CalA(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, PfA));
            mlir::NamedAttribute CalB(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, PfB));
            mlir::NamedAttribute CalCC(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, PfC));
            mlir::Value Acl = emitUnreg("matlab.call_builtin", Ssa,
                                         PtrTy, L, {CalA});
            mlir::Value Bcl = emitUnreg("matlab.call_builtin", Ssa,
                                         PtrTy, L, {CalB});
            mlir::Value Ccl = emitUnreg("matlab.call_builtin", Ssa,
                                         PtrTy, L, {CalCC});
            mlir::Value D1 = getProp(O1, "D");
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "ss__ss"));
            return emitUnreg("matlab.call",
                             {Acl, Bcl, Ccl, D1},
                             PtrTy, L, {CtorCal});
          }
        }

        /* §5.1 sminreal(sys) — structural minimal realisation.
         * Returns a fresh ss with non-reachable + non-observable
         * states dropped. Routes through matlab_sminreal_{A,B,C}. */
        if (Nm == "sminreal" && Cls0 && Cn0 == "ss" &&
            C.Args.size() == 1) {
          mlir::Value Obj = loadObj(C.Args[0]);
          mlir::Value AVal = getProp(Obj, "A");
          mlir::Value BVal = getProp(Obj, "B");
          mlir::Value CVal = getProp(Obj, "C");
          mlir::Value DVal = getProp(Obj, "D");
          llvm::SmallVector<mlir::Value, 3> Ssa{AVal, BVal, CVal};
          mlir::NamedAttribute CalA(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_sminreal_A"));
          mlir::NamedAttribute CalB(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_sminreal_B"));
          mlir::NamedAttribute CalCC(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_sminreal_C"));
          mlir::Value As = emitUnreg("matlab.call_builtin", Ssa,
                                       PtrTy, L, {CalA});
          mlir::Value Bs = emitUnreg("matlab.call_builtin", Ssa,
                                       PtrTy, L, {CalB});
          mlir::Value Cs = emitUnreg("matlab.call_builtin", Ssa,
                                       PtrTy, L, {CalCC});
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ss__ss"));
          return emitUnreg("matlab.call",
                           {As, Bs, Cs, DVal},
                           PtrTy, L, {CtorCal});
        }

        /* §5.1 modred(sys, elim, method) — modal residualisation.
         * `elim` is a vector of state indices to drop; `method` is
         * the string 'Truncate' or 'MatchDC'. Result is a fresh ss
         * via the matlab_modred_{A,B,C} runtime triple. */
        if (Nm == "modred" && Cls0 && Cn0 == "ss" &&
            (C.Args.size() == 2 || C.Args.size() == 3)) {
          mlir::Value Obj  = loadObj(C.Args[0]);
          mlir::Value Elim = lowerExpr(*C.Args[1]);
          /* Box scalar f64 elim into a 1×1 matrix so the runtime
           * sees a uniform ptr operand (covers `modred(sys, [2],
           * 'Truncate')` where [2] collapses to scalar). */
          if (Elim.getType() == F64) {
            mlir::NamedAttribute BoxCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_mat_from_scalar"));
            Elim = emitUnreg("matlab.call_builtin", {Elim}, PtrTy, L,
                              {BoxCal});
          }
          if (Elim.getType() != PtrTy) Elim.setType(PtrTy);
          /* Pick method_id from the third arg's char/string literal.
           * Default = 0 (Truncate) when no method is given. */
          double mid = 0.0;
          if (C.Args.size() == 3 && C.Args[2]) {
            const CharLiteral *CL =
                dynamic_cast<const CharLiteral *>(C.Args[2]);
            const StringLiteral *SL =
                CL ? nullptr
                   : dynamic_cast<const StringLiteral *>(C.Args[2]);
            llvm::StringRef Tok =
                CL ? CL->Value : (SL ? SL->Value : "");
            if (Tok == "MatchDC") mid = 1.0;
          }
          mlir::Value MidV = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, mid)).getResult();
          mlir::Value AVal = getProp(Obj, "A");
          mlir::Value BVal = getProp(Obj, "B");
          mlir::Value CVal = getProp(Obj, "C");
          mlir::Value DVal = getProp(Obj, "D");
          llvm::SmallVector<mlir::Value, 5> Ssa{AVal, BVal, CVal, Elim, MidV};
          mlir::NamedAttribute CalA(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_modred_A"));
          mlir::NamedAttribute CalB(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_modred_B"));
          mlir::NamedAttribute CalCC(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_modred_C"));
          mlir::Value Ar = emitUnreg("matlab.call_builtin", Ssa,
                                       PtrTy, L, {CalA});
          mlir::Value Br = emitUnreg("matlab.call_builtin", Ssa,
                                       PtrTy, L, {CalB});
          mlir::Value Cr = emitUnreg("matlab.call_builtin", Ssa,
                                       PtrTy, L, {CalCC});
          mlir::NamedAttribute CtorCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "ss__ss"));
          return emitUnreg("matlab.call",
                           {Ar, Br, Cr, DVal},
                           PtrTy, L, {CtorCal});
        }

        /* §5.2 append(sys1, sys2) / blkdiag(sys1, sys2) — block-
         * diagonal MIMO append. Same shape as feedback/series above
         * but routes to matlab_append_ss_{A,B,C}. Result D is
         * block-diagonal(D1, D2) — strictly-proper plants get
         * D = sys1.D directly (zeros). */
        if ((Nm == "append" || Nm == "blkdiag") &&
            Cls0 && Cn0 == "ss" && C.Args.size() == 2 && C.Args[1]) {
          auto *AN1 = dynamic_cast<const NameExpr *>(C.Args[1]);
          if (AN1 && AN1->Ref && AN1->Ref->PinnedClass &&
              AN1->Ref->PinnedClass->Name == "ss") {
            mlir::Value O1 = loadObj(C.Args[0]);
            mlir::Value O2 = loadObj(C.Args[1]);
            mlir::Value A1 = getProp(O1, "A");
            mlir::Value B1 = getProp(O1, "B");
            mlir::Value C1 = getProp(O1, "C");
            mlir::Value A2 = getProp(O2, "A");
            mlir::Value B2 = getProp(O2, "B");
            mlir::Value C2 = getProp(O2, "C");
            llvm::SmallVector<mlir::Value, 6> Ssa{A1, B1, C1, A2, B2, C2};
            mlir::NamedAttribute CalA(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_append_ss_A"));
            mlir::NamedAttribute CalB(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_append_ss_B"));
            mlir::NamedAttribute CalCC(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_append_ss_C"));
            mlir::Value Aa = emitUnreg("matlab.call_builtin", Ssa,
                                        PtrTy, L, {CalA});
            mlir::Value Ba = emitUnreg("matlab.call_builtin", Ssa,
                                        PtrTy, L, {CalB});
            mlir::Value Ca = emitUnreg("matlab.call_builtin", Ssa,
                                        PtrTy, L, {CalCC});
            mlir::Value D1 = getProp(O1, "D");
            mlir::NamedAttribute CtorCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "ss__ss"));
            return emitUnreg("matlab.call",
                             {Aa, Ba, Ca, D1},
                             PtrTy, L, {CtorCal});
          }
        }
      }
      /* disp(s) where s is a tracked string binding -> matlab_string_disp.
       * Also handles disp("literal") by routing a StringLiteral arg
       * and disp(expr) where expr is a call to a known string-
       * returning builtin (e.g. disp(upper(s))). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1) {
        bool IsStr = false;
        if (C.Args[0]->Kind == NodeKind::StringLiteral) IsStr = true;
        else if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          if (AN->Ref && StringBindings.count(AN->Ref) > 0) IsStr = true;
          /* Cross-REPL-input fallback: this compilation may not have
           * seen the assigning input that populated StringBindings;
           * use the binding's persisted InferredType. */
          else if (AN->Ref && AN->Ref->InferredType &&
                   AN->Ref->InferredType->K == Type::Kind::StringArray)
            IsStr = true;
        }
        else if (auto *CC = dynamic_cast<const CallOrIndex *>(C.Args[0])) {
          if (auto *CN = dynamic_cast<const NameExpr *>(CC->Callee)) {
            if (CN->Ref && CN->Ref->Kind == BindingKind::Builtin) {
              auto Nm = CN->Name;
              if (Nm == "fgetl" || Nm == "sprintf" || Nm == "num2str" ||
                  Nm == "upper" || Nm == "lower" || Nm == "strtrim" ||
                  Nm == "strrep" || Nm == "strcat" || Nm == "regexprep" ||
                  Nm == "char" ||  /* #234 — char([codes]) / char(code) is a string */
                  Nm == "bin" || Nm == "hex" || Nm == "dec")
                IsStr = true;
            }
          }
        }
        /* disp(c{k}) where c{k} is a string-typed cell element (#206). */
        else if (dynamic_cast<const CellIndex *>(C.Args[0]) &&
                 isStringExpr(C.Args[0]))
          IsStr = true;
        if (IsStr) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_disp"));
          return emitUnreg("matlab.call_builtin", {V},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }
      /* Phase 1.1.C — disp(typed_int_matrix). When the arg's Sema
       * type is a non-scalar Int32 / UInt8 array, route through the
       * typed disp entry (matlab_mat_i32_disp / matlab_mat_u8_disp)
       * instead of the polymorphic matlab_disp_mat path which expects
       * f64 layout. The check works for both NameExpr (binding's
       * InferredType is propagated to the AST node) and direct
       * `disp(int32(M))` because Sema annotates the CallOrIndex's
       * inferred type when the cast result is well-typed. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1) {
        const Expr *Arg = C.Args[0];
        const Type *ArgTy = Arg ? Arg->Ty : nullptr;
        /* NameExpr cross-REPL fallback: when the AST didn't get a
         * fresh Ty in this compile, fall back to the binding's
         * persisted InferredType. */
        if ((!ArgTy || ArgTy->K != Type::Kind::Array)) {
          if (auto *AN = dynamic_cast<const NameExpr *>(Arg))
            if (AN->Ref && AN->Ref->InferredType)
              ArgTy = AN->Ref->InferredType;
        }
        if (ArgTy && ArgTy->K == Type::Kind::Array) {
          auto &AT = static_cast<const ArrayType &>(*ArgTy);
          if (AT.S.K != Shape::Rank::Scalar &&
              (AT.Elt == Dtype::Int32 || AT.Elt == Dtype::UInt8)) {
            mlir::Value V = lowerExpr(*Arg);
            llvm::StringRef Suf =
                (AT.Elt == Dtype::Int32) ? "i32" : "u8";
            std::string Callee = ("matlab_mat_" + Suf + "_disp").str();
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
        }
      }
      /* strlen(s) on a string binding -> matlab_string_len. The
       * cross-REPL-input fallback consults the binding's
       * InferredType (seeded by the resolver's workspace hook) so
       * a fresh-input `strlen(t)` after an earlier `t = "..."` still
       * routes to the string runtime instead of leaving an
       * unconvertible matlab.call_builtin in the JIT module. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "strlen" && C.Args.size() == 1) {
        auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]);
        bool IsStr = AN && AN->Ref &&
                     (StringBindings.count(AN->Ref) ||
                      (AN->Ref->InferredType &&
                       AN->Ref->InferredType->K == Type::Kind::StringArray));
        if (IsStr) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::Value V = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_len"));
          return emitUnreg("matlab.call_builtin", {V}, F64, L, {Cal});
        }
      }
      /* isstring(x) compile-time fold. Same cross-input fallback as
       * strlen above. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "isstring" && C.Args.size() == 1) {
        auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]);
        auto F64 = mlir::Float64Type::get(&MCtx);
        double Val = 0.0;
        if (C.Args[0]->Kind == NodeKind::StringLiteral) Val = 1.0;
        else if (AN && AN->Ref && StringBindings.count(AN->Ref)) Val = 1.0;
        else if (AN && AN->Ref && AN->Ref->InferredType &&
                 AN->Ref->InferredType->K == Type::Kind::StringArray)
          Val = 1.0;
        return mlir::arith::ConstantOp::create(
            B, L, F64, mlir::FloatAttr::get(F64, Val));
      }
      /* Phase 5.3: height(T), width(T), numel(T), size(T, dim) on a
       * table binding — dispatch through the matlab_table_* runtime. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "height" || N->Name == "width" ||
           N->Name == "numel"  || N->Name == "length") &&
          C.Args.size() == 1 && C.Args[0])
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          bool IsTT = ArgN->Ref && isTimetableBinding(ArgN->Ref);
          bool IsT  = ArgN->Ref && isTableBinding(ArgN->Ref);
          if (IsT || IsTT) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            const char *Prefix = IsTT ? "matlab_timetable_" : "matlab_table_";
            std::string Callee;
            if (N->Name == "height")      Callee = std::string(Prefix) + "height";
            else if (N->Name == "width")  Callee = std::string(Prefix) + "width";
            else if (N->Name == "numel")  Callee = std::string(Prefix) + "numel";
            else /* length */             Callee = std::string(Prefix) + "height";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call_builtin", {V}, F64, L, {Cal});
          }
        }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "size" && C.Args.size() == 2 && C.Args[0])
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          bool IsTT = ArgN->Ref && isTimetableBinding(ArgN->Ref);
          bool IsT  = ArgN->Ref && isTableBinding(ArgN->Ref);
          if (IsT || IsTT) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::Value D = lowerExpr(*C.Args[1]);
            const char *Callee = IsTT ? "matlab_timetable_size_dim"
                                       : "matlab_table_size_dim";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call_builtin", {V, D}, F64, L, {Cal});
          }
        }
      /* Phase 5.1: disp(t) where t is a datetime / duration binding —
       * dispatch to the typed runtime.
       * Phase 5.2: disp(c) for categorical too. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1 && C.Args[0]) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0])) {
          /* Vec checks before scalar so a binding tagged as both
           * scalar+vec (defensive) routes to the vec entry. */
          if (ArgN->Ref && DatetimeVecBindings.count(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_datetime_vec_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (ArgN->Ref && DurationVecBindings.count(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_duration_vec_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (ArgN->Ref && DatetimeBindings.count(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_datetime_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (ArgN->Ref && DurationBindings.count(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_duration_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (ArgN->Ref && CategoricalBindings.count(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_categorical_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (ArgN->Ref && isTableBinding(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_table_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          if (ArgN->Ref && isTimetableBinding(ArgN->Ref)) {
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_timetable_disp"));
            return emitUnreg("matlab.call_builtin", {V},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
        }
      }
      /* Phase 5.2: length(c) / numel(c) / categories(c) /
       * iscategory(c, name) on a categorical binding. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "length" || N->Name == "numel") &&
          C.Args.size() == 1 && C.Args[0])
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && CategoricalBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_categorical_length"));
            return emitUnreg("matlab.call_builtin", {V}, F64, L, {Cal});
          }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "categories" && C.Args.size() == 1 && C.Args[0])
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && CategoricalBindings.count(ArgN->Ref)) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_categorical_categories"));
            return emitUnreg("matlab.call_builtin", {V}, PtrTy, L, {Cal});
          }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "iscategory" && C.Args.size() == 2)
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && CategoricalBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value Carg = lowerExpr(*C.Args[0]);
            const Expr *KE = C.Args[1];
            mlir::Value K;
            if (auto *CL = dynamic_cast<const CharLiteral *>(KE)) {
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
              mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                          mlir::NoneType::get(&MCtx),
                                          L, {VA});
              mlir::NamedAttribute SCal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
              K = emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {SCal});
            } else {
              K = lowerExpr(*KE);
            }
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_categorical_iscategory"));
            return emitUnreg("matlab.call_builtin", {Carg, K},
                             F64, L, {Cal});
          }
      /* disp(ME.message) inside a catch body — route to the dedicated
       * matlab_err_disp_message runtime that prints the stored error
       * text. We only recognise the single-arg 'message' field on a
       * catch-var; other fields fall through to the generic struct
       * get path (which returns 0.0 for missing fields). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "disp" && C.Args.size() == 1) {
        if (auto *F = dynamic_cast<const FieldAccess *>(C.Args[0]))
          if (auto *B0 = dynamic_cast<const NameExpr *>(F->Base))
            if (B0->Ref && CatchBindings.count(B0->Ref) &&
                F->Field == "message") {
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx,
                                         "matlab_err_disp_message"));
              return emitUnreg("matlab.call_builtin", {},
                               mlir::NoneType::get(&MCtx), L, {Cal});
            }
      }
      /* isstruct(x): compile-time fold based on whether x's binding
       * has been initialised as a struct. Any other ptr (matrix) or
       * scalar returns 0.0. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "isstruct" && C.Args.size() == 1) {
        auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]);
        auto F64 = mlir::Float64Type::get(&MCtx);
        double Val = 0.0;
        if (ArgN && ArgN->Ref && StructInitialised.count(ArgN->Ref))
          Val = 1.0;
        return mlir::arith::ConstantOp::create(
            B, L, F64, mlir::FloatAttr::get(F64, Val));
      }
      /* dbg(x) / dbg(x, 'label') — source-located debug print to
       * stderr. Works like disp but prefixes the current file:line
       * and the argument's name (when the arg is a bare NameExpr;
       * otherwise the literal label or "<expr>").
       *
       * Routes to matlab_dbg_f64 for scalar args and matlab_dbg_mat
       * for matrix/ptr args. The filename is extracted from the
       * call site's SourceLocation so traces point at the .m
       * source, not the generated C / LLVM IR. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "dbg" &&
          (C.Args.size() == 1 || C.Args.size() == 2) &&
          C.Args[0]) {
        std::string FileName = "<repl>";
        int32_t Line = 0;
        if (SM && C.Range.Begin.isValid()) {
          FileName = std::string(SM->getName(C.Range.Begin.File));
          Line = (int32_t)SM->getLineColumn(C.Range.Begin).Line;
        }
        /* Pick a label: explicit 2nd-arg string, a NameExpr's name,
         * or the empty string (runtime substitutes "<expr>"). */
        std::string Label;
        if (C.Args.size() == 2 && C.Args[1]) {
          if (auto *Lit = dynamic_cast<const StringLiteral *>(C.Args[1]))
            Label = Lit->Value;
          else if (auto *Lit = dynamic_cast<const CharLiteral *>(C.Args[1]))
            Label = Lit->Value;
        }
        if (Label.empty())
          if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]))
            Label = std::string(AN->Name);

        mlir::Value V = lowerExpr(*C.Args[0]);
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        mlir::NamedAttribute FileA(
            mlir::StringAttr::get(&MCtx, "value"),
            mlir::StringAttr::get(&MCtx, FileName));
        mlir::Value FileV = emitUnreg("matlab.const_char", {},
                                       mlir::NoneType::get(&MCtx), L, {FileA});
        mlir::Value LineV = mlir::arith::ConstantOp::create(
            B, L, I32, mlir::IntegerAttr::get(I32, (int64_t)Line));
        mlir::NamedAttribute LabelA(
            mlir::StringAttr::get(&MCtx, "value"),
            mlir::StringAttr::get(&MCtx, Label));
        mlir::Value LabelV = emitUnreg("matlab.const_char", {},
                                        mlir::NoneType::get(&MCtx), L, {LabelA});
        bool IsMat = V && (V.getType() == PtrTy ||
                           mlir::isa<mlir::RankedTensorType,
                                     mlir::UnrankedTensorType>(V.getType()));
        llvm::StringRef Callee = IsMat ? "matlab_dbg_mat" : "matlab_dbg_f64";
        if (!V) {
          /* If lowerExpr couldn't produce a value, fall back to
           * f64(0) — better than crashing the REPL. */
          V = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, 0.0));
        }
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        return emitUnreg("matlab.call_builtin",
                         {FileV, LineV, LabelV, V},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      /* who / whos / clear — REPL workspace ergonomics. All route to
       * matlab_ws_* runtime entries directly. `clear()` with no args
       * clears the whole workspace; `clear x` (command syntax) and
       * `clear('x')` (function syntax) drop one name. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "who" && C.Args.empty()) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_ws_who"));
        return emitUnreg("matlab.call_builtin", {},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "whos" && C.Args.empty()) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_ws_whos"));
        return emitUnreg("matlab.call_builtin", {},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      /* pwd() — current directory as a matlab_string* (ptr). The bare-name
       * `pwd` form is lowered in the NameExpr value path; both emit
       * call_builtin @pwd, dispatched to matlab_pwd in LowerTensorOps. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "pwd" && C.Args.empty()) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "pwd"));
        return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
      }
      /* keyboard() — call form drops into the debugger pause same as the
       * bare `keyboard` statement. No-op in release (-g not set). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "keyboard" && C.Args.empty()) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_dbg_keyboard_hook"));
        return emitUnreg("matlab.call_builtin", {},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      /* tic() — call form. Same effect as bare `tic`: starts the
       * thread-local timer slot. Returns no value. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "tic" && C.Args.empty()) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_tic"));
        return emitUnreg("matlab.call_builtin", {},
                         mlir::NoneType::get(&MCtx), L, {Cal});
      }
      /* toc() — call form returns elapsed seconds (f64). Used in
       * expressions like `t = toc()` or `disp(toc())`. The bare-name
       * `toc` statement form prints "Elapsed time is ..." instead. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "toc" && C.Args.empty()) {
        auto F64 = mlir::Float64Type::get(&MCtx);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_toc"));
        return emitUnreg("matlab.call_builtin", {}, F64, L, {Cal});
      }
      /* pause() / pause(n) — call form. With no args, blocks for a
       * keypress; with one numeric arg, sleeps for that many seconds. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "pause") {
        if (C.Args.empty()) {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_pause_keypress"));
          return emitUnreg("matlab.call_builtin", {},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
        if (C.Args.size() == 1) {
          mlir::Value SecsV = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_pause"));
          return emitUnreg("matlab.call_builtin", {SecsV},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "clear") {
        if (C.Args.empty()) {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_clear"));
          return emitUnreg("matlab.call_builtin", {},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
        /* clear('x') / clear('x', 'y'): one runtime call per name. */
        for (const Expr *A : C.Args) {
          std::string Nm;
          if (auto *Lit = dynamic_cast<const StringLiteral *>(A)) Nm = Lit->Value;
          else if (auto *Lit = dynamic_cast<const CharLiteral *>(A)) Nm = Lit->Value;
          else continue;
          mlir::Value NameV = emitFieldNameChar(Nm, L);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_ws_clear_one"));
          emitUnregOp("matlab.call_builtin", {NameV},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
        }
        return emitUnreg("matlab.undef", {},
                         mlir::NoneType::get(&MCtx), L);
      }
      /* cd('dir') / cd("dir") — call form of the change-directory command.
       * A literal path routes to matlab_cd (chdir, in-process so it persists
       * across REPL turns). Mirrors `clear`'s literal-arg posture: a
       * non-literal argument (e.g. cd(pathVar)) is not yet supported. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "cd") {
        /* Return the call_builtin's result directly (NoneType) rather than a
         * trailing matlab.undef — a dangling undef survives the lowering
         * passes and fails LLVM translation. */
        if (C.Args.empty()) {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_cd_home"));
          return emitUnreg("matlab.call_builtin", {},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
        if (C.Args.size() == 1) {
          std::string Path;
          if (auto *Lit = dynamic_cast<const StringLiteral *>(C.Args[0]))
            Path = Lit->Value;
          else if (auto *Lit = dynamic_cast<const CharLiteral *>(C.Args[0]))
            Path = Lit->Value;
          if (!Path.empty()) {
            mlir::Value PathV = emitFieldNameChar(Path, L);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_cd"));
            return emitUnreg("matlab.call_builtin", {PathV},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
          /* Dynamic path: cd(pathVar) / cd(fullfile(...)) / cd(sprintf(...)).
           * Route any string-valued argument through matlab_cd_str, which
           * reads the chars from the matlab_string at runtime. Detect strings
           * the way disp does — current-turn StringBindings (isStringExpr)
           * plus the Sema StringArray type / persisted InferredType, so a
           * cross-REPL-turn `p = '...'; cd(p)` is still recognised. */
          const Expr *PA = C.Args[0];
          bool ArgIsString = isStringExpr(PA) ||
                             (PA->Ty && PA->Ty->K == Type::Kind::StringArray);
          if (!ArgIsString)
            if (auto *NE = dynamic_cast<const NameExpr *>(PA))
              if (NE->Ref && NE->Ref->InferredType &&
                  NE->Ref->InferredType->K == Type::Kind::StringArray)
                ArgIsString = true;
          if (ArgIsString) {
            mlir::Value PathV = lowerExpr(*C.Args[0]);
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            if (PathV.getType() != PtrTy) PathV.setType(PtrTy);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_cd_str"));
            return emitUnreg("matlab.call_builtin", {PathV},
                             mlir::NoneType::get(&MCtx), L, {Cal});
          }
        }
        /* A non-string argument (e.g. cd(42)) isn't a valid path — fall
         * through to the generic path so it reports an unsupported shape
         * instead of silently changing directory. */
      }
      /* #147: isequal on two STRING operands. matlab_isequal takes two
       * matlab_mat* and reads rows/cols/data, but a matlab_string has a
       * different layout, so `isequal("ab","ab")` mis-reads the string and
       * returns 0. Sema can't distinguish string from matrix at the runtime
       * boundary, but isStringExpr knows here — route a both-string isequal
       * to the strcmp path (matlab_strcmp; element-count + byte compare,
       * returns 1.0 when equal). Non-string isequal falls through to the
       * normal matlab_isequal. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "isequal" && C.Args.size() == 2 &&
          isStringExpr(C.Args[0]) && isStringExpr(C.Args[1])) {
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value A = lowerExpr(*C.Args[0]);
        mlir::Value B = lowerExpr(*C.Args[1]);
        if (A.getType() != PtrTy) A.setType(PtrTy);
        if (B.getType() != PtrTy) B.setType(PtrTy);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "strcmp"));
        return emitUnreg("matlab.call_builtin", {A, B}, F64, L, {Cal});
      }
      /* Phase 5.1: datetime / duration constructors. Each maps to a
       * dedicated runtime entry that returns a fresh ptr-typed
       * descriptor. Arithmetic and display dispatch live further
       * down (via DatetimeBindings / DurationBindings tags). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "datetime") {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        if (C.Args.empty()) {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_datetime_now"));
          return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
        }
        if (C.Args.size() == 1) {
          /* datetime("now") — string arg, accepted only as the literal
           * "now" for v1. Other string forms (ISO date, format) are
           * deferred. Treat any string arg as "now" for now. */
          if (isStringExpr(C.Args[0]) ||
              (C.Args[0] && C.Args[0]->Kind == NodeKind::CharLiteral)) {
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_datetime_now"));
            return emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
          }
        }
        if (C.Args.size() == 3) {
          mlir::Value Y = lowerExpr(*C.Args[0]);
          mlir::Value M = lowerExpr(*C.Args[1]);
          mlir::Value D = lowerExpr(*C.Args[2]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_datetime_ymd"));
          return emitUnreg("matlab.call_builtin", {Y, M, D},
                           PtrTy, L, {Cal});
        }
        if (C.Args.size() == 6) {
          mlir::Value Y = lowerExpr(*C.Args[0]);
          mlir::Value M = lowerExpr(*C.Args[1]);
          mlir::Value D = lowerExpr(*C.Args[2]);
          mlir::Value H = lowerExpr(*C.Args[3]);
          mlir::Value Mn = lowerExpr(*C.Args[4]);
          mlir::Value S = lowerExpr(*C.Args[5]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_datetime_ymdhms"));
          return emitUnreg("matlab.call_builtin", {Y, M, D, H, Mn, S},
                           PtrTy, L, {Cal});
        }
      }
      /* Phase 5.3: table(col1, col2, ..., 'VariableNames', {n1, n2}).
       * v1 supports auto-named (Var1..VarN) and explicit
       * 'VariableNames' tail-arg forms. Each column-arg is lowered
       * as a matrix value and added via matlab_table_add_column. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "table") {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        /* Locate an optional `'VariableNames', {...}` pair at the end. */
        size_t NCol = C.Args.size();
        const CellLiteral *NamesCell = nullptr;
        for (size_t i = 0; i + 1 < C.Args.size(); ++i) {
          const Expr *AE = C.Args[i];
          if (!AE) continue;
          if (auto *CL = dynamic_cast<const CharLiteral *>(AE)) {
            if (CL->Value == "VariableNames") {
              if (auto *NL = dynamic_cast<const CellLiteral *>(C.Args[i + 1])) {
                NamesCell = NL;
                NCol = i;
                break;
              }
            }
          }
          if (auto *SL = dynamic_cast<const StringLiteral *>(AE)) {
            if (SL->Value == "VariableNames") {
              if (auto *NL = dynamic_cast<const CellLiteral *>(C.Args[i + 1])) {
                NamesCell = NL;
                NCol = i;
                break;
              }
            }
          }
        }
        mlir::NamedAttribute NewC(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_table_new"));
        mlir::Value T = emitUnreg("matlab.call_builtin", {}, PtrTy, L, {NewC});
        for (size_t i = 0; i < NCol; ++i) {
          if (!C.Args[i]) continue;
          /* Resolve the column name. */
          std::string ColName;
          if (NamesCell && i < NamesCell->Rows[0].size()) {
            const Expr *NE = NamesCell->Rows[0][i];
            if (auto *CL = dynamic_cast<const CharLiteral *>(NE))
              ColName = std::string(CL->Value);
            else if (auto *SL = dynamic_cast<const StringLiteral *>(NE))
              ColName = std::string(SL->Value);
          }
          if (ColName.empty()) ColName = "Var" + std::to_string(i + 1);
          mlir::Value Col = lowerExpr(*C.Args[i]);
          mlir::Value NameV = emitFieldNameChar(ColName, L);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_table_add_column"));
          emitUnregOp("matlab.call_builtin", {T, NameV, Col},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
        }
        return T;
      }
      /* Phase 5.4 (cont.): timetable(col1, ..., 'VariableNames',
       * {n1,n2,...}, 'RowTimes', dt). Mirrors the table arm above
       * but folds the trailing 'RowTimes' name-value pair and emits
       * matlab_timetable_new + matlab_timetable_add_column +
       * matlab_timetable_set_row_times. The 'VariableNames' arg is
       * still optional. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "timetable") {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        size_t NCol = C.Args.size();
        const CellLiteral *NamesCell = nullptr;
        const Expr *RowTimesExpr = nullptr;
        auto matchKey = [](const Expr *AE, const char *Key) -> bool {
          if (auto *CL = dynamic_cast<const CharLiteral *>(AE))
            return CL->Value == Key;
          if (auto *SL = dynamic_cast<const StringLiteral *>(AE))
            return SL->Value == Key;
          return false;
        };
        /* Scan from the back for both name-value pairs and shrink
         * NCol so the column-arg loop stops before them. */
        for (size_t i = 0; i + 1 < C.Args.size(); ++i) {
          const Expr *AE = C.Args[i];
          if (!AE) continue;
          if (matchKey(AE, "VariableNames")) {
            if (auto *NL = dynamic_cast<const CellLiteral *>(C.Args[i + 1])) {
              NamesCell = NL;
              if (NCol > i) NCol = i;
            }
          } else if (matchKey(AE, "RowTimes")) {
            RowTimesExpr = C.Args[i + 1];
            if (NCol > i) NCol = i;
          }
        }
        mlir::NamedAttribute NewC(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_timetable_new"));
        mlir::Value TT = emitUnreg("matlab.call_builtin", {},
                                    PtrTy, L, {NewC});
        if (RowTimesExpr) {
          mlir::Value RT = lowerExpr(*RowTimesExpr);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_set_row_times"));
          emitUnregOp("matlab.call_builtin", {TT, RT},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
        }
        for (size_t i = 0; i < NCol; ++i) {
          if (!C.Args[i]) continue;
          std::string ColName;
          if (NamesCell && !NamesCell->Rows.empty() &&
              i < NamesCell->Rows[0].size()) {
            const Expr *NE = NamesCell->Rows[0][i];
            if (auto *CL = dynamic_cast<const CharLiteral *>(NE))
              ColName = std::string(CL->Value);
            else if (auto *SL = dynamic_cast<const StringLiteral *>(NE))
              ColName = std::string(SL->Value);
          }
          if (ColName.empty()) ColName = "Var" + std::to_string(i + 1);
          mlir::Value Col = lowerExpr(*C.Args[i]);
          mlir::Value NameV = emitFieldNameChar(ColName, L);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_add_column"));
          emitUnregOp("matlab.call_builtin", {TT, NameV, Col},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
        }
        return TT;
      }
      /* table2timetable(T, 'RowTimes', dt) — promote a plain table
       * to a timetable. The table is consumed. Other Name=Value
       * pairs (StartTime, SampleRate, TimeStep) are deferred. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "table2timetable" && C.Args.size() >= 1) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        const Expr *RowTimesExpr = nullptr;
        for (size_t i = 1; i + 1 < C.Args.size(); ++i) {
          const Expr *AE = C.Args[i];
          if (!AE) continue;
          if (auto *CL = dynamic_cast<const CharLiteral *>(AE))
            if (CL->Value == "RowTimes") { RowTimesExpr = C.Args[i + 1]; break; }
          if (auto *SL = dynamic_cast<const StringLiteral *>(AE))
            if (SL->Value == "RowTimes") { RowTimesExpr = C.Args[i + 1]; break; }
        }
        mlir::Value TB = lowerExpr(*C.Args[0]);
        mlir::Value RT;
        if (RowTimesExpr) {
          RT = lowerExpr(*RowTimesExpr);
        } else {
          /* No RowTimes given — produce a NULL pointer so the runtime
           * conversion still completes (timetable with empty
           * RowTimes). LLVM's ConstantPointerNull is the cheapest
           * way to get that. */
          auto NullAttr = mlir::IntegerAttr::get(
              mlir::IntegerType::get(&MCtx, 64), 0);
          (void)NullAttr;
          mlir::Value Zero = mlir::arith::ConstantOp::create(
              B, L, mlir::IntegerType::get(&MCtx, 64),
              mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 64), 0));
          RT = mlir::LLVM::IntToPtrOp::create(B, L, PtrTy, Zero).getResult();
        }
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_table2timetable"));
        return emitUnreg("matlab.call_builtin", {TB, RT}, PtrTy, L, {Cal});
      }
      /* Predicate: does this expression yield a matlab_timetable *?
       * Covers NameExpr bindings, TT-subscript CallOrIndex (TMW(...) -> TT),
       * and the TT-returning builtins (timetable, table2timetable,
       * retime, synchronize, fillmissing, movavg, macd). Used by
       * the head/summary/fillmissing/movavg/macd dispatches below so
       * `head(TMW(idx, :), 4)` style nested-subscript args work. */
      auto exprIsTimetable = [&](const Expr *E) -> bool {
        if (!E) return false;
        if (auto *NE = dynamic_cast<const NameExpr *>(E))
          return NE->Ref && isTimetableBinding(NE->Ref);
        if (auto *CX = dynamic_cast<const CallOrIndex *>(E))
          if (auto *NE = dynamic_cast<const NameExpr *>(CX->Callee)) {
            if (NE->Ref && isTimetableBinding(NE->Ref)) return true;
            if (NE->Name == "timetable" || NE->Name == "table2timetable" ||
                NE->Name == "retime" || NE->Name == "synchronize" ||
                NE->Name == "fillmissing" || NE->Name == "movavg" ||
                NE->Name == "macd")
              return true;
          }
        return false;
      };
      /* movavg(TT, 'simple'|'exponential', period) -> matlab_timetable.
       * Operates on the first numeric column of TT (the canonical
       * TMW(:, 'Close') input). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "movavg" && C.Args.size() == 3 &&
          C.Args[0] && C.Args[1] && C.Args[2]) {
        if (exprIsTimetable(C.Args[0])) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          auto I32 = mlir::IntegerType::get(&MCtx, 32);
          std::string Type;
          if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[1]))  Type = std::string(CL->Value);
          else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[1])) Type = std::string(SL->Value);
          int32_t code = 0;
          if      (Type == "simple")      code = 0;
          else if (Type == "exponential") code = 1;
          mlir::Value Tv = lowerExpr(*C.Args[0]);
          mlir::Value Tc = mlir::arith::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, code));
          mlir::Value Pv = lowerExpr(*C.Args[2]);
          /* Period arrives as f64; convert to i32. */
          mlir::Value Pi = mlir::arith::FPToSIOp::create(B, L, I32, Pv);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_movavg"));
          return emitUnreg("matlab.call_builtin", {Tv, Tc, Pi},
                           PtrTy, L, {Cal});
        }
      }
      /* macd(TT) -> 3-column matlab_timetable {MACD, Signal, Histogram}. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "macd" && C.Args.size() == 1 && C.Args[0]) {
        if (exprIsTimetable(C.Args[0])) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          mlir::Value Tv = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_macd"));
          return emitUnreg("matlab.call_builtin", {Tv}, PtrTy, L, {Cal});
        }
      }
      /* fillmissing(TT, 'linear'|'previous'|'next') -> matlab_timetable.
       * The constant-replacement form (fillmissing(TT, k) with k
       * numeric) lands later. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "fillmissing" && C.Args.size() == 2 &&
          C.Args[0] && C.Args[1]) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        if (exprIsTimetable(C.Args[0])) {
          std::string Method;
          if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[1]))
            Method = std::string(CL->Value);
          else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[1]))
            Method = std::string(SL->Value);
          int32_t code = 0;
          if      (Method == "linear")   code = 0;
          else if (Method == "previous") code = 1;
          else if (Method == "next")     code = 2;
          mlir::Value Tv = lowerExpr(*C.Args[0]);
          mlir::Value Mv = mlir::arith::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, code));
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_fillmissing"));
          return emitUnreg("matlab.call_builtin", {Tv, Mv},
                           PtrTy, L, {Cal});
        }
      }
      /* summary(TT) / head(TT[, n]) — display-only on a timetable. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "summary" && C.Args.size() == 1 && C.Args[0]) {
        if (exprIsTimetable(C.Args[0])) {
          mlir::Value Tv = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_summary"));
          return emitUnreg("matlab.call_builtin", {Tv},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "head" && C.Args.size() >= 1 && C.Args[0]) {
        if (exprIsTimetable(C.Args[0])) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::Value Tv = lowerExpr(*C.Args[0]);
          mlir::Value NV;
          if (C.Args.size() == 2 && C.Args[1]) {
            NV = lowerExpr(*C.Args[1]);
          } else {
            NV = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, 0.0));
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_head"));
          return emitUnreg("matlab.call_builtin", {Tv, NV},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }
      /* synchronize(TT1, TT2, cadence, method) -> matlab_timetable.
       * Aligns both inputs onto the same cadence with the given
       * aggregator then horz-cats. Same cadence/method codes as
       * retime. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "synchronize" && C.Args.size() == 4 &&
          C.Args[0] && C.Args[1] && C.Args[2] && C.Args[3]) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        auto strArg = [](const Expr *E, std::string &Out) -> bool {
          if (auto *CL = dynamic_cast<const CharLiteral *>(E))  { Out = std::string(CL->Value); return true; }
          if (auto *SL = dynamic_cast<const StringLiteral *>(E)){ Out = std::string(SL->Value); return true; }
          return false;
        };
        std::string Cadence, Method;
        if (strArg(C.Args[2], Cadence) && strArg(C.Args[3], Method)) {
          int32_t cad = 0, aggCode = 0;
          if      (Cadence == "daily")   cad = 0;
          else if (Cadence == "weekly")  cad = 1;
          else if (Cadence == "monthly") cad = 2;
          else if (Cadence == "yearly")  cad = 3;
          if      (Method == "firstvalue") aggCode = 0;
          else if (Method == "lastvalue")  aggCode = 1;
          else if (Method == "max")        aggCode = 2;
          else if (Method == "min")        aggCode = 3;
          else if (Method == "sum")        aggCode = 4;
          else if (Method == "mean")       aggCode = 5;
          mlir::Value T1 = lowerExpr(*C.Args[0]);
          mlir::Value T2 = lowerExpr(*C.Args[1]);
          mlir::Value CadV = mlir::arith::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, cad));
          mlir::Value AggV = mlir::arith::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, aggCode));
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_synchronize"));
          return emitUnreg("matlab.call_builtin", {T1, T2, CadV, AggV},
                           PtrTy, L, {Cal});
        }
      }
      /* retime(TT, cadence, method) -> matlab_timetable.
       * Cadence ∈ {'daily','weekly','monthly','yearly'};
       * method  ∈ {'firstvalue','lastvalue','max','min','sum','mean'}.
       * Both literal-string args fold to integer codes at lower time. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "retime" && C.Args.size() == 3 &&
          C.Args[0] && C.Args[1] && C.Args[2]) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        auto strArg = [](const Expr *E, std::string &Out) -> bool {
          if (auto *CL = dynamic_cast<const CharLiteral *>(E))  { Out = std::string(CL->Value); return true; }
          if (auto *SL = dynamic_cast<const StringLiteral *>(E)){ Out = std::string(SL->Value); return true; }
          return false;
        };
        std::string Cadence, Method;
        if (strArg(C.Args[1], Cadence) && strArg(C.Args[2], Method)) {
          int32_t cad = 0;
          if      (Cadence == "daily")   cad = 0;
          else if (Cadence == "weekly")  cad = 1;
          else if (Cadence == "monthly") cad = 2;
          else if (Cadence == "yearly")  cad = 3;
          int32_t aggCode = 0;
          if      (Method == "firstvalue") aggCode = 0;
          else if (Method == "lastvalue")  aggCode = 1;
          else if (Method == "max")        aggCode = 2;
          else if (Method == "min")        aggCode = 3;
          else if (Method == "sum")        aggCode = 4;
          else if (Method == "mean")       aggCode = 5;
          mlir::Value Tv = lowerExpr(*C.Args[0]);
          mlir::Value CadV = mlir::arith::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, cad));
          mlir::Value AggV = mlir::arith::ConstantOp::create(
              B, L, I32, mlir::IntegerAttr::get(I32, aggCode));
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_retime"));
          return emitUnreg("matlab.call_builtin", {Tv, CadV, AggV},
                           PtrTy, L, {Cal});
        }
      }
      /* timerange(t1, t2)            -> closed   (mode 0)
       * timerange(t1, t2, 'closed')  -> closed
       * timerange(t1, t2, 'openright') / 'open' / 'openleft' similarly.
       * Returns a matlab_timerange * used as a row index on TT(tr,:).
       * MATLAB's default is 'openright'; we keep 'closed' as the
       * no-mode default for the doc-page example which passes
       * 'closed' explicitly. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "timerange" &&
          (C.Args.size() == 2 || C.Args.size() == 3)) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto I32 = mlir::IntegerType::get(&MCtx, 32);
        int32_t modeCode = 0; /* closed */
        if (C.Args.size() == 3 && C.Args[2]) {
          std::string ModeStr;
          if (auto *CL = dynamic_cast<const CharLiteral *>(C.Args[2]))
            ModeStr = std::string(CL->Value);
          else if (auto *SL = dynamic_cast<const StringLiteral *>(C.Args[2]))
            ModeStr = std::string(SL->Value);
          if      (ModeStr == "closed")    modeCode = 0;
          else if (ModeStr == "openright") modeCode = 1;
          else if (ModeStr == "openleft")  modeCode = 2;
          else if (ModeStr == "open")      modeCode = 3;
        }
        mlir::Value T1 = lowerExpr(*C.Args[0]);
        mlir::Value T2 = lowerExpr(*C.Args[1]);
        mlir::Value ModeV = mlir::arith::ConstantOp::create(
            B, L, I32, mlir::IntegerAttr::get(I32, modeCode));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_timerange_new"));
        return emitUnreg("matlab.call_builtin", {T1, T2, ModeV},
                         PtrTy, L, {Cal});
      }
      /* Phase 5.2: categorical([str, str, ...]) — construct from a
       * single argument that's a 1-row MatrixLiteral of string /
       * char literals (the natural `categorical(["a","b","a"])`
       * idiom). Each element is materialised as a matlab_string *
       * via matlab_string_from_literal, packed into a stack array,
       * and passed with an i64 length to matlab_categorical_from_strs. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "categorical" && C.Args.size() == 1 && C.Args[0]) {
        auto *ML = dynamic_cast<const MatrixLiteral *>(C.Args[0]);
        if (ML && ML->Rows.size() == 1) {
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          auto F64 = mlir::Float64Type::get(&MCtx);
          /* Build each string descriptor and store into a cell, then
           * pass the cell's storage pointer + count. We piggyback on
           * matlab_cell as a temporary array of ptrs since it already
           * has set_mat semantics for ptr-typed values. The runtime
           * is given a void** + int64_t — we'll wire that via a tiny
           * dedicated bridge built inline using matlab.const_char +
           * matlab_string_from_literal + matlab_cell_set_mat into a
           * cell, then extract via matlab_cell_get_mat at runtime
           * (avoiding a new ABI). For v1 we use a simpler approach:
           * emit one matlab_categorical_from_str call per element... no.
           * Simpler still: pass the cell as a void**-ish pointer and
           * teach the runtime helper to walk it. */
          size_t N0 = ML->Rows[0].size();
          mlir::Value Cnt = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, (double)N0));
          mlir::NamedAttribute NewC(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_cell_new"));
          mlir::Value Cell = emitUnreg("matlab.call_builtin", {Cnt},
                                         PtrTy, L, {NewC});
          for (size_t i = 0; i < N0; ++i) {
            const Expr *Ei = ML->Rows[0][i];
            if (!Ei) continue;
            mlir::Value SV;
            if (auto *CL = dynamic_cast<const CharLiteral *>(Ei)) {
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
              mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                          mlir::NoneType::get(&MCtx),
                                          L, {VA});
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
              SV = emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
            } else if (auto *SL = dynamic_cast<const StringLiteral *>(Ei)) {
              mlir::NamedAttribute VA(
                  mlir::StringAttr::get(&MCtx, "value"),
                  mlir::StringAttr::get(&MCtx, std::string(SL->Value)));
              mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                          mlir::NoneType::get(&MCtx),
                                          L, {VA});
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
              SV = emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
            } else {
              SV = lowerExpr(*Ei);
            }
            mlir::Value Idx = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, (double)(i + 1)));
            mlir::NamedAttribute SCal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_cell_set_mat"));
            emitUnregOp("matlab.call_builtin", {Cell, Idx, SV},
                        {mlir::NoneType::get(&MCtx)}, L, {SCal});
          }
          /* The runtime entry takes (void **strs, int64 n). We pass
           * the cell's `.ptr_vals` field — which we expose via a thin
           * accessor. Easiest: have the runtime entry take the cell
           * and an integer count, and walk via cell_get_mat. */
          mlir::Value CntF = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, (double)N0));
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_categorical_from_cell"));
          return emitUnreg("matlab.call_builtin", {Cell, CntF},
                           PtrTy, L, {Cal});
        }
      }
      /* duration unit constructors: seconds(n), minutes(n), hours(n),
       * days(n), years(n).
       *   - scalar f64 arg → matlab_duration_<unit>      → matlab_duration *
       *   - matrix arg     → matlab_duration_<unit>_vec  → matlab_duration_vec *
       * The matrix arm covers `days(0:251)` and the natural row-times
       * recipe `datetime(2014,1,1) + days(0:251)`. The vec descriptor
       * is tagged by the caller via DurationVecBindings on the LHS so
       * downstream arithmetic / disp dispatch through the vec ABI. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "seconds" || N->Name == "minutes" ||
           N->Name == "hours"   || N->Name == "days"    ||
           N->Name == "years") &&
          C.Args.size() == 1 && C.Args[0]) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value V = lowerExpr(*C.Args[0]);
        bool IsMat = V && (V.getType() == PtrTy ||
                           mlir::isa<mlir::RankedTensorType,
                                     mlir::UnrankedTensorType>(V.getType()));
        std::string Callee = "matlab_duration_" + std::string(N->Name);
        if (IsMat) Callee += "_vec";
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        return emitUnreg("matlab.call_builtin", {V}, PtrTy, L, {Cal});
      }
      /* Phase 5.4: length / numel / size on a datetime_vec or
       * duration_vec binding. Routes to the matlab_*_vec_length /
       * _size_dim runtime entries; without this the polymorphic
       * length() would treat the descriptor pointer as a matlab_mat
       * and return 0. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "length" || N->Name == "numel") &&
          C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref &&
              (DatetimeVecBindings.count(ArgN->Ref) ||
               DurationVecBindings.count(ArgN->Ref))) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            const char *Callee = DatetimeVecBindings.count(ArgN->Ref)
                ? "matlab_datetime_vec_length"
                : "matlab_duration_vec_length";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            return emitUnreg("matlab.call_builtin", {V}, F64, L, {Cal});
          }
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "size" && C.Args.size() == 2) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && DatetimeVecBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value V = lowerExpr(*C.Args[0]);
            mlir::Value D = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_datetime_vec_size_dim"));
            return emitUnreg("matlab.call_builtin", {V, D}, F64, L, {Cal});
          }
      }
      /* Phase 4: dictionary() / dictionary(k1, v1, k2, v2, ...) ->
       * matlab_dict_new + per-pair set. v1 supports zero-arg and an
       * even number of trailing key/value pairs; the constructor
       * mirrors containers.Map(). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "dictionary") {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto coerceKey = [&](const Expr *KeyExpr,
                             bool &KeyIsStr) -> mlir::Value {
          if (auto *CL = dynamic_cast<const CharLiteral *>(KeyExpr)) {
            mlir::NamedAttribute VA(
                mlir::StringAttr::get(&MCtx, "value"),
                mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
            mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                        mlir::NoneType::get(&MCtx), L, {VA});
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
            KeyIsStr = true;
            return emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
          }
          mlir::Value Kv = lowerExpr(*KeyExpr);
          KeyIsStr = Kv && (Kv.getType() == PtrTy || isStringExpr(KeyExpr));
          return Kv;
        };
        mlir::NamedAttribute NewCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_dict_new"));
        mlir::Value D = emitUnreg("matlab.call_builtin", {},
                                   PtrTy, L, {NewCal});
        if (C.Args.size() % 2 == 0) {
          for (size_t i = 0; i + 1 < C.Args.size(); i += 2) {
            if (!C.Args[i] || !C.Args[i+1]) continue;
            bool KeyIsStr = false;
            mlir::Value K = coerceKey(C.Args[i], KeyIsStr);
            mlir::Value V = lowerExpr(*C.Args[i+1]);
            bool ValIsMat = V && (V.getType() == PtrTy ||
                                  mlir::isa<mlir::RankedTensorType,
                                            mlir::UnrankedTensorType>(V.getType()));
            std::string Callee = "matlab_dict_set_";
            Callee += KeyIsStr ? "str_" : "num_";
            Callee += ValIsMat ? "mat" : "f64";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            emitUnregOp("matlab.call_builtin", {D, K, V},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
          }
        }
        return D;
      }
      /* iscell(x): compile-time fold. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "iscell" && C.Args.size() == 1) {
        auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]);
        auto F64 = mlir::Float64Type::get(&MCtx);
        double Val = 0.0;
        if (ArgN && ArgN->Ref && CellBindings.count(ArgN->Ref)) Val = 1.0;
        return mlir::arith::ConstantOp::create(
            B, L, F64, mlir::FloatAttr::get(F64, Val));
      }
      /* numel(C) / length(C) on a known cell -> matlab_cell_numel. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "numel" || N->Name == "length") &&
          C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && CellBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value Arg = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_cell_numel"));
            return emitUnreg("matlab.call_builtin", {Arg}, F64, L, {Cal});
          }
      }
      /* numel(s) / length(s) where s is a runtime char/string value produced
       * inline by a string-returning builtin (blanks, strcat, strtrim, upper,
       * regexprep, ...). Without this the matlab_string* is read as a
       * matlab_mat by matlab_length / matlab_numel and the descriptor's bytes
       * print as a garbage double (#234 — the "minimum safe step": kill the
       * UB). Routes to matlab_string_len -> the char count, matching MATLAB
       * (length('     ') == 5). The string-scalar BINDING case
       * (s = "hi"; length(s) -> 1) is intentionally left unchanged. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "numel" || N->Name == "length") &&
          C.Args.size() == 1 &&
          dynamic_cast<const CallOrIndex *>(C.Args[0]) &&
          isStringExpr(C.Args[0])) {
        auto F64 = mlir::Float64Type::get(&MCtx);
        mlir::Value Arg = lowerExpr(*C.Args[0]);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_string_len"));
        return emitUnreg("matlab.call_builtin", {Arg}, F64, L, {Cal});
      }
      /* numel(s) / length(s) on a scalar struct -> 1. Without this the
       * struct ptr is read as a matlab_mat and numel returns garbage
       * (rows*cols off the struct descriptor). Struct ARRAYS are excluded
       * here (they have their own length handling). */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "numel" || N->Name == "length") &&
          C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref &&
              (ArgN->Ref->IsStruct || StructInitialised.count(ArgN->Ref) ||
               StructBindings.count(ArgN->Ref)) &&
              !ArgN->Ref->IsStructArray &&
              !StructArrayBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::NamedAttribute VA(
                mlir::StringAttr::get(&MCtx, "value"),
                mlir::FloatAttr::get(F64, 1.0));
            return emitUnreg("matlab.const_float", {}, F64, L, {VA});
          }
      }
      /* Phase 4: numel(d) / length(d) on a dict binding ->
       * matlab_dict_length. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "numel" || N->Name == "length") &&
          C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && DictBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value D = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_dict_length"));
            return emitUnreg("matlab.call_builtin", {D}, F64, L, {Cal});
          }
      }
      /* isKey(d, k) -> matlab_dict_has_<str|num>.
       * remove(d, k) -> matlab_dict_remove_<str|num>.
       * CharLiteral keys coerce to matlab_string* via from_literal. */
      auto dictBuiltin2 = [&](llvm::StringRef Op) -> mlir::Value {
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value D = lowerExpr(*C.Args[0]);
        const Expr *KeyExpr = C.Args[1];
        mlir::Value K;
        bool KeyIsStr = false;
        if (auto *CL = dynamic_cast<const CharLiteral *>(KeyExpr)) {
          mlir::NamedAttribute VA(
              mlir::StringAttr::get(&MCtx, "value"),
              mlir::StringAttr::get(&MCtx, std::string(CL->Value)));
          mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                      mlir::NoneType::get(&MCtx), L, {VA});
          mlir::NamedAttribute SCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
          K = emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {SCal});
          KeyIsStr = true;
        } else {
          K = lowerExpr(*KeyExpr);
          KeyIsStr = K && (K.getType() == PtrTy || isStringExpr(KeyExpr));
        }
        std::string Callee = "matlab_dict_";
        Callee += std::string(Op) + "_";
        Callee += KeyIsStr ? "str" : "num";
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        return emitUnreg("matlab.call_builtin", {D, K}, F64, L, {Cal});
      };
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "isKey" && C.Args.size() == 2)
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && DictBindings.count(ArgN->Ref))
            return dictBuiltin2("has");
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "remove" && C.Args.size() == 2)
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && DictBindings.count(ArgN->Ref))
            return dictBuiltin2("remove");
      /* Phase 2: numel(S) / length(S) on a struct-array binding ->
       * matlab_struct_arr_length. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          (N->Name == "numel" || N->Name == "length") &&
          C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && isStructArrayBinding(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value Slot = ensureStructArraySlot(ArgN->Ref,
                                                     ArgN->Name, L);
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value Arr = emitLoad(Slot, PtrTy, L);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_struct_arr_length"));
            return emitUnreg("matlab.call_builtin", {Arr}, F64, L, {Cal});
          }
      }
      /* size(S, dim) on a struct-array binding -> matlab_struct_arr_size_dim. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "size" && C.Args.size() == 2) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && isStructArrayBinding(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value Slot = ensureStructArraySlot(ArgN->Ref,
                                                     ArgN->Name, L);
            mlir::Value Arr = emitLoad(Slot, PtrTy, L);
            mlir::Value D = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_struct_arr_size_dim"));
            return emitUnreg("matlab.call_builtin", {Arr, D},
                             F64, L, {Cal});
          }
      }
      /* Phase 1.3: size(C, dim) on a known cell -> matlab_cell_size_dim.
       * Without this, size would route through the matrix runtime and
       * read garbage from the cell layout. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "size" && C.Args.size() == 2) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && CellBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value A = lowerExpr(*C.Args[0]);
            mlir::Value D = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_cell_size_dim"));
            return emitUnreg("matlab.call_builtin", {A, D}, F64, L, {Cal});
          }
      }
      /* size(A, dim) / numel(A) / ndims(A) on a 3-D binding route to
       * the matlab_mat3 runtime; the 2-D variants treat the descriptor
       * as a matlab_mat* and would read wrong fields. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "size" && C.Args.size() == 2) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (isThreeDBinding(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value A = lowerExpr(*C.Args[0]);
            mlir::Value D = lowerExpr(*C.Args[1]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_size3_dim"));
            return emitUnreg("matlab.call_builtin", {A, D}, F64, L, {Cal});
          }
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "numel" && C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (isThreeDBinding(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value A = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_numel3"));
            return emitUnreg("matlab.call_builtin", {A}, F64, L, {Cal});
          }
      }
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "ndims" && C.Args.size() == 1) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (isThreeDBinding(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value A = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_ndims3"));
            return emitUnreg("matlab.call_builtin", {A}, F64, L, {Cal});
          }
      }
      llvm::SmallVector<mlir::Value, 4> Args;
      /* sprintf / fopen take string parameters; a single-quote char-array
       * literal lowers to a `matlab.const_char` tensor (not a matlab_string*),
       * which the sprintf/fopen lowering rejects.  Wrap CharLiteral args in
       * matlab_string_from_literal — the same shape a double-quote "..."
       * string produces — so `sprintf('%d', x)` / `fopen('p', 'w')` work like
       * their double-quote equivalents.  (A char-array *variable* format is a
       * separate, deeper gap and is not handled here.) */
      bool WrapCharArgs = (N->Name == "sprintf" || N->Name == "fopen");
      for (const Expr *A : C.Args) {
        if (!A) continue;
        if (WrapCharArgs && A->Kind == NodeKind::CharLiteral) {
          auto &S = static_cast<const CharLiteral &>(*A);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          mlir::NamedAttribute VA(mlir::StringAttr::get(&MCtx, "value"),
                                  mlir::StringAttr::get(&MCtx, S.Value));
          mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                     mlir::NoneType::get(&MCtx), L, {VA});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
          Args.push_back(emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal}));
        } else {
          Args.push_back(lowerExpr(*A));
        }
      }
      /* Variadic callee: if the user function's last declared input is
       * named "varargin", pack trailing args into a matlab_cell and
       * pass it as the last argument. The leading declared-1 args are
       * passed positionally. A call with only declared-1 args still
       * packs an empty cell so the callee's signature stays uniform. */
      unsigned OrigArity = (unsigned)Args.size();
      bool Packed = false;
      if (N && N->Ref && N->Ref->Kind == BindingKind::Function &&
          N->Ref->FuncDef && !N->Ref->FuncDef->Inputs.empty() &&
          N->Ref->FuncDef->Inputs.back() == "varargin") {
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        unsigned DeclIn = (unsigned)N->Ref->FuncDef->Inputs.size();
        unsigned Fixed = DeclIn - 1;
        if (Args.size() >= Fixed) {
          Packed = true;
          /* Build the cell out of the trailing overflow args. */
          unsigned ExtraN = (unsigned)Args.size() - Fixed;
          mlir::Value Cnt = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, (double)ExtraN));
          mlir::NamedAttribute New(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_cell_new"));
          mlir::Value Cell = emitUnreg("matlab.call_builtin", {Cnt},
                                        PtrTy, L, {New});
          for (unsigned i = 0; i < ExtraN; ++i) {
            mlir::Value Idx = mlir::arith::ConstantOp::create(
                B, L, F64, mlir::FloatAttr::get(F64, (double)(i + 1)));
            mlir::Value V = Args[Fixed + i];
            bool IsMat = V && (V.getType() == PtrTy ||
                               mlir::isa<mlir::RankedTensorType,
                                         mlir::UnrankedTensorType>(V.getType()));
            llvm::StringRef Callee = IsMat ? "matlab_cell_set_mat"
                                            : "matlab_cell_set_f64";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            emitUnregOp("matlab.call_builtin", {Cell, Idx, V},
                        {mlir::NoneType::get(&MCtx)}, L, {Cal});
          }
          Args.resize(Fixed);
          Args.push_back(Cell);
        }
      }
      if (N && N->Ref) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, std::string(N->Name)));
        llvm::SmallVector<mlir::NamedAttribute, 2> AllAttrs = {Cal};
        /* For fprintf/sprintf, tag which arguments are string-typed (a
         * bitmask, operand-index keyed) so LowerIO — which runs after the
         * Sema types are gone — can route `%s` operands as strings and box
         * everything else as numeric matrices.  Without this a string arg is
         * forced through the numeric path and SIGSEGVs. */
        if (N->Name == "fprintf" || N->Name == "sprintf") {
          int64_t StrMask = 0;
          for (size_t i = 0; i < C.Args.size() && i < 63; ++i) {
            const Expr *E = C.Args[i];
            if (!E) continue;
            bool isStr = false;
            if (auto *AN = dynamic_cast<const NameExpr *>(E))
              if (AN->Ref && StringBindings.count(AN->Ref)) isStr = true;
            if (isStringExpr(E)) isStr = true;
            if (const Type *T = E->Ty) {
              if (T->K == Type::Kind::StringArray) isStr = true;
              else if (T->K == Type::Kind::Array &&
                       static_cast<const ArrayType *>(T)->Elt == Dtype::Char)
                isStr = true;
            }
            if (isStr) StrMask |= (int64_t(1) << i);
          }
          if (StrMask)
            AllAttrs.push_back(mlir::NamedAttribute(
                mlir::StringAttr::get(&MCtx, "str_mask"),
                mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 64),
                                       StrMask)));
        }
        /* Record the original (pre-packing) call-site arity so the
         * monomorphiser can bucket by user-visible arity for
         * varargin-packed callees; otherwise nargin inside the body
         * would always equal declared-arity regardless of how many
         * args the user actually passed. */
        if (Packed) {
          AllAttrs.push_back(mlir::NamedAttribute(
              mlir::StringAttr::get(&MCtx, "user_arity"),
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(&MCtx, 64),
                  (int64_t)OrigArity)));
        }
        /* Refine the call's MLIR result type for known builtins when
         * Sema left it open. The ExprStmt implicit-display path checks
         * V.getType() at lowerExpr-return time and skips NoneType, so
         * bare `det(A)` at the REPL used to silently drop its output.
         * Split into "always f64" vs. "always ptr" based on the runtime
         * signature — matches the LowerTensorOps dispatch table. */
        mlir::Type ResTy = RT;
        if (N->Ref->Kind == BindingKind::Builtin &&
            mlir::isa<mlir::NoneType>(ResTy)) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          static const llvm::StringSet<> F64Ret = {
            "det", "norm", "trace", "length", "numel", "ndims",
            "isempty", "isequal", "rank", "cond", "sub2ind", "mod", "rem",
            "fix", "round", "floor", "ceil",
            /* SPT order helpers — return scalar f64 in single-result
             * form. Multi-LHS `[n, Wn] = buttord(...)` splits in the
             * dedicated multi-return dispatch in LowerTensorOps. */
            "buttord", "cheb1ord", "cheb2ord",
            /* Tier-3 §4.3 scalar reductions. */
            "rms", "peak2peak", "peak2rms", "rssq",
            "risetime", "falltime", "dutycycle",
            /* §4.3 pulse-statistics tail. */
            "slewrate", "pulseperiod", "pulsewidth",
            "overshoot", "undershoot", "settlingtime",
            /* Tier-3 §4.4 scalar reductions. */
            "finddelay", "dtw",
          };
          static const llvm::StringSet<> PtrRet = {
            "zeros", "ones", "eye", "magic", "rand", "randn",
            "sum", "prod", "mean", "min", "max", "cumsum", "cumprod",
            "sort", "sortrows", "unique", "ismember", "setdiff",
            "intersect", "union", "horzcat", "vertcat", "kron",
            "chol", "pinv", "permute", "ipermute", "squeeze", "flip", "fliplr",
            "flipud", "rot90", "size", "transpose", "ctranspose",
            "diag", "reshape", "repmat", "inv", "svd", "eig", "expm", "logm", "hess",
            "schur", "qz", "lyap", "dlyap", "lyapchol",
            "care", "dare", "icare", "idare", "lqr", "dlqr",
            "ctrb", "obsv", "place", "damp", "hsvd", "balreal_T",
            "balred", "balred_A", "balred_B", "balred_C", "dcgain_ss",
            "stepinfo",
            "kalman", "kalmd", "kalman_L", "kalmd_L",
            "c2d", "c2d_tustin", "d2c", "d2c_tustin", "pole",
            "feedback_ss", "series_ss", "parallel_ss", "append_ss",
            "gram_c", "gram_o", "step_ss", "bode_ss", "lsim_ss", "bode_tf",
            /* §3.1 — model-object short forms (value-returning). */
            "step", "bode", "dcgain", "lsim", "bandwidth",
            /* Tier-2 follow-on builtins (matrix-arg + model-object
             * dispatch in CallOrIndex). */
            "impulse", "initial", "freqresp", "nyquist", "allmargin",
            "impulse_ss", "initial_ss",
            "freqresp_ss", "freqresp_tf",
            "nyquist_ss", "nyquist_tf", "allmargin_ss",
            /* Tier-3 follow-on builtins (acker = place; gram / norm
             * short forms with char/scalar second arg; lqry =
             * output-weighted LQR). */
            "acker", "gram", "norm", "lqry",
            /* Tier-4 follow-on builtins. pade / minreal are
             * multi-return splitters; hsvd / balreal_T model-object
             * short forms route through CallOrIndex dispatch. */
            "pade", "minreal",
            /* Class-returning model-object short forms (Sema's
             * pinnedOfRhs handles the LHS slot pin). */
            "feedback", "series", "parallel", "append", "blkdiag",
            /* Tier-4 reduction + delay tail. */
            "sminreal", "modred", "thiran",
            "find", "ind2sub", "linspace", "logspace",
            /* Complex: all return a matrix descriptor (matlab_mat* or
             * matlab_mat_c*), uniformly ptr at MLIR level. */
            "conj", "real", "imag", "angle", "complex",
            "fft", "ifft", "fft2", "ifft2",
            "conv", "conv2",
            "filter", "any", "all", "tril", "triu",
            "fftshift", "ifftshift",
            "std", "var", "median", "diff",
            "meshgrid", "ndgrid", "peaks",
            "xcorr", "polyval", "polyfit", "roots", "poly",
            "polyder", "polyint", "residue",
            /* Tier-1 §2.1 — IIR lowpass design + frequency response. */
            "butter", "cheby1", "cheby2", "freqz",
            /* §2.1 follow-on — standalone bilinear + analog freqs. */
            "bilinear", "freqs", "tf2zp", "zp2tf", "besself",
            "tf2sos", "sos2tf",
            /* Note: buttord, cheb1ord return scalar f64 (multi-LHS form
             * splits into n -> f64 and Wn -> f64) — they're not in
             * PtrRet. Sema's default-typing does not type them; the
             * F64Ret list below covers them. */
            /* Tier-1 §2.2 — FIR design + Savitzky-Golay. */
            "fir1", "sgolay", "sgolayfilt",
            /* Tier-1 §2.5 — close-the-loop filter helpers. */
            "filtfilt", "sosfilt", "impz", "stepz", "grpdelay",
            /* Tier-2 §3.4 — transforms tail. */
            "dct", "idct", "fwht", "hilbert", "goertzel",
            /* Tier-2 §3.1 — nonparametric spectral estimation. */
            "periodogram", "pwelch",
            /* Tier-2 §3.3 — time-frequency. */
            "spectrogram",
            /* Tier-2 §3.2 — linear prediction + parametric PSD. */
            "levinson", "lpc", "aryule", "arburg", "pyulear", "pburg",
            /* Tier-2 §3.1 cross-spectral helpers. */
            "cpsd", "mscohere", "tfestimate",
            /* Tier-3 §4.3 — findpeaks (single-LHS = peaks-only). The
             * scalar reductions rms/peak2peak/peak2rms/rssq are in
             * F64Ret below. */
            "findpeaks",
            "medfilt1", "hampel", "envelope", "midcross",
            "statelevels",
            /* Tier-3 §4.1 multirate. */
            "upfirdn", "decimate", "interp", "resample",
            /* Tier-3 §4.2 waveform generators. */
            "chirp", "sawtooth", "square", "gauspuls",
            "rectpuls", "tripuls", "sinc",
            /* Tier-3 §4.4 — xcov returns a matrix; finddelay/dtw scalars. */
            "xcov",
            "interp1", "trapz", "cumtrapz", "gradient",
            "hamming", "hann", "blackman",
            /* Tier-1 windows tail (signal_toolbox_roadmap §2.3) — all
             * return a column vector descriptor. */
            "rectwin", "triang", "bartlett", "barthannwin", "bohmanwin",
            "parzenwin", "nuttallwin", "blackmanharris", "flattopwin",
            "kaiser", "tukeywin", "gausswin", "chebwin", "taylorwin",
            /* Tier-3: linalg helpers + image-processing wrappers + interp2.
             * rank/cond return f64 (not ptr), so they live in F64Ret above. */
            "null", "orth", "imfilter", "padarray",
            "interp2", "upsample", "downsample",
            /* Plotting — figure / gcf return an opaque matlab_figure*
             * descriptor; everything else in the family returns void. */
            "figure", "gcf",
            /* Animation — getframe returns an opaque matlab_frame*,
             * VideoWriter returns an opaque matlab_videowriter*. */
            "getframe", "VideoWriter",
          };
          if (F64Ret.contains(N->Name)) ResTy = F64;
          else if (PtrRet.contains(N->Name)) ResTy = PtrTy;
        }
        /* #286: the same NoneType-result gap affects user functions. Sema
         * commonly leaves a user function's call result open (E.Ty == any)
         * — especially at the REPL, where the callee was defined in an
         * earlier input so the consuming statement can't see its inferred
         * output type. That left a bare `f(40)` lowering to a NoneType
         * matlab.call, so the ExprStmt implicit-display path skipped it and
         * no `ans = 42` was printed (builtins already dodged this via the
         * block above).
         *
         * Scope this strictly to the bare, non-suppressed top-level call of
         * an ExprStmt (BareDisplayCall): that's the only site that needs a
         * concrete type for the implicit display, which is lowered by an
         * early LowerTensorOps pass that runs before the late call-result
         * reconciliation. Refining every call site instead over-types
         * model-object methods (`ys = step(G, ts)`, library-classdef
         * methods not in CurTU->Classes), leaving unconverted
         * matlab.store/alloc behind them. */
        if (N->Ref->Kind == BindingKind::Function && N->Ref->FuncDef &&
            BareDisplayCall == static_cast<const void *>(&C) &&
            mlir::isa<mlir::NoneType>(ResTy) &&
            !N->Ref->FuncDef->OutputRefs.empty()) {
          /* Prefer Sema's concrete inferred output type (covers a
           * matrix-returning `mk() = [1 2; 3 4]`, typed → ptr); fall back
           * to f64 for an output whose type depends on the arg (the common
           * `f(x) = x + 2` interactive case). */
          mlir::Type RT0;
          if (N->Ref->FuncDef->OutputRefs[0] &&
              N->Ref->FuncDef->OutputRefs[0]->InferredType)
            RT0 = mirTy(N->Ref->FuncDef->OutputRefs[0]->InferredType);
          ResTy = (RT0 && !mlir::isa<mlir::NoneType>(RT0))
                      ? RT0
                      : (mlir::Type)mlir::Float64Type::get(&MCtx);
        }
        return emitUnreg(N->Ref->Kind == BindingKind::Builtin
                              ? "matlab.call_builtin" : "matlab.call",
                          Args, ResTy, L, AllAttrs);
      }
      mlir::Value CV = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
      llvm::SmallVector<mlir::Value, 4> Os;
      Os.push_back(CV);
      for (auto V : Args) Os.push_back(V);
      return emitUnreg("matlab.call_indirect", Os, RT, L);
    }
    // Index
    /* #81: a function handle stored in a struct field / classdef property
     * (`s.h = @inc; v = s.h(5)` or `obj.StepFcn = @step; obj.StepFcn(a)`).
     * The handle's target was resolved at the store and recorded in
     * FieldHandleBindings; emit a direct `matlab.call @<name>(args)` —
     * the same shape a syntactic `inc(5)` lowers to — instead of leaving
     * an unconverted matlab.subscript on the field-loaded handle value. */
    if (auto *F = dynamic_cast<const FieldAccess *>(C.Callee))
      if (auto *BN = dynamic_cast<const NameExpr *>(F->Base))
        if (BN->Ref) {
          auto It = FieldHandleBindings.find({BN->Ref, std::string(F->Field)});
          if (It != FieldHandleBindings.end() && It->second) {
            llvm::SmallVector<mlir::Value, 4> CArgs;
            for (const Expr *A : C.Args) if (A) CArgs.push_back(lowerExpr(*A));
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, std::string(It->second->Name)));
            return emitUnreg("matlab.call", CArgs, RT, L, {Cal});
          }
        }
    /* REPL/DAP workspace-backed function handle: `f(args)` where `f`
     * round-trips through the workspace as a bare function pointer
     * (kind=13).  In ReplMode a handle variable isn't a local slot — it
     * is stored/loaded via matlab_ws_set/get_handle — so the in-memory
     * call_indirect chain (which traces back to a make_handle/addressof)
     * is broken by the store→load round-trip.  Instead, load the stored
     * pointer and invoke it through the matlab_call_handle_s* trampoline,
     * which takes the function pointer as its first argument.  This is
     * what stops `f(0)` from mis-lowering into matlab_subscript1_s on the
     * code pointer (the SIGSEGV in issue #77).
     *
     * Scope: all-f64-scalar arguments, arity 1..3 (the common math /
     * user-function handle shape).  Captured anons and matrix-argument
     * handles fall through to the existing paths. */
    if (ReplMode && InScriptBody) {
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        if (NE->Ref && NE->Ref->Kind == BindingKind::Var &&
            Slots.find(NE->Ref) == Slots.end()) {
          /* Named handle to a user function visible in this module
           * (`p = @mySq; p(6)` — same turn or whole-program DAP launch).
           * Emit a direct `matlab.call @mySq(args)` so the user-call
           * refinement / RefineFuncSigs gives the callee a concrete f64
           * signature and lowers its body — an address-only reference
           * (@mySq stored in the workspace) never produces a call site,
           * leaving the body's matmul/const_char unconverted. */
          auto HTIt = HandleTargetRef.find(NE->Ref);
          if (HTIt != HandleTargetRef.end() && HTIt->second &&
              HTIt->second->FuncDef) {
            llvm::SmallVector<mlir::Value, 4> CArgs;
            for (const Expr *Arg : C.Args)
              if (Arg) CArgs.push_back(lowerExpr(*Arg));
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, std::string(HTIt->second->Name)));
            return emitUnreg("matlab.call", CArgs, RT, L, {Cal});
          }
        }
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        if (NE->Ref && NE->Ref->Kind == BindingKind::Var &&
            Slots.find(NE->Ref) == Slots.end() &&
            C.Args.size() <= 3) {
          bool WsHandle = NE->Ref->IsHandle;
          if (!WsHandle) {
            auto HIt = HandleBindings.find(NE->Ref);
            if (HIt != HandleBindings.end() && HIt->second.empty())
              WsHandle = true;
          }
          /* Require every argument to be a scalar double in Sema's view
           * so we can commit to the trampoline before lowering (avoids
           * double-lowering args with side effects). */
          auto isScalarDoubleTy = [](const Type *T) {
            if (!T || T->K != Type::Kind::Array) return false;
            auto &A = static_cast<const ArrayType &>(*T);
            return A.Elt == Dtype::Double && A.S.K == Shape::Rank::Scalar;
          };
          bool ArgsScalar = WsHandle;
          for (const Expr *Arg : C.Args)
            if (!Arg || !isScalarDoubleTy(Arg->Ty)) { ArgsScalar = false; break; }
          if (WsHandle && ArgsScalar) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value Fn = lowerExpr(*C.Callee);   /* matlab_ws_get_handle */
            llvm::SmallVector<mlir::Value, 4> CallArgs;
            CallArgs.push_back(Fn);
            bool AllF64 = true;
            for (const Expr *Arg : C.Args) {
              mlir::Value AV = lowerExpr(*Arg);
              if (AV.getType() != F64) { AllF64 = false; break; }
              CallArgs.push_back(AV);
            }
            /* Only commit if every lowered argument is a native f64 (the
             * trampoline ABI).  If not, the ops just emitted are dead and
             * get DCE'd; fall through to the existing handle/subscript
             * path below. */
            if (AllF64) {
              const char *Tramp = C.Args.empty() ? "matlab_call_handle_s0"
                                : (C.Args.size() == 1 ? "matlab_call_handle_s1"
                                : (C.Args.size() == 2 ? "matlab_call_handle_s2"
                                                      : "matlab_call_handle_s3"));
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Tramp));
              return emitUnreg("matlab.call_builtin", CallArgs, F64, L, {Cal});
            }
          }
          /* #119: MATRIX-argument cross-turn handle call.  When the args
           * aren't all scalar doubles and the recovered handle carries a
           * known return-kind (HandleRetKind, stamped by the Resolver from
           * the kind=13 signature side-channel), dispatch to the matrix
           * trampoline with the matching result type — scalar return
           * (matlab_call_handle_m{1,2}) or matrix return (..._mm{1,2}).
           * Without this the call falls through to a subscript on the code
           * pointer and SIGSEGVs.  Arity 1..2 (the objective / residual
           * shapes); commit only when every arg lowers to a matlab_mat*
           * pointer. */
          if (WsHandle && !ArgsScalar && NE->Ref->HandleRetKind >= 0 &&
              !C.Args.empty() && C.Args.size() <= 2) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value Fn = lowerExpr(*C.Callee);   /* matlab_ws_get_handle */
            llvm::SmallVector<mlir::Value, 4> CallArgs;
            CallArgs.push_back(Fn);
            bool AllPtr = (Fn.getType() == PtrTy);
            for (const Expr *Arg : C.Args) {
              mlir::Value AV = lowerExpr(*Arg);
              if (AV.getType() != PtrTy) { AllPtr = false; break; }
              CallArgs.push_back(AV);
            }
            if (AllPtr) {
              bool Scalar = (NE->Ref->HandleRetKind == 0);
              const char *Tramp =
                  Scalar ? (C.Args.size() == 1 ? "matlab_call_handle_m1"
                                               : "matlab_call_handle_m2")
                         : (C.Args.size() == 1 ? "matlab_call_handle_mm1"
                                               : "matlab_call_handle_mm2");
              mlir::Type ResTy = Scalar ? (mlir::Type)mlir::Float64Type::get(&MCtx)
                                        : (mlir::Type)PtrTy;
              mlir::NamedAttribute Cal(
                  mlir::StringAttr::get(&MCtx, "callee"),
                  mlir::StringAttr::get(&MCtx, Tramp));
              return emitUnreg("matlab.call_builtin", CallArgs, ResTy, L, {Cal});
            }
          }
        }
    }
    // Detect the "call through a handle" case: if the callee is a NameExpr
    // whose binding was assigned from @(x)... / @name, emit a
    // matlab.call_indirect instead of a matlab.subscript.
    bool IsHandleCall = false;
    const std::vector<mlir::Value> *CapSlots = nullptr;
    if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
      if (NE->Ref) {
        auto It = HandleBindings.find(NE->Ref);
        if (It != HandleBindings.end()) {
          IsHandleCall = true;
          CapSlots = &It->second;
        }
      }

    /* System-Object callable-instance sugar: `obj(args)` where `obj` is
     * class-pinned to a class with a `step` method dispatches to
     * `step(obj, args)`.  MATLAB's `comm.*` / `dsp.*` / `phased.*`
     * System Object surface is built on this idiom — the user writes
     * `out = sys(in)` and the runtime routes to `step(sys, in)`.
     *
     * Detected only when:
     *   - the callee is a bare NameExpr (rules out chained calls);
     *   - its binding pins to a ClassDef that defines `step` directly
     *     or through its Super chain;
     *   - we're not already routing this as a handle-fn call (a
     *     function-handle binding wins because the user is explicitly
     *     invoking a stored handle).
     *
     * When detected, emit a direct `matlab.call @ClassName__step(obj,
     * args)` and short-circuit the subscript path — mirrors how
     * `obj.step(args)` already lowers, just without the dot syntax. */
    if (!IsHandleCall) {
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        if (NE->Ref && NE->Ref->PinnedClass) {
          const ClassDef *Owner = nullptr;
          /* `step` is the System-Object idiom; `feval` is the Curve-Fitting
           * idiom (`f(xq)` on a `cfit` evaluates the model). Whichever the
           * pinned class defines becomes the paren-call target. */
          const char *MethName = nullptr;
          for (const ClassDef *CC = NE->Ref->PinnedClass; CC; CC = CC->Super) {
            for (const Function *Mth : CC->Methods)
              if (Mth && (Mth->Name == "step" || Mth->Name == "feval")) {
                Owner = CC;
                MethName = (Mth->Name == "feval") ? "feval" : "step";
                break;
              }
            if (Owner) break;
          }
          if (Owner) {
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::Value Recv = lowerExpr(*C.Callee);
            llvm::SmallVector<mlir::Value, 4> Args;
            Args.push_back(Recv);
            for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
            std::string Callee = std::string(Owner->Name) + "__" + MethName;
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Callee));
            /* `step` typically returns a value — Sema may or may not
             * have typed the call.  Pass the RT through unchanged when
             * it's already concrete; otherwise default to ptr (which
             * the runtime auto-boxes for scalar f64 returns at the
             * use site, matching how class-method dispatch above
             * treats `obj.step(...)` returns). */
            mlir::Type ResTy = mlir::isa<mlir::NoneType>(RT) ?
                                 (mlir::Type)PtrTy : RT;
            return emitUnreg("matlab.call", Args, ResTy, L, {Cal});
          }
        }
    }

    mlir::Value Arr = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
    /* Bit-slice extension: `x(hi:lo)` on a scalar integer with a
     * constant descending range. Sema annotated the result type as
     * a uint scalar of the rounded-up slice width. We emit a
     * matlab.call_builtin @bitslice with hi/lo as i64 attrs; the
     * LowerScalarsToArith pass converts it to arith.shrui /
     * arith.trunci / arith.andi when the operand has a typed scalar
     * int (which the HW pipeline anchors via the snapshot pattern).
     *
     * Detected BEFORE lowering args, since lowering a RangeExpr emits
     * a `matlab_range(...)` runtime call that the HW pipeline rejects.
     */
    llvm::SmallVector<mlir::Value, 4> Idx;
    Idx.push_back(Arr);
    /* For an anon call with captures, the outlined function's signature
     * is (captures..., explicit_args...). We load each capture spill
     * slot (captured-at-@-time value) and prepend them to the arg list
     * before the user-written arguments. The slot's own type gives the
     * load type — handles both scalar (f64) and matrix-pointer captures
     * (matlab.alloc of a tensor type lowers to a ptr-typed slot later). */
    if (IsHandleCall && CapSlots) {
      for (mlir::Value Spill : *CapSlots)
        Idx.push_back(emitLoad(Spill, Spill.getType(), L));
    }
    // Lower each arg with subscript context pushed so any EndExpr inside
    // resolves to size(Arr, thisDim). Context is per-arg so that sibling
    // args don't leak each other's dim.
    for (size_t a = 0; a < C.Args.size(); ++a) {
      const Expr *Arg = C.Args[a];
      if (!Arg) continue;
      // Single-subscript indexing: `end` means numel(Arr), not size(,1).
      // Use sentinel dim 0 → matlab_end_of_dim treats it as numel.
      int64_t EndDim = (C.Args.size() == 1) ? 0 : (int64_t)(a + 1);
      if (!IsHandleCall)
        SubscriptCtx.push_back({Arr, EndDim});
      Idx.push_back(lowerExpr(*Arg));
      if (!IsHandleCall) SubscriptCtx.pop_back();
    }
    if (IsHandleCall)
      return emitUnreg("matlab.call_indirect", Idx, RT, L);
    /* fi-array subscript: A(i) / A(i,j) / A(idx_vec) on a fi-typed base
     * routes directly to the typed-int matrix subscript helpers. We
     * detect the fi base via the callee expression's Sema type. */
    {
      const Type *BaseT = C.Callee ? C.Callee->Ty : nullptr;
      if (BaseT && BaseT->K == Type::Kind::Array) {
        auto &BA = static_cast<const ArrayType &>(*BaseT);
        if (BA.Elt == Dtype::Fixed && BA.FxSpec &&
            BA.S.K != Shape::Rank::Scalar &&
            !C.Args.empty()) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
          auto I64 = mlir::IntegerType::get(&MCtx, 64);
          // Idx[0] is the base (already lowered above).
          // Per-element subscript: every index is an f64 scalar — or
          // `none`-typed, which the HDL flow produces for `arg + 1`
          // where `arg` is pragma-typed (the matlab.add result type
          // doesn't propagate at lowering time and gets refined by
          // RefineSlotTypes later). Vectors and ranges always come
          // through as tensor / ptr types, so accepting `none` here
          // doesn't conflict with the slice path.
          bool AllScalarF64 = true;
          for (size_t i = 1; i < Idx.size(); ++i) {
            mlir::Type T = Idx[i].getType();
            if (T == F64 || mlir::isa<mlir::NoneType>(T)) continue;
            AllScalarF64 = false; break;
          }
          if (AllScalarF64 && (C.Args.size() == 1 || C.Args.size() == 2)) {
            std::string Cn = std::string("matlab_mat_") +
                (BA.FxSpec->Signed ? "i64_" : "u64_") +
                "subscript" +
                std::to_string((int)C.Args.size()) + "_s";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Cn));
            llvm::SmallVector<mlir::NamedAttribute, 8> AA;
            AA.push_back(Cal);
            auto Atrs = buildFixedAttrs(&MCtx, *BA.FxSpec);
            for (auto &E0 : Atrs) AA.push_back(E0);
            return emitUnreg("matlab.call_builtin", Idx, I64, L, AA);
          }
          // Slice path: the index produced a non-scalar (range / vector).
          // For 1-D fi vectors we route through matlab_mat_*_slice1.
          if (C.Args.size() == 1 && !AllScalarF64) {
            mlir::Value IxV = Idx[1];
            // The index value may be a tensor/range; ensure it's a ptr by
            // boxing through matlab_mat_from_buf path. The existing
            // tensor lowering rewrites range-typed values to ptr later,
            // so we leave this hook lazy: emit the call and let the
            // tensor pass adapt the operand. For literal integer ranges
            // produced by `matlab.range`, the LowerTensorOps pass already
            // emits a matlab_range(...) call yielding ptr — that's our
            // happy path here.
            std::string Cn = std::string("matlab_mat_") +
                (BA.FxSpec->Signed ? "i64_" : "u64_") + "slice1";
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, Cn));
            llvm::SmallVector<mlir::NamedAttribute, 8> AA;
            AA.push_back(Cal);
            auto Atrs = buildFixedAttrs(&MCtx, *BA.FxSpec);
            for (auto &E0 : Atrs) AA.push_back(E0);
            return emitUnreg("matlab.call_builtin", {Idx[0], IxV},
                             PtrTy, L, AA);
          }
        }
      }
    }

    /* 3-D subscript on a matlab_mat3 binding: A(i,j,k) scalar element →
     * matlab_subscript3_s; A(:,:,k) whole plane → matlab_subscript3_slice
     * (returns a 2-D matrix). */
    if (C.Args.size() == 3) {
      /* 3-D base may be a plain variable or a struct field / property
       * (#78).  Idx[0] is the already-lowered base mat3 ptr in both. */
      bool Is3DRead = false;
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        Is3DRead = isThreeDBinding(NE->Ref);
      else if (auto *F = dynamic_cast<const FieldAccess *>(C.Callee))
        if (auto *BN = dynamic_cast<const NameExpr *>(F->Base))
          Is3DRead = BN->Ref &&
                     ThreeDStructFields.count({BN->Ref, std::string(F->Field)});
      if (Is3DRead) {
          bool c0 = dynamic_cast<const ColonExpr *>(C.Args[0]) != nullptr;
          bool c1 = dynamic_cast<const ColonExpr *>(C.Args[1]) != nullptr;
          bool c2 = dynamic_cast<const ColonExpr *>(C.Args[2]) != nullptr;
          if (c0 && c1 && !c2) {
            auto SlicePtr = mlir::LLVM::LLVMPointerType::get(&MCtx);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_subscript3_slice"));
            return emitUnreg("matlab.call_builtin", {Idx[0], Idx[3]}, SlicePtr, L, {Cal});
          }
          if (!c0 && !c1 && !c2) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_subscript3_s"));
            return emitUnreg("matlab.call_builtin", Idx, F64, L, {Cal});
          }
        }
    }
    mlir::NamedAttribute NA(
        mlir::StringAttr::get(&MCtx, "nindices"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 64),
                               (int64_t)C.Args.size()));
    /* Eagerly refine the subscript result type when Sema left it as
     * None: scalar per-element access (all f64 indices) returns f64
     * via matlab_subscript{1,2}_s; anything involving a colon or a
     * slice returns a matrix (ptr). The rewriter in LowerTensorOps
     * reads this result type to pick the fast f64 path vs. the
     * generic slice path, so leaving it None would send scalar
     * A(i,j) through the slower path and — relevant to the REPL —
     * make the implicit-display check skip the result entirely. */
    mlir::Type SubRT = RT;
    if (mlir::isa<mlir::NoneType>(SubRT)) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      auto F64 = mlir::Float64Type::get(&MCtx);
      bool AllScalarF64 = true;
      /* Idx[0] is the base; indices start at 1. */
      for (size_t i = 1; i < Idx.size(); ++i)
        if (Idx[i].getType() != F64) { AllScalarF64 = false; break; }
      SubRT = AllScalarF64 ? (mlir::Type)F64 : (mlir::Type)PtrTy;
    }
    return emitUnreg("matlab.subscript", Idx, SubRT, L, {NA});
  }
  case NodeKind::CellIndex: {
    /* C{i} read (1-D) — routes to matlab_cell_get_f64 by default, or
     * matlab_cell_get_mat when Sema concretely says matrix.
     * C{r, k} read (2-D, Phase 1.3) — same dispatch but on the _2d
     * runtime entry.
     */
    auto &C = static_cast<const CellIndex &>(E);
    if (C.Args.empty() || C.Args.size() > 2)
      return emitUnreg("matlab.undef", {}, RT, L);
    mlir::Value Arr = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    bool WantMat = mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(RT);
    /* Sema can't type per-element, so a constant-index read of a slot we
     * recorded as matrix/string-stored (CellMatElems) forces get_mat —
     * otherwise `disp(c{k})` defaults to get_f64 and prints 0 for a
     * multi-element matrix element. */
    /* #206: a string-typed element (CellStrElems) reads back as a real
     * matlab_string* via matlab_cell_get_str — not the char-code row matrix
     * matlab_cell_get_mat would yield — so disp / string ops work. */
    bool WantStr = false;
    if (!WantMat && C.Args.size() == 1) {
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        if (NE->Ref) {
          int64_t k = -1;
          if (auto *IL = dynamic_cast<const IntegerLiteral *>(C.Args[0])) {
            try { k = std::stoll(std::string(IL->Text)); } catch (...) { k = -1; }
          } else if (dynamic_cast<const EndExpr *>(C.Args[0])) {
            auto Cnt = CellElemCount.find(NE->Ref);
            if (Cnt != CellElemCount.end()) k = Cnt->second;
          }
          auto SIt = CellStrElems.find(NE->Ref);
          if (k > 0 && SIt != CellStrElems.end() && SIt->second.count(k))
            WantStr = true;
          /* #233: a whole cell-of-strings (e.g. `parts = strsplit(...)`) reads
           * every element as a matlab_string* — including a runtime / variable
           * index that CellStrElems (constant-index only) can't cover. */
          if (!WantStr && CellAllStrBindings.count(NE->Ref))
            WantStr = true;
          auto It = CellMatElems.find(NE->Ref);
          if (!WantStr && k > 0 && It != CellMatElems.end() &&
              It->second.count(k))
            WantMat = true;
        }
    }
    mlir::Type ResTy = (WantMat || WantStr) ? (mlir::Type)PtrTy : (mlir::Type)F64;
    if (C.Args.size() == 1) {
      // Push a cell-numel sentinel (dim -1) so `end` in `c{end}` resolves to
      // matlab_cell_numel(c) rather than emitting a bare matlab.end.
      if (Arr) SubscriptCtx.push_back({Arr, -1});
      mlir::Value Idx = lowerExpr(*C.Args[0]);
      if (Arr) SubscriptCtx.pop_back();
      llvm::StringRef Callee = WantStr ? "matlab_cell_get_str"
                              : WantMat ? "matlab_cell_get_mat"
                                        : "matlab_cell_get_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      return emitUnreg("matlab.call_builtin", {Arr, Idx}, ResTy, L, {Cal});
    }
    mlir::Value R = lowerExpr(*C.Args[0]);
    mlir::Value K = lowerExpr(*C.Args[1]);
    llvm::StringRef Callee = WantMat ? "matlab_cell_get_mat_2d"
                                      : "matlab_cell_get_f64_2d";
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, Callee));
    return emitUnreg("matlab.call_builtin", {Arr, R, K}, ResTy, L, {Cal});
  }
  case NodeKind::FieldAccess: {
    /* s.x read  OR  s.a.b read. resolveStructBase walks a nested
     * chain via matlab_struct_get_child_struct so the intermediate
     * level always lands on a real struct pointer.
     *
     * If the base variable is pinned to a user class (e.g. because
     * it was assigned from `ClassName(...)` or is a class-method's
     * `obj` parameter), route through matlab_obj_get_* instead so the
     * class_id tag is preserved. */
    auto &F = static_cast<const FieldAccess &>(E);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    auto F64 = mlir::Float64Type::get(&MCtx);
    /* Fixed-Point Designer property access: `n.WordLength` / `.FractionLength`
     * / `.Signed` / `.IntegerLength` are compile-time constants drawn from
     * the FixedSpec. `n.Value` (real-world double) and `n.bin/hex/dec` are
     * Phase-4 surface and fall through. */
    if (F.Base && F.Base->Ty && F.Base->Ty->K == Type::Kind::Array) {
      auto &BA = static_cast<const ArrayType &>(*F.Base->Ty);
      if (BA.Elt == Dtype::Fixed && BA.FxSpec) {
        double Val = 0.0;
        bool Match = true;
        if (F.Field == "WordLength")          Val = (double)BA.FxSpec->WordLength;
        else if (F.Field == "FractionLength") Val = (double)BA.FxSpec->FractionLength;
        else if (F.Field == "Signed")         Val = BA.FxSpec->Signed ? 1.0 : 0.0;
        else if (F.Field == "IntegerLength")  Val = (double)BA.FxSpec->integerLength();
        else Match = false;
        if (Match) {
          mlir::NamedAttribute VA(
              mlir::StringAttr::get(&MCtx, "value"),
              mlir::FloatAttr::get(F64, Val));
          return emitUnreg("matlab.const_float", {}, F64, L, {VA});
        }
      }
    }
    /* Phase 5.3: T.<name> read — Base is a NameExpr in TableBindings.
     * Returns the column matlab_mat *. */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && isTableBinding(BN->Ref)) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value Tv = lowerExpr(*F.Base);
        mlir::Value NameV = emitFieldNameChar(F.Field, L);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_table_get_column"));
        return emitUnreg("matlab.call_builtin", {Tv, NameV},
                         PtrTy, L, {Cal});
      }
    /* Phase 5.4 (cont.): TMW.<name> read on a timetable binding.
     *   TMW.Time              -> matlab_datetime_vec * (RowTimes)
     *   TMW.<colName>         -> matlab_mat * (numeric column)
     * The implicit-display + arithmetic dispatch for the returned
     * value follows the same datetime_vec / matrix flow as the
     * standalone constructors. */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && isTimetableBinding(BN->Ref)) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value Tv = lowerExpr(*F.Base);
        if (F.Field == "Time") {
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_get_row_times"));
          return emitUnreg("matlab.call_builtin", {Tv}, PtrTy, L, {Cal});
        }
        mlir::Value NameV = emitFieldNameChar(F.Field, L);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_timetable_get_column"));
        return emitUnreg("matlab.call_builtin", {Tv, NameV},
                         PtrTy, L, {Cal});
      }
    /* Phase 2: s(i).x read — Base is `CallOrIndex(NameExpr s, [i])`
     * where s is a struct-array binding. Pull the i-th element via
     * matlab_struct_arr_get and field-get on the result. */
    if (auto *CI = dynamic_cast<const CallOrIndex *>(F.Base)) {
      auto *NE = dynamic_cast<const NameExpr *>(CI->Callee);
      if (NE && NE->Ref && isStructArrayBinding(NE->Ref) &&
          CI->Args.size() == 1 && CI->Args[0]) {
        mlir::Value Slot = ensureStructArraySlot(NE->Ref, NE->Name, L);
        mlir::Value Arr = emitLoad(Slot, PtrTy, L);
        mlir::Value Idx = lowerExpr(*CI->Args[0]);
        mlir::NamedAttribute GCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_struct_arr_get"));
        mlir::Value Elem = emitUnreg("matlab.call_builtin", {Arr, Idx},
                                      PtrTy, L, {GCal});
        mlir::Value NameV = emitFieldNameChar(F.Field, L);
        bool WantMat = mlir::isa<mlir::RankedTensorType,
                                  mlir::UnrankedTensorType>(RT);
        /* A field recorded matrix/string-valued against the array binding
         * (e.g. fastaread's Header/Sequence) reads via get_mat even when
         * Sema left the element field type open — without this it defaults
         * to get_f64 and returns 0 for a char-string field. */
        if (!WantMat &&
            MatStructFields.count({NE->Ref, std::string(F.Field)}))
          WantMat = true;
        /* #258 (struct-array variant): a struct array that round-trips into a
         * later REPL turn loses its per-field element-kind (MatStructFields is
         * same-TU only), so a matrix/string field (`s(1).Header`) defaulted to
         * a scalar get_f64 and read back 0 / empty.  For a cross-turn struct-
         * array binding (IsStructArray, re-pinned from kind=14) fetch via
         * get_mat — kind-aware, boxes a true scalar to 1x1.  Gated to the
         * cross-turn case (NOT same-TU StructArrayBindings, which carry
         * MatStructFields), mirroring the plain-struct fix. */
        if (!WantMat && NE->Ref->IsStructArray &&
            !StructArrayBindings.count(NE->Ref))
          WantMat = true;
        llvm::StringRef Callee = WantMat ? "matlab_struct_get_mat"
                                          : "matlab_struct_get_f64";
        mlir::NamedAttribute SCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        mlir::Type ResTy = WantMat ? (mlir::Type)PtrTy : (mlir::Type)F64;
        return emitUnreg("matlab.call_builtin", {Elem, NameV},
                         ResTy, L, {SCal});
      }
    }
    const ClassDef *PinnedCls = nullptr;
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && BN->Ref->PinnedClass) PinnedCls = BN->Ref->PinnedClass;
    /* Nested property read on a class whose property is itself typed as a
     * class (issue #28): `R.Displacement.Magnitude` — F.Base is the inner
     * FieldAccess `R.Displacement`.  Resolve the inner property's class
     * annotation (`Displacement pdeDisplacement`) so the leaf read sees a
     * pinned class and fetches matrix fields via `_get_mat`.  Looks the
     * class up by name in the current TU's class list. */
    if (!PinnedCls)
      if (auto *FB = dynamic_cast<const FieldAccess *>(F.Base))
        if (auto *BBN = dynamic_cast<const NameExpr *>(FB->Base))
          if (BBN->Ref && BBN->Ref->PinnedClass) {
            for (const ClassDef *CC = BBN->Ref->PinnedClass; CC; CC = CC->Super) {
              const ClassProp *Found = nullptr;
              for (const auto &P : CC->Props)
                if (P.Name == FB->Field) { Found = &P; break; }
              if (!Found) continue;
              if (!Found->TypeName.empty() && CurTU)
                for (const ClassDef *K : CurTU->Classes)
                  if (K && K->Name == Found->TypeName) { PinnedCls = K; break; }
              break;
            }
          }
    /* Enumeration member reference: `ClassName.Member`. The base is a
     * NameExpr whose binding is BindingKind::Class, not a pinned var.
     * Each member gets its 0-based position as an f64 constant;
     * equality comparisons then work via plain numeric compare. */
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base)) {
      if (BN->Ref && BN->Ref->Kind == BindingKind::Class &&
          BN->Ref->ClassDef) {
        const ClassDef *CD = BN->Ref->ClassDef;
        for (size_t i = 0; i < CD->EnumMembers.size(); ++i) {
          if (CD->EnumMembers[i] == F.Field) {
            mlir::NamedAttribute VA(
                mlir::StringAttr::get(&MCtx, "value"),
                mlir::FloatAttr::get(F64, (double)i));
            return emitUnreg("matlab.const_float", {}, F64, L, {VA});
          }
        }
      }
    }
    if (PinnedCls) {
      /* Dependent property: no stored backing — dispatch to the
       * class's get.Prop method (emitted as ClassName__get.Prop). */
      const ClassProp *DepProp = nullptr;
      const ClassDef *DepOwner = nullptr;
      for (const ClassDef *CC = PinnedCls; CC; CC = CC->Super) {
        for (const auto &P : CC->Props)
          if (P.Name == F.Field) {
            if (P.Dependent) { DepProp = &P; DepOwner = CC; }
            break;
          }
        if (DepProp) break;
      }
      if (DepProp) {
        mlir::Value Obj = lowerExpr(*F.Base);
        std::string Callee = std::string(DepOwner->Name) +
                              "__get_" + std::string(F.Field);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, Callee));
        mlir::Type ResTy = F64;
        return emitUnreg("matlab.call", {Obj}, ResTy, L, {Cal});
      }
      mlir::Value Obj = lowerExpr(*F.Base);
      mlir::Value NameV = emitFieldNameChar(F.Field, L);
      bool WantMat = mlir::isa<mlir::RankedTensorType,
                                mlir::UnrankedTensorType>(RT);
      /* CST prelude classes (tf / ss / zpk / pid / frd) carry
       * matrix-typed properties (Numerator / A / Z / etc.) almost
       * exclusively. Sema can't see through the property to the
       * stored type, so without this nudge a `G.Numerator` read
       * defaults to `_get_f64` and silently returns 0 when the
       * field actually holds an NxM vector. Force `_get_mat` for
       * these classes — `matlab_struct_get_mat` auto-boxes f64
       * fields back to 1×1 matrices, so scalar properties that
       * happen to land on the same dispatch still print the same
       * digits via `matlab_disp_mat`. Other user classdefs (Vec2,
       * BasicClass, …) keep the f64 default so their scalar
       * arithmetic stays in scalar lanes. */
      bool IsCstClass = false;
      if (PinnedCls) {
        llvm::StringRef CN = PinnedCls->Name;
        IsCstClass = (CN == "tf" || CN == "ss" || CN == "zpk" ||
                      CN == "pid" || CN == "frd" ||
                      /* PDE Toolbox classdef façade: femodel carries
                       * cell-array properties (FaceBC / FaceLoad /
                       * EdgeBC / …) and matrix-valued results
                       * (Geometry / Mesh).  materialProperties /
                       * faceBC / faceLoad value-types hold matrix or
                       * string fields with default `[]` so Sema can't
                       * distinguish — same nudge as the CST family
                       * forces `_get_mat` for downstream reads. */
                      CN == "femodel" || CN == "materialProperties" ||
                      CN == "faceBC"  || CN == "edgeBC" || CN == "vertexBC" ||
                      CN == "faceLoad" || CN == "edgeLoad" ||
                      CN == "vertexLoad" || CN == "cellLoad" ||
                      CN == "StaticStructuralResults" ||
                      CN == "StationaryResults" ||
                      CN == "ThermalResults" ||
                      CN == "ElectrostaticResults" ||
                      CN == "MagneticResults" ||
                      CN == "DCConductionResults" ||
                      CN == "TransientStructuralResults" ||
                      CN == "ModalStructuralResults" ||
                      CN == "FrequencyStructuralResults" ||
                      CN == "HarmonicEMResults" ||
                      CN == "pdeDisplacement" || CN == "pdeModeShapes" ||
                      /* MPC Toolbox Tier-1: the mpc classdef caches
                       * Sx / Su / Su1 / H / R / L / Wy / Wdu / umin /
                       * umax / A / B / C as matrix properties; the
                       * scalar slots (Ts / p / m / rho_eps) flow as
                       * f64 separately via the explicit scalar
                       * accessors.  Tier-2 adds mpcmoveopt — its
                       * MVMin / MVMax / OutputMin / OutputMax are
                       * matrix-valued, the Use_* flags are scalar. */
                      CN == "mpc" || CN == "mpcstate" ||
                      CN == "mpcmoveopt" || CN == "mpcsimopt" ||
                      CN == "explicitMPC" || CN == "nlmpc");
      }
      if (IsCstClass && !WantMat &&
          !mlir::isa<mlir::Float64Type>(RT)) WantMat = true;
      /* MATLAB property type annotation override: when the classdef
       * declares the property as `Name string`, the parser stamps
       * `TypeName = "string"` on the ClassProp.  Route reads to
       * `matlab_obj_get_string` so the returned `matlab_string *`
       * flows through string-aware downstream sites (disp, concat,
       * etc.) rather than being mis-typed as f64. */
      bool IsString = false;
      bool IsMatField = false;
      if (PinnedCls) {
        for (const ClassDef *CC = PinnedCls; CC; CC = CC->Super) {
          for (const auto &P : CC->Props)
            if (P.Name == F.Field) {
              if (P.TypeName == "string") IsString = true;
              else if (P.TypeName == "complex" || P.TypeName == "matrix" ||
                       P.TypeName == "double_col" || P.TypeName == "col")
                IsMatField = true;
              else if (!P.TypeName.empty() && CurTU) {
                /* Property typed as another class (e.g. a result's
                 * `Displacement pdeDisplacement` sub-object, #28): fetch
                 * the child struct as a ptr so the next `.field` read
                 * resolves against it. */
                for (const ClassDef *K : CurTU->Classes)
                  if (K && K->Name == P.TypeName) { IsMatField = true; break; }
              }
              break;
            }
          if (IsString || IsMatField) break;
        }
      }
      llvm::StringRef Callee = IsString               ? "matlab_obj_get_string"
                               : (WantMat || IsMatField) ? "matlab_obj_get_mat"
                                                          : "matlab_obj_get_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      mlir::Type ResTy = (IsString || WantMat || IsMatField) ? (mlir::Type)PtrTy
                                                              : (mlir::Type)F64;
      return emitUnreg("matlab.call_builtin", {Obj, NameV}, ResTy, L, {Cal});
    }
    mlir::Value SPtr = resolveStructBase(F.Base, L);
    if (!SPtr) return emitUnreg("matlab.undef", {}, RT, L);
    mlir::Value NameV = emitFieldNameChar(F.Field, L);
    /* Default to f64 (scalar field). Only fetch as a matrix when Sema
     * concretely says tensor — a `none`/`any` type, common when Sema
     * can't specialise through struct fields, falls back to f64. Users
     * who want matrix fields can annotate or the runtime will box a
     * 1×1 transparently. */
    bool WantMat = mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(RT);
    /* A field recorded as matrix-valued at its assignment reads as a matrix
     * even when Sema left the field type open (`none`/`any`). */
    if (!WantMat)
      if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
        if (BN->Ref && MatStructFields.count({BN->Ref, std::string(F.Field)}))
          WantMat = true;
    /* `s(i).Field` — the base is an index expression, not the array name.
     * Dig out the underlying struct-array binding so a matrix/string field
     * recorded against it (e.g. fastaread's Header/Sequence) reads as a
     * matrix rather than defaulting to get_f64 (which returns 0). */
    if (!WantMat)
      if (auto *CI = dynamic_cast<const CallOrIndex *>(F.Base))
        if (auto *BN = dynamic_cast<const NameExpr *>(CI->Callee))
          if (BN->Ref && MatStructFields.count({BN->Ref, std::string(F.Field)}))
            WantMat = true;
    /* #258: a struct that round-trips into a later REPL turn loses its per-
     * field element-kind (MatStructFields is same-TU only), so a matrix field
     * (`l2info.Payload`) defaults to get_f64 and a builtin consuming it
     * (`biterr(a, s.Payload)`) backs off to "unsupported call shape".  For a
     * cross-turn struct binding (IsStruct, re-pinned from kind=12) whose field
     * type Sema can't see, fetch via get_mat — matlab_struct_get_mat is kind-
     * aware and boxes a genuine scalar field into a 1x1, so this is safe for
     * scalar fields too.  Gated to the cross-turn case (NOT same-TU
     * StructBindings/StructInitialised, which already carry MatStructFields)
     * so same-turn scalar-field reads keep their native-f64 fast path. */
    if (!WantMat)
      if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
        if (BN->Ref && BN->Ref->IsStruct &&
            !StructBindings.count(BN->Ref) &&
            !StructInitialised.count(BN->Ref))
          WantMat = true;
    llvm::StringRef Callee = WantMat ? "matlab_struct_get_mat"
                                      : "matlab_struct_get_f64";
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, Callee));
    mlir::Type ResTy = WantMat ? (mlir::Type)PtrTy : (mlir::Type)F64;
    return emitUnreg("matlab.call_builtin", {SPtr, NameV}, ResTy, L, {Cal});
  }
  case NodeKind::DynamicField: {
    /* s.(name_expr). v1 handles the compile-time-constant case where
     * name_expr is a literal char/string (the common use when
     * templating fieldnames from a small set). For runtime-varying
     * names we'd need a runtime entry that takes a char-matrix name;
     * that's a follow-up. */
    auto &F = static_cast<const DynamicField &>(E);
    mlir::Value SPtr = resolveStructBase(F.Base, L);
    if (!SPtr) return emitUnreg("matlab.undef", {}, RT, L);
    std::string FieldName;
    if (auto *Lit = dynamic_cast<const StringLiteral *>(F.Name))
      FieldName = Lit->Value;
    else if (auto *Lit = dynamic_cast<const CharLiteral *>(F.Name))
      FieldName = Lit->Value;
    else
      return emitUnreg("matlab.undef", {}, RT, L);
    auto F64 = mlir::Float64Type::get(&MCtx);
    mlir::Value NameV = emitFieldNameChar(FieldName, L);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_struct_get_f64"));
    return emitUnreg("matlab.call_builtin", {SPtr, NameV}, F64, L, {Cal});
  }
  case NodeKind::MatrixLiteral: {
    auto &M = static_cast<const MatrixLiteral &>(E);
    bool SingleRow = M.Rows.size() == 1;
    /* #105: object-array literal `[A; B; C]` / `[A B C]` over classdef
     * instances builds an object array via the generic runtime carrier
     * (matlab_dlnet_oa_new + matlab_dlnet_oa_append) — the same form the
     * explicit objArrayNew/Append API uses — NOT a matlab_vertcat (which
     * reinterprets the matlab_obj* pointers as matlab_mat* and concatenates
     * garbage, then crashes when objArrayGet/extractdata read it back).
     *
     * LowerTensorOps has an equivalent detector, but it keys off the
     * operand's *defining op* (a classdef-method func.call or a load from a
     * class_id-tagged alloc) — which only matches the AOT lane.  In ReplMode
     * a script-scope classdef var reads through matlab_ws_get_mat (an opaque
     * call_builtin), so that detector misses it and the literal falls through
     * to matlab_vertcat → the #105 crash.  Detect here at the AST level where
     * the Sema-pinned class is visible (Ref->PinnedClass), so the object
     * array is built identically on both lanes.  Require EVERY element to be
     * a classdef-pinned NameExpr; a partial / mixed literal falls through.
     * Require 2+ elements — a single `[A]` is just `A`, not an array. */
    {
      llvm::SmallVector<const Expr *, 8> elems;
      bool allClassdef = true;
      for (auto &Row : M.Rows)
        for (const Expr *Cx : Row) {
          auto *NE = dynamic_cast<const NameExpr *>(Cx);
          if (!NE || !NE->Ref || NE->Ref->PinnedClass == nullptr) {
            allClassdef = false;
          }
          elems.push_back(Cx);
        }
      if (allClassdef && elems.size() >= 2) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::NamedAttribute NewCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_dlnet_oa_new"));
        mlir::Value Arr = emitUnreg("matlab.call_builtin", {}, PtrTy, L, {NewCal});
        for (const Expr *Cx : elems) {
          mlir::Value Elem = lowerExpr(*Cx);
          mlir::NamedAttribute AppCal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_dlnet_oa_append"));
          Arr = emitUnreg("matlab.call_builtin", {Arr, Elem}, PtrTy, L, {AppCal});
        }
        return Arr;
      }
    }
    /* Phase 5.4 (cont.): [TT1 TT2 ... TTN] over timetable bindings —
     * pairwise-reduce through matlab_timetable_horzcat. All entries
     * must be NameExprs in TimetableBindings; mixed-type bracket
     * concats fall through to the matrix lane. */
    if (SingleRow && !M.Rows[0].empty()) {
      bool AllTT = true;
      for (const Expr *Cx : M.Rows[0]) {
        auto *NE = dynamic_cast<const NameExpr *>(Cx);
        if (!NE || !NE->Ref || !isTimetableBinding(NE->Ref)) {
          AllTT = false; break;
        }
      }
      if (AllTT) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        mlir::Value Acc = lowerExpr(*M.Rows[0][0]);
        for (size_t i = 1; i < M.Rows[0].size(); ++i) {
          mlir::Value Rhs = lowerExpr(*M.Rows[0][i]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_timetable_horzcat"));
          Acc = emitUnreg("matlab.call_builtin", {Acc, Rhs},
                          PtrTy, L, {Cal});
        }
        return Acc;
      }
    }
    /* Char-array / string bracket concat: `['x = ', num2str(v), ' kg']`
     * is MATLAB's classic "build a string from pieces" idiom. If any
     * element of a single-row literal is a char/string, treat the
     * whole bracket as string concatenation: each element is coerced
     * to a matlab_string* (chars via matlab_string_from_literal,
     * scalars via num2str, ptrs passed through) and chained through
     * matlab_string_concat. The result type is a matlab_string*
     * (LLVM ptr) so subsequent disp/strlen/+ route to the string
     * runtime. */
    if (SingleRow) {
      bool HasStringy = false;
      for (const Expr *Cx : M.Rows[0])
        if (Cx && (Cx->Kind == NodeKind::CharLiteral ||
                   Cx->Kind == NodeKind::StringLiteral ||
                   isStringExpr(Cx))) { HasStringy = true; break; }
      if (HasStringy) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto charToString = [&](llvm::StringRef Text) -> mlir::Value {
          mlir::NamedAttribute VA(
              mlir::StringAttr::get(&MCtx, "value"),
              mlir::StringAttr::get(&MCtx, Text));
          mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                      mlir::NoneType::get(&MCtx), L, {VA});
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
          return emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
        };
        auto toString = [&](const Expr *Cx) -> mlir::Value {
          if (auto *CL = dynamic_cast<const CharLiteral *>(Cx))
            return charToString(CL->Value);
          mlir::Value V = lowerExpr(*Cx);
          /* AST-side this is already a string (StringLiteral, string
           * binding, or a string-returning builtin call). The lowered
           * type may still be `none` for builtin calls whose result
           * type wasn't refined (num2str/upper/...); upgrade to ptr
           * so downstream matlab_string_concat picks it up. */
          if (isStringExpr(Cx)) {
            if (V.getType() != PtrTy) V.setType(PtrTy);
            return V;
          }
          if (V.getType() == PtrTy) return V;
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
            if (IT.getWidth() == 1)
              V = mlir::arith::UIToFPOp::create(B, L, F64, V);
            else
              V = mlir::arith::SIToFPOp::create(B, L, F64, V);
          } else if (mlir::isa<mlir::FloatType>(V.getType()) &&
                     V.getType() != F64) {
            V = mlir::arith::ExtFOp::create(B, L, F64, V);
          }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "num2str"));
          return emitUnreg("matlab.call_builtin", {V}, PtrTy, L, {Cal});
        };
        mlir::Value Acc;
        for (const Expr *Cx : M.Rows[0]) {
          if (!Cx) continue;
          mlir::Value V = toString(Cx);
          if (!Acc) { Acc = V; continue; }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_concat"));
          Acc = emitUnreg("matlab.call_builtin", {Acc, V}, PtrTy, L, {Cal});
        }
        if (Acc) return Acc;
        /* All-empty row — fall through to the generic path. */
      }
    }
    /* Phase 1.3 — cell concatenation. `[a, b]` / `[a; b]` where every
     * element is a cell-bound NameExpr or a CellLiteral chains
     * matlab_cell_concat_row / _concat_col into a fresh cell. The
     * runtime helpers borrow element pointers (no deep copy).
     *
     * Detection: each AST element is either a CellLiteral or a
     * NameExpr whose binding is in CellBindings. If any element fails
     * the check, fall through to the generic numeric concat path. */
    auto isCellElem = [&](const Expr *X) -> bool {
      if (!X) return false;
      if (X->Kind == NodeKind::CellLiteral) return true;
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        if (NE->Ref && CellBindings.count(NE->Ref)) return true;
      return false;
    };
    bool AllCells = !M.Rows.empty();
    for (auto &R : M.Rows) {
      if (R.empty()) { AllCells = false; break; }
      for (const Expr *X : R)
        if (!isCellElem(X)) { AllCells = false; break; }
      if (!AllCells) break;
    }
    if (AllCells) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      llvm::SmallVector<mlir::Value, 4> RowAccs;
      for (auto &R : M.Rows) {
        mlir::Value Acc;
        for (const Expr *X : R) {
          mlir::Value V = lowerExpr(*X);
          if (!Acc) { Acc = V; continue; }
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_cell_concat_row"));
          Acc = emitUnreg("matlab.call_builtin", {Acc, V}, PtrTy, L, {Cal});
        }
        RowAccs.push_back(Acc);
      }
      mlir::Value Out = RowAccs.front();
      for (size_t i = 1; i < RowAccs.size(); ++i) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_cell_concat_col"));
        Out = emitUnreg("matlab.call_builtin", {Out, RowAccs[i]},
                        PtrTy, L, {Cal});
      }
      return Out;
    }

    /* Phase 6.2 — symbolic matrix literal. `[a 1; 2 b]` where any
     * element is sym-typed routes through matlab_symmat_zeros +
     * per-cell matlab_symmat_set, producing a matlab_symmat* (kind=8).
     * Without this, the f64 matrix path would call matlab_mat_from_buf
     * with the sym* pointers as data — at runtime, those reinterpret as
     * f64 garbage. Each cell is boxed via matlab_sym_from_double for
     * numeric literals; sym entries flow through directly. */
    {
      bool AnySymCell = false;
      bool AllSymCells = !M.Rows.empty();
      for (auto &Row : M.Rows) {
        if (Row.empty()) { AllSymCells = false; break; }
        for (const Expr *Cx : Row) {
          if (Cx && exprIsSym(Cx)) { AnySymCell = true; }
        }
      }
      if (AnySymCell) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        auto F64 = mlir::Float64Type::get(&MCtx);
        auto I64 = mlir::IntegerType::get(&MCtx, 64);
        size_t Rows = M.Rows.size();
        size_t Cols = 0;
        for (auto &Row : M.Rows) Cols = std::max(Cols, Row.size());
        auto i64Const = [&](int64_t v) -> mlir::Value {
          return mlir::arith::ConstantOp::create(
              B, L, I64, mlir::IntegerAttr::get(I64, v)).getResult();
        };
        mlir::NamedAttribute ZerosCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_symmat_zeros"));
        mlir::Value Mat = emitUnreg("matlab.call_builtin",
                                     {i64Const((int64_t)Rows),
                                      i64Const((int64_t)Cols)},
                                     PtrTy, L, {ZerosCal});
        mlir::NamedAttribute SetCal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_symmat_set"));
        mlir::NamedAttribute FromDouble(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_sym_from_double"));
        for (size_t i = 0; i < Rows; ++i) {
          for (size_t j = 0; j < M.Rows[i].size(); ++j) {
            const Expr *Cx = M.Rows[i][j];
            if (!Cx) continue;
            mlir::Value V = lowerExpr(*Cx);
            /* Box numeric scalars into a sym; sym values flow directly. */
            if (V && V.getType() == F64)
              V = emitUnreg("matlab.call_builtin", {V},
                            PtrTy, L, {FromDouble});
            /* Sym-typed loads (e.g. NameExpr referencing a `syms`
             * binding) may come back none-typed from the slot load
             * before RefineSlotTypes runs. Force the type to PtrTy
             * since exprIsSym already verified the source is sym. */
            if (V && V.getType() != PtrTy) V.setType(PtrTy);
            emitUnregOp("matlab.call_builtin",
                        {Mat, i64Const((int64_t)i), i64Const((int64_t)j), V},
                        {mlir::NoneType::get(&MCtx)}, L, {SetCal});
          }
        }
        return Mat;
      }
    }

    /* fi-typed row vector: route every element through matlab_mat_i64
     * (or _u64) helpers and chain concat calls. We accept scalar fi
     * elements (wrap via matlab_mat_i64_from_scalar) and existing fi
     * arrays (use directly). */
    if (SingleRow && E.Ty && E.Ty->K == Type::Kind::Array) {
      auto &OutA = static_cast<const ArrayType &>(*E.Ty);
      if (OutA.Elt == Dtype::Fixed && OutA.FxSpec) {
        auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
        bool Signed = OutA.FxSpec->Signed;
        std::string FromScalar = std::string("matlab_mat_") +
            (Signed ? "i64_" : "u64_") + "from_scalar";
        std::string Concat = std::string("matlab_mat_") +
            (Signed ? "i64_" : "u64_") + "concat_row";
        llvm::SmallVector<mlir::Value, 4> Pieces;
        for (const Expr *Cx : M.Rows[0]) {
          if (!Cx) continue;
          mlir::Value V = lowerExpr(*Cx);
          if (V.getType() != PtrTy) {
            // Scalar (i8/i16/i32/i64) — wrap to a 1-element typed matrix
            // via matlab_mat_*_from_scalar(int64).
            auto I64 = mlir::IntegerType::get(&MCtx, 64);
            if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
              if (IT.getWidth() < 64)
                V = Signed
                    ? (mlir::Value)mlir::arith::ExtSIOp::create(B, L, I64, V)
                    : (mlir::Value)mlir::arith::ExtUIOp::create(B, L, I64, V);
            }
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, FromScalar));
            llvm::SmallVector<mlir::NamedAttribute, 8> AA;
            AA.push_back(Cal);
            auto Atrs = buildFixedAttrs(&MCtx, *OutA.FxSpec);
            for (auto &E0 : Atrs) AA.push_back(E0);
            V = emitUnreg("matlab.call_builtin", {V}, PtrTy, L, AA);
          }
          Pieces.push_back(V);
        }
        if (Pieces.empty()) {
          /* Empty matrix — fall through to the generic path. */
        } else {
          mlir::Value Acc = Pieces[0];
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Concat));
          llvm::SmallVector<mlir::NamedAttribute, 8> AA;
          AA.push_back(Cal);
          auto Atrs = buildFixedAttrs(&MCtx, *OutA.FxSpec);
          for (auto &E0 : Atrs) AA.push_back(E0);
          for (size_t i = 1; i < Pieces.size(); ++i) {
            Acc = emitUnreg("matlab.call_builtin", {Acc, Pieces[i]},
                             PtrTy, L, AA);
          }
          return Acc;
        }
      }
    }
    llvm::SmallVector<mlir::Value, 4> Rows;
    for (auto &R : M.Rows) {
      llvm::SmallVector<mlir::Value, 4> Cs;
      for (const Expr *C : R) if (C) Cs.push_back(lowerExpr(*C));
      // For a single-row literal the concat_row *is* the matrix result, so
      // give it the sema-inferred type. Multi-row literals feed concat_col
      // and the row type stays opaque.
      mlir::Type RowTy = SingleRow ? RT : mlir::NoneType::get(&MCtx);
      mlir::Value Row = emitUnreg("matlab.concat_row", Cs, RowTy, L);
      Rows.push_back(Row);
    }
    if (SingleRow) return Rows.front();
    return emitUnreg("matlab.concat_col", Rows, RT, L);
  }
  case NodeKind::CellLiteral: {
    /* {a, b, c, ...} creates a matlab_cell and sets slot i = expr_i.
     * Single-row literals: 1-D shape (rows=1, cols=N). Multi-row
     * literals (Phase 1.3): 2-D shape with explicit
     * matlab_cell_new_2d(rows, cols) and per-cell
     * matlab_cell_set_<f64|mat>_2d(c, r, k, v). All rows must have the
     * same length — the parser enforces that for the matrix grammar
     * and the same shape carries here.
     *
     * Kind is picked from each element's MLIR type at the call site:
     * ptr -> matlab_cell_set_mat, else -> matlab_cell_set_f64. */
    auto &M = static_cast<const CellLiteral &>(E);
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    /* A char/string literal element (`{'a', 'b'}`) becomes a matlab_string*
     * ptr stored with cell_set_str (kind=3); cell_get_mat exposes it as a
     * char-code row so the char-matrix consumers (legend, …) keep working.
     * Other elements lower normally. Sets `IsStr` when a string was made. */
    auto lowerCellElem = [&](const Expr *El, bool &IsStr) -> mlir::Value {
      std::string Txt;
      if (auto *CL = dynamic_cast<const CharLiteral *>(El)) { Txt = std::string(CL->Value); IsStr = true; }
      else if (auto *SL = dynamic_cast<const StringLiteral *>(El)) { Txt = std::string(SL->Value); IsStr = true; }
      if (IsStr) {
        mlir::NamedAttribute VA(mlir::StringAttr::get(&MCtx, "value"),
                                mlir::StringAttr::get(&MCtx, Txt));
        mlir::Value Ch = emitUnreg("matlab.const_char", {},
                                    mlir::NoneType::get(&MCtx), L, {VA});
        mlir::NamedAttribute Cal(mlir::StringAttr::get(&MCtx, "callee"),
                                 mlir::StringAttr::get(&MCtx, "matlab_string_from_literal"));
        return emitUnreg("matlab.call_builtin", {Ch}, PtrTy, L, {Cal});
      }
      return lowerExpr(*El);
    };
    bool TwoD = M.Rows.size() > 1;
    if (TwoD) {
      size_t Rcount = M.Rows.size();
      size_t Ccount = M.Rows[0].size();
      mlir::Value RC = mlir::arith::ConstantOp::create(
          B, L, F64, mlir::FloatAttr::get(F64, (double)Rcount));
      mlir::Value CC = mlir::arith::ConstantOp::create(
          B, L, F64, mlir::FloatAttr::get(F64, (double)Ccount));
      mlir::NamedAttribute New(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_cell_new_2d"));
      mlir::Value Cell = emitUnreg("matlab.call_builtin", {RC, CC},
                                    PtrTy, L, {New});
      for (size_t r = 0; r < Rcount; ++r) {
        for (size_t k = 0; k < M.Rows[r].size() && k < Ccount; ++k) {
          const Expr *El = M.Rows[r][k];
          if (!El) continue;
          mlir::Value V = lowerExpr(*El);
          mlir::Value Ri = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, (double)(r + 1)));
          mlir::Value Ki = mlir::arith::ConstantOp::create(
              B, L, F64, mlir::FloatAttr::get(F64, (double)(k + 1)));
          bool IsMat = V && (V.getType() == PtrTy ||
                             mlir::isa<mlir::RankedTensorType,
                                       mlir::UnrankedTensorType>(V.getType()));
          llvm::StringRef Callee = IsMat ? "matlab_cell_set_mat_2d"
                                          : "matlab_cell_set_f64_2d";
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, Callee));
          emitUnregOp("matlab.call_builtin", {Cell, Ri, Ki, V},
                      {mlir::NoneType::get(&MCtx)}, L, {Cal});
        }
      }
      return Cell;
    }
    /* 1-D path. */
    struct CellElem { mlir::Value V; bool IsStr; };
    llvm::SmallVector<CellElem, 8> Elems;
    for (auto &R : M.Rows)
      for (const Expr *El : R)
        if (El) { bool S = false; mlir::Value V = lowerCellElem(El, S);
                  Elems.push_back({V, S}); }
    mlir::Value Cnt = mlir::arith::ConstantOp::create(
        B, L, F64, mlir::FloatAttr::get(F64, (double)Elems.size()));
    mlir::NamedAttribute New(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_cell_new"));
    mlir::Value Cell = emitUnreg("matlab.call_builtin", {Cnt},
                                  PtrTy, L, {New});
    for (size_t i = 0; i < Elems.size(); ++i) {
      mlir::Value Idx = mlir::arith::ConstantOp::create(
          B, L, F64, mlir::FloatAttr::get(F64, (double)(i + 1)));
      mlir::Value V = Elems[i].V;
      /* Tensor and ptr both route to set_mat — a literal matrix is
       * tensor-typed at lowering time and gets retyped to ptr by
       * LowerTensorOps later. A string element routes to set_str. */
      bool IsMat = V && (V.getType() == PtrTy ||
                         mlir::isa<mlir::RankedTensorType,
                                   mlir::UnrankedTensorType>(V.getType()));
      llvm::StringRef Callee = Elems[i].IsStr ? "matlab_cell_set_str"
                              : IsMat          ? "matlab_cell_set_mat"
                                               : "matlab_cell_set_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      emitUnregOp("matlab.call_builtin", {Cell, Idx, V},
                  {mlir::NoneType::get(&MCtx)}, L, {Cal});
    }
    return Cell;
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<const AnonFunction &>(E);
    std::string Joined;
    for (size_t i = 0; i < A.Params.size(); ++i) {
      if (i) Joined += ",";
      Joined += std::string(A.Params[i]);
    }
    mlir::NamedAttribute PA(
        mlir::StringAttr::get(&MCtx, "params"),
        mlir::StringAttr::get(&MCtx, Joined));

    /* Detect captures: free variables in the body that aren't params,
     * builtins, or user functions. These become additional leading
     * block args + matlab.make_anon operands so each call_indirect can
     * thread the captured values through to the outlined llvm.func. */
    std::vector<Binding *> Captures;
    std::unordered_set<Binding *> Seen;
    if (A.Body) collectCaptures(A.Body, A.ParamRefs, Captures, Seen);

    auto F64Ty = mlir::Float64Type::get(&MCtx);

    /* Materialize each capture at the @-site. Capture element type comes
     * from the binding's Sema-inferred type: scalar -> f64, tensor ->
     * ptr (via the slot's tensor type which LowerTensorOps later
     * retypes). The outer slot load, the spill slot, the make_anon
     * operand and the corresponding anon-region block argument all
     * share this capture type. */
    llvm::SmallVector<mlir::Value, 4> CaptureVals;
    llvm::SmallVector<mlir::Type, 4>  CaptureTys;
    std::vector<mlir::Value> CaptureSpills;
    for (Binding *Bnd : Captures) {
      const Type *BTy = Bnd->InferredType ? Bnd->InferredType
                                          : TC.scalar(Dtype::Double);
      mlir::Value Cur;
      mlir::Type MTy;
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      if (ReplMode && InScriptBody && Slots.find(Bnd) == Slots.end() &&
          MatrixWsBindings.count(Bnd)) {
        /* #77: a workspace-backed matrix capture (`@(s) M*s` with
         * `M = [..]`) has no local slot; read it from the workspace as a
         * ptr (matlab_ws_get_mat) so the capture is typed as a matrix
         * rather than defaulting to f64 via a fabricated scalar slot. */
        mlir::Value NameV = emitFieldNameChar(Bnd->Name, L);
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, "matlab_ws_get_mat"));
        Cur = emitUnreg("matlab.call_builtin", {NameV}, PtrTy, L, {Cal});
        MTy = PtrTy;
      } else {
        /* Prefer the outer slot's concrete MLIR type: Sema-level
         * InferredType is often still `any` for script-scope matrix
         * assignments even though the slot was allocated with a real
         * tensor type by the matrix-literal store earlier. */
        mlir::Value OuterSlot = getOrCreateSlot(Bnd, BTy, Bnd->Name, L);
        MTy = OuterSlot.getType();
        if (mlir::isa<mlir::NoneType>(MTy)) MTy = F64Ty;
        Cur = emitLoad(OuterSlot, MTy, L);
      }
      /* Spill slot mirrors the outer slot's type so call-site reloads
       * see the same shape. */
      mlir::Value SpillSlot;
      if (MTy == F64Ty) {
        SpillSlot = emitAlloc(TC.scalar(Dtype::Double), Bnd->Name, L);
      } else {
        /* Emit a raw matlab.alloc with the concrete MLIR type; other
         * paths can't synthesize the Sema Type* for arbitrary tensors. */
        mlir::NamedAttribute NA(
            mlir::StringAttr::get(&MCtx, "name"),
            mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Bnd->Name)));
        SpillSlot = emitUnreg("matlab.alloc", {}, MTy, L, {NA});
      }
      emitStore(Cur, SpillSlot, L);
      CaptureVals.push_back(Cur);
      CaptureTys.push_back(MTy);
      CaptureSpills.push_back(SpillSlot);
    }
    PendingCaptures[&A] = CaptureSpills;

    mlir::OpBuilder::InsertionGuard G(B);
    /* Block args: [captures (typed per capture)..., params (f64)...]. */
    llvm::SmallVector<mlir::Type> ArgTys;
    ArgTys.append(CaptureTys.begin(), CaptureTys.end());
    for (size_t i = 0; i < A.Params.size(); ++i) ArgTys.push_back(F64Ty);
    llvm::SmallVector<mlir::Location> ArgLocs(Captures.size() +
                                              A.Params.size(), L);
    mlir::Operation *Op = emitUnregOp("matlab.make_anon", CaptureVals, {RT},
                                      L, {PA}, /*NumRegions=*/1);
    mlir::Block *Body = B.createBlock(&Op->getRegion(0),
                                      Op->getRegion(0).end(),
                                      ArgTys, ArgLocs);
    B.setInsertionPointToEnd(Body);

    /* Swap in a fresh Slots map for the body. Captures AND params both
     * get inner spill slots whose type is the block arg's type.
     * Captures reuse the block arg's type (tensor or f64); params are
     * f64 (scalar-only for v1). */
    auto Saved = Slots;
    Slots.clear();
    for (size_t i = 0; i < Captures.size(); ++i) {
      mlir::Type MTy = Body->getArgument(i).getType();
      mlir::Value Slot;
      if (MTy == F64Ty) {
        Slot = emitAlloc(TC.scalar(Dtype::Double), Captures[i]->Name, L);
      } else {
        mlir::NamedAttribute NA(
            mlir::StringAttr::get(&MCtx, "name"),
            mlir::FlatSymbolRefAttr::get(
                &MCtx, std::string(Captures[i]->Name)));
        Slot = emitUnreg("matlab.alloc", {}, MTy, L, {NA});
      }
      Slots[Captures[i]] = Slot;
      emitStore(Body->getArgument(i), Slot, L);
    }
    for (size_t i = 0; i < A.ParamRefs.size(); ++i) {
      Binding *Bnd = A.ParamRefs[i];
      if (!Bnd) continue;
      mlir::Value Slot = emitAlloc(TC.scalar(Dtype::Double), Bnd->Name, L);
      Slots[Bnd] = Slot;
      emitStore(Body->getArgument(Captures.size() + i), Slot, L);
    }

    mlir::Value V = A.Body ? lowerExpr(*A.Body) : mlir::Value{};
    llvm::SmallVector<mlir::Value, 1> Ys;
    if (V) Ys.push_back(V);
    emitUnregOp("matlab.yield", Ys, {}, L);

    Slots = std::move(Saved);
    return Op->getResult(0);
  }
  case NodeKind::FuncHandle: {
    auto &F = static_cast<const FuncHandle &>(E);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, std::string(F.Name)));
    return emitUnreg("matlab.make_handle", {}, RT, L, {Cal});
  }
  default:
    return emitUnreg("matlab.undef", {}, RT, L);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

mlir::ModuleOp lowerToMLIR(Context &Ctx,
                           TypeContext &TC,
                           DiagnosticEngine &Diag,
                           const TranslationUnit &TU,
                           const SourceManager *SM,
                           bool ReplMode,
                           bool DebugMode) {
  Lowerer L(Ctx.get(), TC, Diag, SM, ReplMode, DebugMode);
  return L.lower(TU);
}

void printModule(std::ostream &OS, mlir::ModuleOp M) {
  std::string S;
  llvm::raw_string_ostream RS(S);
  M.print(RS);
  RS.flush();
  OS << S;
}

} // namespace mlirgen
} // namespace matlab
