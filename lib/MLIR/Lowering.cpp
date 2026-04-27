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
  /* Bindings assigned from a CellLiteral — tracked so calls like
   * numel(C) / length(C) / iscell(C) can dispatch to the matlab_cell_*
   * runtime entries instead of the matrix path. */
  std::unordered_set<Binding *> CellBindings;
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
  std::string CurFnName;
  /* Declared arity of the currently-lowered function — used to fold
   * references to the `nargin` / `nargout` builtins into compile-time
   * constants. Per-call-site arity would need LHS-threaded
   * monomorphisation; this v1 uses the declared counts. */
  unsigned CurFnNargin = 0;
  unsigned CurFnNargout = 0;
  mlir::Value getOrCreateSlot(Binding *Bnd, const Type *T, llvm::StringRef N,
                              mlir::Location L);
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
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "matlab_struct_new"));
    mlir::Value NewPtr = emitUnreg("matlab.call_builtin", {}, PtrTy, L, {Cal});
    emitStore(NewPtr, Slot, L);
  }
  return Slot;
}

mlir::Value Lowerer::resolveStructBase(const Expr *E, mlir::Location L) {
  if (!E) return {};
  if (auto *N = dynamic_cast<const NameExpr *>(E)) {
    if (!N->Ref) return {};
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
   * We always call matlab_ws_get_mat (not _get_f64) because at this
   * point Sema usually can't tell whether the workspace cell holds
   * a scalar or a matrix — the runtime auto-boxes a stored scalar
   * into a 1x1 matrix on _get_mat, so the single PtrTy return type
   * covers both cases. Downstream arithmetic sees a matrix and
   * takes the runtime matrix-op path; `disp` of a 1x1 matrix
   * prints the scalar value. A small overhead for the scalar case
   * but negligible for a human-paced REPL. */
  if (ReplMode && InScriptBody && Bnd->Kind == BindingKind::Var &&
      Slots.find(Bnd) == Slots.end()) {
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    mlir::Value NameV = emitFieldNameChar(Bnd->Name, L);
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

  /* Register every classdef's name with the runtime so the DAP server
   * can resolve a matlab_obj's class_id back to a printable class
   * name. Emitted as the first thing in the script body so the table
   * is populated before any constructor runs. No-op outside
   * DebugMode — the registry is dead weight in production builds. */
  if (DebugMode && CurTU) {
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
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *O = F.OutputRefs[i];
    if (IsCtor && i == 0) {
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
    emitStore(Entry->getArgument(i), Slot, loc(F.Range));
  }
  // Pre-allocate output slots.
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bnd = F.OutputRefs[i];
    if (!Bnd) continue;
    bool IsCtorObj = IsCtor && i == 0;
    mlir::Value Slot;
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
   * then compare against zero to materialise an i1. */
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(CT)) {
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
    mlir::Value V = lowerExpr(*E.E);
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
    emitUnregOp("matlab.call_builtin", {V},
                {mlir::NoneType::get(&MCtx)}, loc(E.Range), {DispCal});
    /* In REPL mode, bind non-named-expression results to `ans` in the
     * workspace so subsequent inputs can reference them. A bare
     * NameExpr already has its value stored under its own name, so
     * no extra write is needed for that case. */
    if (ReplMode && InScriptBody && E.E->Kind != NodeKind::NameExpr) {
      mlir::Value AnsName = emitFieldNameChar("ans", loc(E.Range));
      bool IsMat = V.getType() == PtrTy ||
                    mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(V.getType());
      llvm::StringRef Callee = IsMat ? "matlab_ws_set_mat"
                                      : "matlab_ws_set_f64";
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
    bool RhsIsCellLit = A.RHS && A.RHS->Kind == NodeKind::CellLiteral;
    /* Track string-typed bindings (from "..." literals or string ops)
     * so `+` / disp / strlen / isstring can dispatch correctly.
     * Recursively sees through nested `+` chains so `u = s + " " + t`
     * is recognised. */
    std::function<bool(const Expr *)> isStringExpr =
        [this, &isStringExpr](const Expr *E) -> bool {
      if (!E) return false;
      if (E->Kind == NodeKind::StringLiteral) return true;
      if (auto *N = dynamic_cast<const NameExpr *>(E))
        return N->Ref && StringBindings.count(N->Ref) > 0;
      if (auto *Bi = dynamic_cast<const BinaryOpExpr *>(E))
        if (Bi->Op == BinOp::Add)
          return isStringExpr(Bi->LHS) && isStringExpr(Bi->RHS);
      return false;
    };
    bool RhsIsString = isStringExpr(A.RHS);
    /* String-producing builtins also turn the LHS into a string
     * binding: fgetl, sprintf, num2str, upper, lower, strtrim,
     * strrep, strcat all return matlab_string*, so subsequent
     * disp(s) / strlen(s) / isstring(s) route through the string
     * runtime. */
    if (!RhsIsString && A.RHS && A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *CX = static_cast<const CallOrIndex *>(A.RHS);
      if (auto *NX = dynamic_cast<const NameExpr *>(CX->Callee)) {
        if (NX->Ref && NX->Ref->Kind == BindingKind::Builtin) {
          auto N = NX->Name;
          if (N == "fgetl" || N == "sprintf" || N == "num2str" ||
              N == "upper" || N == "lower" || N == "strtrim" ||
              N == "strrep" || N == "strcat" ||
              N == "bin" || N == "hex" || N == "dec")
            RhsIsString = true;
        }
      }
    }

    /* Track 3-D bindings: RHS is a call to zeros/ones with 3 args. */
    bool RhsIsThreeD = false;
    if (A.RHS && A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *C = static_cast<const CallOrIndex *>(A.RHS);
      if (auto *N = dynamic_cast<const NameExpr *>(C->Callee)) {
        if (C->Args.size() == 3 && N->Ref &&
            N->Ref->Kind == BindingKind::Builtin &&
            (N->Name == "zeros" || N->Name == "ones"))
          RhsIsThreeD = true;
      }
    }

    /* Multi-return call: [V, D] = eig(A). If the LHS arity is > 1 and
     * the RHS is a call to a builtin that has a multi-return variant,
     * emit a matlab.call_builtin with N result types and a nargout
     * attribute so LowerTensorOps can dispatch to the right runtime
     * entry. Each LHS then gets its own result. */
    if (A.LHS.size() > 1 && A.RHS &&
        A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *C = static_cast<const CallOrIndex *>(A.RHS);
      auto *Callee = dynamic_cast<const NameExpr *>(C->Callee);
      if (Callee && Callee->Ref &&
          Callee->Ref->Kind == BindingKind::Builtin) {
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
          /* [V, D] = eig / [Q, R] = qr / [L, U] = lu / [U, S, V] = svd —
           * all ptr (matrix) results. */
          else if ((CN == "eig" || CN == "qr" || CN == "lu" ||
                    CN == "svd") && A.LHS.size() >= 2)
            Rtys.assign(A.LHS.size(), PtrTy);
          /* [row, col] = ind2sub(sz, i) — scalar f64s. */
          else if (CN == "ind2sub" && A.LHS.size() == 2)
            Rtys.assign(A.LHS.size(), F64);
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
    }

    mlir::Value Rhs = A.RHS ? lowerExpr(*A.RHS) : mlir::Value{};
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
      for (const Expr *L : A.LHS) {
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) HandleBindings[N->Ref] = Caps;
      }
    }
    if (RhsIsCellLit) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) CellBindings.insert(N->Ref);
    }
    if (RhsIsString) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) StringBindings.insert(N->Ref);
    }
    if (RhsIsThreeD) {
      for (const Expr *L : A.LHS)
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) ThreeDBindings.insert(N->Ref);
    }
    for (const Expr *L : A.LHS) if (L) lowerLValueStore(*L, Rhs);
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
      emitUnregOp("matlab.call_builtin", {Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
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
    mlir::Operation *ForOp = emitUnregOp(
        F.IsParfor ? "matlab.parfor" : "matlab.for",
        ForOperands, {}, loc(F.Range), {VarAttr}, /*NumRegions=*/1);
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
      }
      emitUnregOp("matlab.call_builtin", {IdV, Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(N.Range), Attrs);
      return;
    }
    /* REPL script-level Var writes route through matlab_ws_set_*.
     * Like loadBinding above, skip the workspace path when a local
     * slot already exists (e.g. for-loop induction variable) — the
     * slot is the canonical store site during that loop. */
    if (ReplMode && InScriptBody && N.Ref->Kind == BindingKind::Var && Rhs &&
        Slots.find(N.Ref) == Slots.end()) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::Value NameV = emitFieldNameChar(N.Name, loc(N.Range));
      bool IsMat = (Rhs.getType() == PtrTy ||
                    mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(Rhs.getType()));
      /* When the Sema-inferred class for this binding is set, the
       * pointer is a matlab_obj* (class instance) rather than a
       * matlab_mat*. Route to matlab_ws_set_obj so the workspace
       * tracks kind=2; the DAP server uses that to render the
       * variable as `1x1 ClassName` and to expose its properties. */
      bool IsObj = N.Ref->PinnedClass != nullptr && IsMat;
      llvm::StringRef Callee = IsObj ? "matlab_ws_set_obj"
                                     : (IsMat ? "matlab_ws_set_mat"
                                              : "matlab_ws_set_f64");
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
      if (Base) SubscriptCtx.push_back({Base, (int64_t)(a + 1)});
      Os.push_back(lowerExpr(*Arg));
      if (Base) SubscriptCtx.pop_back();
    }
    if (Rhs) Os.push_back(Rhs);
    /* 3-D scalar store: A(i, j, k) = v on a matlab_mat3 binding
     * routes to matlab_subscript3_store. */
    if (C.Args.size() == 3 && Rhs) {
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        if (NE->Ref && ThreeDBindings.count(NE->Ref)) {
          mlir::NamedAttribute Cal3(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_subscript3_store"));
          emitUnregOp("matlab.call_builtin", Os,
                      {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal3});
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
    /* C{i} = Rhs. Evaluate the cell ptr, the index, and Rhs; route
     * to matlab_cell_set_f64 or matlab_cell_set_mat based on value
     * kind. Single 1-D index for v1. */
    auto &C = static_cast<const CellIndex &>(LHS);
    if (C.Args.size() != 1 || !C.Callee) return;
    mlir::Value Cell = lowerExpr(*C.Callee);
    mlir::Value Idx = lowerExpr(*C.Args[0]);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    bool IsMat = Rhs && (Rhs.getType() == PtrTy ||
                         mlir::isa<mlir::RankedTensorType,
                                   mlir::UnrankedTensorType>(Rhs.getType()));
    llvm::StringRef Callee = IsMat ? "matlab_cell_set_mat"
                                    : "matlab_cell_set_f64";
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, Callee));
    emitUnregOp("matlab.call_builtin", {Cell, Idx, Rhs},
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
     * instead so class_id + property table is preserved. */
    auto &F = static_cast<const FieldAccess &>(LHS);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
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
      mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
      llvm::StringRef Callee = (Rhs && Rhs.getType() == PtrTy)
          ? "matlab_obj_set_mat" : "matlab_obj_set_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      emitUnregOp("matlab.call_builtin", {Obj, NameV, Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(F.Range), {Cal});
      return;
    }
    mlir::Value SPtr = resolveStructBase(F.Base, loc(F.Range));
    if (!SPtr) return;
    mlir::Value NameV = emitFieldNameChar(F.Field, loc(F.Range));
    llvm::StringRef Callee = (Rhs && Rhs.getType() == PtrTy)
        ? "matlab_struct_set_mat"
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
      mlir::NamedAttribute VA(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx),
                                (double)Dim));
      mlir::Value DimV = emitUnreg("matlab.const_float", {},
                                   mlir::Float64Type::get(&MCtx), L, {VA});
      return emitUnreg("matlab.end", {Base, DimV}, RT, L);
    }
    return emitUnreg("matlab.end", {}, RT, L);
  }
  case NodeKind::ColonExpr:
    return emitUnreg("matlab.colon", {}, RT, L);
  case NodeKind::BinaryOp: {
    auto &Bi = static_cast<const BinaryOpExpr &>(E);
    /* String concatenation: `"a" + "b"` or `s1 + s2` where both
     * operands are known strings (or themselves string-producing
     * subexpressions). Detect BEFORE lowering operands so we pick
     * the right runtime call and attach the ptr result type up
     * front (the generic matlab.add path would produce f64). */
    std::function<bool(const Expr *)> isStringOperand =
        [this, &isStringOperand](const Expr *X) -> bool {
      if (!X) return false;
      if (X->Kind == NodeKind::StringLiteral) return true;
      if (auto *N = dynamic_cast<const NameExpr *>(X))
        return N->Ref && StringBindings.count(N->Ref) > 0;
      if (auto *Bi2 = dynamic_cast<const BinaryOpExpr *>(X))
        if (Bi2->Op == BinOp::Add)
          return isStringOperand(Bi2->LHS) && isStringOperand(Bi2->RHS);
      return false;
    };
    if (Bi.Op == BinOp::Add &&
        isStringOperand(Bi.LHS) && isStringOperand(Bi.RHS)) {
      auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
      mlir::Value LHS = lowerExpr(*Bi.LHS);
      mlir::Value RHS = lowerExpr(*Bi.RHS);
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
    auto pinnedFromExpr = [](const Expr *X) -> const ClassDef * {
      if (auto *NE = dynamic_cast<const NameExpr *>(X))
        if (NE->Ref && NE->Ref->PinnedClass) return NE->Ref->PinnedClass;
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
          mlir::Value LHS = Bi.LHS ? lowerExpr(*Bi.LHS) : mlir::Value{};
          mlir::Value RHS = Bi.RHS ? lowerExpr(*Bi.RHS) : mlir::Value{};
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
    mlir::Value LHS = Bi.LHS ? lowerExpr(*Bi.LHS) : mlir::Value{};
    mlir::Value RHS = Bi.RHS ? lowerExpr(*Bi.RHS) : mlir::Value{};
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
    return emitUnreg(binOpName(Bi.Op), {LHS, RHS}, ResTy, L);
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(E);
    mlir::Value A = U.Operand ? lowerExpr(*U.Operand) : mlir::Value{};
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
    if (C.Resolved == CallKind::Call) {
      auto *N = dynamic_cast<const NameExpr *>(C.Callee);
      auto PtrTyConst = mlir::LLVM::LLVMPointerType::get(&MCtx);
      /* Constructor call: `ClassName(args)` where ClassName resolves to
       * a user classdef. Route to the emitted `ClassName__ClassName`
       * function, returning a matlab_obj*. If the class has no explicit
       * constructor, emit `matlab_obj_new(class_id)` directly and skip
       * arg-binding — MATLAB's implicit default constructor is no-arg. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Class &&
          N->Ref->ClassDef) {
        const ClassDef *CD = N->Ref->ClassDef;
        bool HasCtor = false;
        for (const Function *Mth : CD->Methods)
          if (Mth && Mth->Name == CD->Name) { HasCtor = true; break; }
        if (HasCtor) {
          llvm::SmallVector<mlir::Value, 4> Args;
          for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
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
        const ClassDef *PCls = nullptr;
        if (auto *BN = dynamic_cast<const NameExpr *>(FA->Base))
          if (BN->Ref && BN->Ref->PinnedClass) PCls = BN->Ref->PinnedClass;
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
            return emitUnreg("matlab.call", Args, RT, L, {Cal});
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
              return emitUnreg("matlab.call", Args, RT, L, {Cal});
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
              return emitUnreg("matlab.call", Args, RT, L, {Cal});
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
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_persistent_isempty"));
            return emitUnreg("matlab.call_builtin", {IdV}, F64, L, {Cal});
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
        else if (auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]))
          IsStr = AN->Ref && StringBindings.count(AN->Ref) > 0;
        else if (auto *CC = dynamic_cast<const CallOrIndex *>(C.Args[0])) {
          if (auto *CN = dynamic_cast<const NameExpr *>(CC->Callee)) {
            if (CN->Ref && CN->Ref->Kind == BindingKind::Builtin) {
              auto Nm = CN->Name;
              if (Nm == "fgetl" || Nm == "sprintf" || Nm == "num2str" ||
                  Nm == "upper" || Nm == "lower" || Nm == "strtrim" ||
                  Nm == "strrep" || Nm == "strcat" ||
                  Nm == "bin" || Nm == "hex" || Nm == "dec")
                IsStr = true;
            }
          }
        }
        if (IsStr) {
          mlir::Value V = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_disp"));
          return emitUnreg("matlab.call_builtin", {V},
                           mlir::NoneType::get(&MCtx), L, {Cal});
        }
      }
      /* strlen(s) on a string binding -> matlab_string_len. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "strlen" && C.Args.size() == 1) {
        auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]);
        if (AN && AN->Ref && StringBindings.count(AN->Ref)) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::Value V = lowerExpr(*C.Args[0]);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_string_len"));
          return emitUnreg("matlab.call_builtin", {V}, F64, L, {Cal});
        }
      }
      /* isstring(x) compile-time fold. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "isstring" && C.Args.size() == 1) {
        auto *AN = dynamic_cast<const NameExpr *>(C.Args[0]);
        auto F64 = mlir::Float64Type::get(&MCtx);
        double Val = 0.0;
        if (C.Args[0]->Kind == NodeKind::StringLiteral) Val = 1.0;
        else if (AN && AN->Ref && StringBindings.count(AN->Ref)) Val = 1.0;
        return mlir::arith::ConstantOp::create(
            B, L, F64, mlir::FloatAttr::get(F64, Val));
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
      /* size(A, dim) / numel(A) / ndims(A) on a 3-D binding route to
       * the matlab_mat3 runtime; the 2-D variants treat the descriptor
       * as a matlab_mat* and would read wrong fields. */
      if (N && N->Ref && N->Ref->Kind == BindingKind::Builtin &&
          N->Name == "size" && C.Args.size() == 2) {
        if (auto *ArgN = dynamic_cast<const NameExpr *>(C.Args[0]))
          if (ArgN->Ref && ThreeDBindings.count(ArgN->Ref)) {
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
          if (ArgN->Ref && ThreeDBindings.count(ArgN->Ref)) {
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
          if (ArgN->Ref && ThreeDBindings.count(ArgN->Ref)) {
            auto F64 = mlir::Float64Type::get(&MCtx);
            mlir::Value A = lowerExpr(*C.Args[0]);
            mlir::NamedAttribute Cal(
                mlir::StringAttr::get(&MCtx, "callee"),
                mlir::StringAttr::get(&MCtx, "matlab_ndims3"));
            return emitUnreg("matlab.call_builtin", {A}, F64, L, {Cal});
          }
      }
      llvm::SmallVector<mlir::Value, 4> Args;
      for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
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
            "isempty", "isequal", "rank", "sub2ind", "mod", "rem",
            "fix", "round", "floor", "ceil",
          };
          static const llvm::StringSet<> PtrRet = {
            "zeros", "ones", "eye", "magic", "rand", "randn",
            "sum", "prod", "mean", "min", "max", "cumsum", "cumprod",
            "sort", "sortrows", "unique", "ismember", "setdiff",
            "intersect", "union", "horzcat", "vertcat", "kron",
            "chol", "pinv", "permute", "squeeze", "flip", "fliplr",
            "flipud", "rot90", "size", "transpose", "ctranspose",
            "diag", "reshape", "repmat", "inv", "svd", "eig",
            "find", "ind2sub", "linspace",
            /* Complex: all return a matrix descriptor (matlab_mat* or
             * matlab_mat_c*), uniformly ptr at MLIR level. */
            "conj", "real", "imag", "angle",
            "fft", "ifft", "fft2", "ifft2",
          };
          if (F64Ret.contains(N->Name)) ResTy = F64;
          else if (PtrRet.contains(N->Name)) ResTy = PtrTy;
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

    mlir::Value Arr = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
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
      if (!IsHandleCall)
        SubscriptCtx.push_back({Arr, (int64_t)(a + 1)});
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
          // Per-element subscript: every index is an f64 scalar.
          bool AllScalarF64 = true;
          for (size_t i = 1; i < Idx.size(); ++i)
            if (Idx[i].getType() != F64) { AllScalarF64 = false; break; }
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

    /* 3-D scalar subscript: A(i, j, k) where A is tracked as a
     * matlab_mat3 binding routes to matlab_subscript3_s. */
    if (C.Args.size() == 3) {
      if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
        if (NE->Ref && ThreeDBindings.count(NE->Ref)) {
          auto F64 = mlir::Float64Type::get(&MCtx);
          mlir::NamedAttribute Cal(
              mlir::StringAttr::get(&MCtx, "callee"),
              mlir::StringAttr::get(&MCtx, "matlab_subscript3_s"));
          return emitUnreg("matlab.call_builtin", Idx, F64, L, {Cal});
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
    /* C{i} read — routes to matlab_cell_get_f64 by default, or
     * matlab_cell_get_mat when Sema concretely says matrix. Single
     * 1-D index for v1. */
    auto &C = static_cast<const CellIndex &>(E);
    if (C.Args.size() != 1)
      return emitUnreg("matlab.undef", {}, RT, L);
    mlir::Value Arr = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
    mlir::Value Idx = lowerExpr(*C.Args[0]);
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    bool WantMat = mlir::isa<mlir::RankedTensorType,
                              mlir::UnrankedTensorType>(RT);
    llvm::StringRef Callee = WantMat ? "matlab_cell_get_mat"
                                      : "matlab_cell_get_f64";
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, Callee));
    mlir::Type ResTy = WantMat ? (mlir::Type)PtrTy : (mlir::Type)F64;
    return emitUnreg("matlab.call_builtin", {Arr, Idx}, ResTy, L, {Cal});
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
    const ClassDef *PinnedCls = nullptr;
    if (auto *BN = dynamic_cast<const NameExpr *>(F.Base))
      if (BN->Ref && BN->Ref->PinnedClass) PinnedCls = BN->Ref->PinnedClass;
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
      llvm::StringRef Callee = WantMat ? "matlab_obj_get_mat"
                                        : "matlab_obj_get_f64";
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, Callee));
      mlir::Type ResTy = WantMat ? (mlir::Type)PtrTy : (mlir::Type)F64;
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
     * v1: 1-D only (flattens multi-row literals into a single row).
     * Kind is picked from each element's MLIR type at the call site:
     * ptr -> matlab_cell_set_mat, else -> matlab_cell_set_f64. */
    auto &M = static_cast<const CellLiteral &>(E);
    auto F64 = mlir::Float64Type::get(&MCtx);
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(&MCtx);
    llvm::SmallVector<mlir::Value, 8> Elems;
    for (auto &R : M.Rows)
      for (const Expr *El : R)
        if (El) Elems.push_back(lowerExpr(*El));
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
      mlir::Value V = Elems[i];
      /* Tensor and ptr both route to set_mat — a literal matrix is
       * tensor-typed at lowering time and gets retyped to ptr by
       * LowerTensorOps later. */
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
      /* Prefer the outer slot's concrete MLIR type: Sema-level
       * InferredType is often still `any` for script-scope matrix
       * assignments even though the slot was allocated with a real
       * tensor type by the matrix-literal store earlier. */
      mlir::Value OuterSlot = getOrCreateSlot(Bnd, BTy, Bnd->Name, L);
      mlir::Type MTy = OuterSlot.getType();
      if (mlir::isa<mlir::NoneType>(MTy)) MTy = F64Ty;
      mlir::Value Cur = emitLoad(OuterSlot, MTy, L);
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
