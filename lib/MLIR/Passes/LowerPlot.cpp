// Rewrites matlab.call_builtin ops whose callee names a function from the
// matlab_plot runtime (figure / plot / title / surf / saveas / etc.) into
// llvm.call ops against the corresponding matlab_* C-ABI symbol.
//
// Runs after LowerTensorOps (so matrix args are already PtrTy SSA values
// pointing at malloc'd matlab_mat descriptors) and before LowerIO (which
// handles the disp / fprintf / input set).
//
// String literal handling mirrors LowerIO: matlab.const_char operands are
// materialised into LLVM globals + (ptr, i64-length) pairs.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

LLVM::LLVMFuncOp getOrInsertRuntimeFunc(OpBuilder &B, ModuleOp M,
                                        StringRef Name, Type ResultTy,
                                        ArrayRef<Type> ArgTypes) {
  if (auto Existing = M.lookupSymbol<LLVM::LLVMFuncOp>(Name)) return Existing;
  OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(M.getBody());
  auto FnTy = LLVM::LLVMFunctionType::get(ResultTy, ArgTypes);
  auto F = LLVM::LLVMFuncOp::create(B, M.getLoc(), Name, FnTy);
  F.setLinkage(LLVM::Linkage::External);
  return F;
}

/* String global registry — duplicated here from LowerIO.cpp to avoid a
 * cross-pass header. The dedupe-by-value behavior matches. */
struct StringGlobals {
  ModuleOp M;
  OpBuilder &B;
  int Counter = 0;

  StringGlobals(ModuleOp M, OpBuilder &B) : M(M), B(B) {}

  LLVM::GlobalOp getOrCreate(StringRef Text) {
    for (auto G : M.getOps<LLVM::GlobalOp>()) {
      if (!G.getConstant()) continue;
      auto Attr = mlir::dyn_cast_or_null<StringAttr>(G.getValueAttr());
      if (Attr && Attr.getValue() == Text) return G;
    }
    OpBuilder::InsertionGuard Guard(B);
    B.setInsertionPointToStart(M.getBody());
    MLIRContext *Ctx = M.getContext();
    auto ArrayTy = LLVM::LLVMArrayType::get(
        IntegerType::get(Ctx, 8),
        static_cast<unsigned>(Text.size()));
    std::string Name = ("__matlab_plot_str" + llvm::Twine(Counter++)).str();
    auto G = LLVM::GlobalOp::create(
        B, M.getLoc(), ArrayTy, /*isConstant=*/true,
        LLVM::Linkage::Internal, Name,
        StringAttr::get(Ctx, Text));
    return G;
  }
};

std::optional<std::pair<Value, int64_t>>
materializeStringArg(Value V, OpBuilder &B, StringGlobals &Strings) {
  /* Accept both single-quoted char literals (matlab.const_char) and
   * double-quoted string literals (matlab.const_str). MATLAB's command
   * syntax — e.g. `grid on` parsed as `grid('on')` — produces const_str. */
  Operation *Def = V.getDefiningOp();
  if (!isMatlabOp(Def, "matlab.const_char") &&
      !isMatlabOp(Def, "matlab.const_str")) return std::nullopt;
  auto ValAttr = Def->getAttrOfType<StringAttr>("value");
  if (!ValAttr) return std::nullopt;
  StringRef Text = ValAttr.getValue();
  auto G = Strings.getOrCreate(Text);

  OpBuilder::InsertionGuard Guard(B);
  B.setInsertionPointAfter(Def);
  MLIRContext *Ctx = B.getContext();
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  Value Addr = LLVM::AddressOfOp::create(B, Def->getLoc(), PtrTy, G.getSymName());
  return std::make_pair(Addr, (int64_t)Text.size());
}

/* MATLAB Name/Value properties recognised on series-producing calls. The
 * codegen strips trailing (string, value) pairs whose name is in this
 * set and emits a follow-up matlab_set_series_* call for each. Names are
 * compared case-insensitively to match MATLAB. */
bool isKnownSeriesProp(StringRef Name) {
  /* Lowercased once via case-insensitive equals_insensitive. */
  static const llvm::StringSet<> Props = {
    "linewidth", "linestyle", "color", "marker", "markersize",
    "markerfacecolor", "markeredgecolor", "displayname",
  };
  return Props.contains(Name.lower());
}

/* Names this pass owns. Anything else is left for downstream passes. */
const llvm::StringSet<> &plotBuiltins() {
  static const llvm::StringSet<> S = {
    "figure", "gcf", "close",
    "plot", "plot3", "bar", "scatter", "stem", "stairs", "area",
    "errorbar", "histogram",
    "imshow", "imagesc", "pcolor", "surf", "mesh", "contour",
    "title", "xlabel", "ylabel", "zlabel", "text", "legend",
    "xline", "yline",
    "xticks", "yticks", "xticklabels", "yticklabels",
    "yyaxis", "contourf", "quiver",
    "colorbar", "colormap",
    "grid", "hold", "axis", "box", "xlim", "ylim", "view",
    "loglog", "semilogx", "semilogy",
    "subplot", "saveas", "print",
    /* pdeplot3D — unstructured 3-D triangle-mesh painter. */
    "pdeplot3d", "pdeplot3D",
    "pdeplot3d_deformation", "pdeplot3d_deform_scale",
    /* pdeplot — 2-D unstructured triangle painter. */
    "pdeplot",
  };
  return S;
}

/* Read a literal string operand (matlab.const_char or matlab.const_str)
 * back as plain text. Returns nullopt for non-literal operands. */
std::optional<StringRef> getLiteralString(Value V) {
  Operation *Def = V.getDefiningOp();
  if (!isMatlabOp(Def, "matlab.const_char") &&
      !isMatlabOp(Def, "matlab.const_str")) return std::nullopt;
  auto Attr = Def->getAttrOfType<StringAttr>("value");
  if (!Attr) return std::nullopt;
  return Attr.getValue();
}

/* Returns the count of trailing (string, value) Name/Value pairs whose
 * Name is a known series property. Iterates from the end inward. */
unsigned countTrailingPropPairs(Operation *Op) {
  unsigned N = Op->getNumOperands();
  unsigned pairs = 0;
  while (N >= 2 + pairs * 2) {
    unsigned name_idx = N - 2 - pairs * 2;
    auto S = getLiteralString(Op->getOperand(name_idx));
    if (!S || !isKnownSeriesProp(*S)) break;
    pairs++;
  }
  return pairs;
}

struct Helper {
  ModuleOp M;
  OpBuilder &B;
  StringGlobals &Strings;
  Type I64, F64, PtrTy, VoidTy;

  Helper(ModuleOp M, OpBuilder &B, StringGlobals &Strings)
      : M(M), B(B), Strings(Strings) {
    auto *Ctx = B.getContext();
    I64 = IntegerType::get(Ctx, 64);
    F64 = Float64Type::get(Ctx);
    PtrTy = LLVM::LLVMPointerType::get(Ctx);
    VoidTy = LLVM::LLVMVoidType::get(Ctx);
  }

  Value i64Const(Location L, int64_t v) {
    return LLVM::ConstantOp::create(B, L, I64, B.getI64IntegerAttr(v));
  }
  Value f64Const(Location L, double v) {
    return LLVM::ConstantOp::create(B, L, F64, B.getF64FloatAttr(v));
  }

  /* Coerce a value to f64 if it's an integer; pass through if already F64;
   * fail for unexpected types. */
  std::optional<Value> toF64(Location L, Value V) {
    if (V.getType() == F64) return V;
    if (auto IT = mlir::dyn_cast<IntegerType>(V.getType())) {
      (void)IT;
      return (Value)arith::SIToFPOp::create(B, L, F64, V);
    }
    return std::nullopt;
  }

  /* Emit an LLVM call to a runtime function returning void. */
  void callVoid(Location L, StringRef Name, ArrayRef<Type> ArgTys,
                ValueRange Args) {
    auto Fn = getOrInsertRuntimeFunc(B, M, Name, VoidTy, ArgTys);
    LLVM::CallOp::create(B, L, Fn, Args);
  }
  /* Emit an LLVM call returning a single value of type Ret. */
  Value callRet(Location L, StringRef Name, Type Ret,
                ArrayRef<Type> ArgTys, ValueRange Args) {
    auto Fn = getOrInsertRuntimeFunc(B, M, Name, Ret, ArgTys);
    auto C = LLVM::CallOp::create(B, L, Fn, Args);
    return C.getResult();
  }
};

/* Emit matlab_set_series_* runtime calls for each trailing (Name, Value)
 * pair on Op. Insertion point should already be after the main call.
 * Operates on the absolute operand range [pos_count, N) in pairs. */
void emitPropSetters(Operation *Op, unsigned pos_count,
                     Helper &H, StringGlobals &S) {
  Location L = Op->getLoc();
  unsigned N = Op->getNumOperands();
  for (unsigned i = pos_count; i + 1 < N; i += 2) {
    auto Name = getLiteralString(Op->getOperand(i));
    if (!Name) continue;
    Value Val = Op->getOperand(i + 1);
    std::string lower = Name->lower();
    StringRef name = lower;

    if (name == "linewidth" || name == "markersize") {
      auto F = (Val.getType() == H.F64)
                   ? std::make_optional(Val)
                   : H.toF64(L, Val);
      if (!F) continue;
      H.callVoid(L, name == "linewidth" ? "matlab_set_series_linewidth"
                                        : "matlab_set_series_markersize",
                 {H.F64}, {*F});
    }
    else if (name == "linestyle" || name == "marker" ||
             name == "displayname") {
      auto Pair = materializeStringArg(Val, H.B, S);
      if (!Pair) continue;
      auto [Ptr, Len] = *Pair;
      StringRef rt = (name == "linestyle")  ? "matlab_set_series_linestyle"
                  : (name == "marker")      ? "matlab_set_series_marker"
                  :                            "matlab_set_series_displayname";
      H.callVoid(L, rt, {H.PtrTy, H.I64}, {Ptr, H.i64Const(L, Len)});
    }
    else if (name == "color" || name == "markerfacecolor" ||
             name == "markeredgecolor") {
      /* Color: accept either a single-letter string ('r'/'g'/...) or an
       * RGB triplet matlab_mat. Foreground color only — face/edge map
       * to the same setter for now. */
      if (auto Lit = getLiteralString(Val)) {
        if (Lit->size() == 1) {
          /* Translate single-letter color to RGB and call set_series_color. */
          double r=0, g=0, b=0;
          switch (Lit->front()) {
            case 'r': r=0.85; g=0.33; b=0.10; break;
            case 'g': r=0.47; g=0.67; b=0.19; break;
            case 'b': r=0.00; g=0.45; b=0.74; break;
            case 'c': r=0.30; g=0.75; b=0.93; break;
            case 'm': r=0.49; g=0.18; b=0.56; break;
            case 'y': r=0.93; g=0.69; b=0.13; break;
            case 'k': r=0.00; g=0.00; b=0.00; break;
            case 'w': r=1.00; g=1.00; b=1.00; break;
            default: continue;
          }
          H.callVoid(L, "matlab_set_series_color",
                     {H.F64, H.F64, H.F64},
                     {H.f64Const(L, r), H.f64Const(L, g), H.f64Const(L, b)});
        }
      } else if (Val.getType() == H.PtrTy) {
        H.callVoid(L, "matlab_set_series_color_mat", {H.PtrTy}, {Val});
      }
    }
  }
}

/* Each rewriter returns true on success. The caller erases the op. */
bool rewriteSimpleStringArg(Operation *Op, Helper &H, StringGlobals &S,
                            StringRef RuntimeName) {
  if (Op->getNumOperands() != 1) return false;
  auto Pair = materializeStringArg(Op->getOperand(0), H.B, S);
  if (!Pair) return false;
  auto [Ptr, Len] = *Pair;
  H.B.setInsertionPoint(Op);
  H.callVoid(Op->getLoc(), RuntimeName, {H.PtrTy, H.I64},
             {Ptr, H.i64Const(Op->getLoc(), Len)});
  return true;
}

bool rewriteVoidNoArg(Operation *Op, Helper &H, StringRef RuntimeName) {
  if (Op->getNumOperands() != 0) return false;
  H.B.setInsertionPoint(Op);
  H.callVoid(Op->getLoc(), RuntimeName, {}, {});
  return true;
}

bool rewritePtrReturnNoArg(Operation *Op, Helper &H, StringRef RuntimeName) {
  if (Op->getNumOperands() != 0) return false;
  H.B.setInsertionPoint(Op);
  Value V = H.callRet(Op->getLoc(), RuntimeName, H.PtrTy, {}, {});
  if (Op->getNumResults() == 1)
    Op->getResult(0).replaceAllUsesWith(V);
  return true;
}

/* Pass through one or more matrix args (all PtrTy) to a runtime function
 * returning void. */
bool rewriteMatArgs(Operation *Op, Helper &H, StringRef RuntimeName,
                    unsigned ExpectedArgs) {
  if (Op->getNumOperands() != ExpectedArgs) return false;
  for (unsigned i = 0; i < ExpectedArgs; ++i)
    if (Op->getOperand(i).getType() != H.PtrTy) return false;
  llvm::SmallVector<Type, 4> ArgTys(ExpectedArgs, H.PtrTy);
  H.B.setInsertionPoint(Op);
  H.callVoid(Op->getLoc(), RuntimeName, ArgTys, Op->getOperands());
  return true;
}

bool rewriteCallee(Operation *Op, StringRef Callee, Helper &H,
                   StringGlobals &S) {
  Location L = Op->getLoc();
  unsigned N = Op->getNumOperands();

  // ---- 0-arg / no-arg keyword-style ----
  if (Callee == "figure" && N == 0)
    return rewritePtrReturnNoArg(Op, H, "matlab_figure_new");
  if (Callee == "figure" && N == 1) {
    /* figure(n) — id arg as f64. */
    auto Id = H.toF64(L, Op->getOperand(0));
    if (!Id) return false;
    H.B.setInsertionPoint(Op);
    Value V = H.callRet(L, "matlab_figure_by_id", H.PtrTy, {H.F64}, {*Id});
    if (Op->getNumResults() == 1) Op->getResult(0).replaceAllUsesWith(V);
    return true;
  }
  if (Callee == "gcf")            return rewritePtrReturnNoArg(Op, H, "matlab_gcf");
  if (Callee == "colorbar")       return rewriteVoidNoArg(Op, H, "matlab_colorbar");
  if (Callee == "loglog")         return rewriteVoidNoArg(Op, H, "matlab_loglog");
  if (Callee == "semilogx")       return rewriteVoidNoArg(Op, H, "matlab_semilogx");
  if (Callee == "semilogy")       return rewriteVoidNoArg(Op, H, "matlab_semilogy");

  // ---- close(...): close all if 'all', otherwise no-op for now. ----
  if (Callee == "close") {
    if (N == 0) return rewriteVoidNoArg(Op, H, "matlab_close_all");
    if (N == 1) {
      auto Pair = materializeStringArg(Op->getOperand(0), H.B, S);
      if (Pair && Pair->first) {
        H.B.setInsertionPoint(Op);
        H.callVoid(L, "matlab_close_all", {}, {});
        return true;
      }
    }
    return false;
  }

  // ---- grid / hold / box: keyword 'on' / 'off' or no-arg toggle. ----
  auto rewriteToggle = [&](StringRef onName, StringRef offName) -> bool {
    if (N == 0) {
      H.B.setInsertionPoint(Op);
      H.callVoid(L, onName, {}, {});
      return true;
    }
    if (N == 1) {
      /* Inspect the literal text directly — works for both
       * matlab.const_char (single quotes) and matlab.const_str (double
       * quotes / command-syntax). */
      auto *Def = Op->getOperand(0).getDefiningOp();
      if (!isMatlabOp(Def, "matlab.const_char") &&
          !isMatlabOp(Def, "matlab.const_str")) return false;
      auto Attr = Def->getAttrOfType<StringAttr>("value");
      if (!Attr) return false;
      StringRef T = Attr.getValue();
      H.B.setInsertionPoint(Op);
      H.callVoid(L, T == "off" ? offName : onName, {}, {});
      return true;
    }
    return false;
  };
  if (Callee == "grid") return rewriteToggle("matlab_grid_on", "matlab_grid_off");
  if (Callee == "hold") return rewriteToggle("matlab_hold_on", "matlab_hold_off");
  if (Callee == "box")  return rewriteToggle("matlab_box_on",  "matlab_box_off");

  // ---- title / xlabel / ylabel / zlabel — single string arg. ----
  if (Callee == "title")  return rewriteSimpleStringArg(Op, H, S, "matlab_title");
  if (Callee == "xlabel") return rewriteSimpleStringArg(Op, H, S, "matlab_xlabel");
  if (Callee == "ylabel") return rewriteSimpleStringArg(Op, H, S, "matlab_ylabel");
  if (Callee == "zlabel") return rewriteSimpleStringArg(Op, H, S, "matlab_zlabel");
  if (Callee == "colormap") return rewriteSimpleStringArg(Op, H, S, "matlab_colormap");
  if (Callee == "yyaxis")   return rewriteSimpleStringArg(Op, H, S, "matlab_yyaxis");
  if (Callee == "axis" && N == 1) {
    /* String form `axis equal` / `axis off` — single string operand. */
    auto *Def = Op->getOperand(0).getDefiningOp();
    if (isMatlabOp(Def, "matlab.const_char") ||
        isMatlabOp(Def, "matlab.const_str"))
      return rewriteSimpleStringArg(Op, H, S, "matlab_axis");
    /* Numeric form `axis([xmin xmax ymin ymax])` — 4-element matrix. */
    if (Op->getOperand(0).getType() == H.PtrTy)
      return rewriteMatArgs(Op, H, "matlab_axis_lims", 1);
    return false;
  }
  if (Callee == "print") return rewriteSimpleStringArg(Op, H, S, "matlab_savefig");

  // ---- saveas(fig, 'path') / saveas('path') — ignore the figure handle. ----
  if (Callee == "saveas") {
    Value PathArg;
    if (N == 1)      PathArg = Op->getOperand(0);
    else if (N == 2) PathArg = Op->getOperand(1);
    else return false;
    auto Pair = materializeStringArg(PathArg, H.B, S);
    if (!Pair) return false;
    auto [Ptr, Len] = *Pair;
    H.B.setInsertionPoint(Op);
    H.callVoid(L, "matlab_savefig", {H.PtrTy, H.I64},
               {Ptr, H.i64Const(L, Len)});
    return true;
  }

  // ---- xlim / ylim — single matrix [lo hi] arg. ----
  if (Callee == "xlim" && N == 1 && Op->getOperand(0).getType() == H.PtrTy)
    return rewriteMatArgs(Op, H, "matlab_xlim", 1);
  if (Callee == "ylim" && N == 1 && Op->getOperand(0).getType() == H.PtrTy)
    return rewriteMatArgs(Op, H, "matlab_ylim", 1);

  // ---- view(az, el) — two scalars. ----
  if (Callee == "view" && N == 2) {
    auto Az = H.toF64(L, Op->getOperand(0));
    auto El = H.toF64(L, Op->getOperand(1));
    if (!Az || !El) return false;
    H.B.setInsertionPoint(Op);
    H.callVoid(L, "matlab_view", {H.F64, H.F64}, {*Az, *El});
    return true;
  }

  // ---- subplot(m, n, i) — three scalars. ----
  if (Callee == "subplot" && N == 3) {
    auto Mv = H.toF64(L, Op->getOperand(0));
    auto Nv = H.toF64(L, Op->getOperand(1));
    auto Iv = H.toF64(L, Op->getOperand(2));
    if (!Mv || !Nv || !Iv) return false;
    H.B.setInsertionPoint(Op);
    H.callVoid(L, "matlab_subplot", {H.F64, H.F64, H.F64},
               {*Mv, *Nv, *Iv});
    return true;
  }

  // ---- xticks / yticks: matrix arg with tick locations ----
  if ((Callee == "xticks" || Callee == "yticks") && N == 1 &&
      Op->getOperand(0).getType() == H.PtrTy) {
    return rewriteMatArgs(Op, H,
        Callee == "xticks" ? "matlab_xticks" : "matlab_yticks", 1);
  }

  // ---- xticklabels / yticklabels: varargs of strings ----
  if (Callee == "xticklabels" || Callee == "yticklabels") {
    if (N == 0) return false;
    llvm::SmallVector<std::pair<Value, int64_t>, 8> Strs;
    for (unsigned i = 0; i < N; ++i) {
      auto Pair = materializeStringArg(Op->getOperand(i), H.B, S);
      if (!Pair) return false;
      Strs.push_back(*Pair);
    }
    H.B.setInsertionPoint(Op);
    auto PtrArrTy = LLVM::LLVMArrayType::get(H.PtrTy, (unsigned)N);
    auto LenArrTy = LLVM::LLVMArrayType::get(H.I64,   (unsigned)N);
    Value One = H.i64Const(L, 1);
    Value PtrArr = LLVM::AllocaOp::create(H.B, L, H.PtrTy, PtrArrTy, One, /*alignment=*/0);
    Value LenArr = LLVM::AllocaOp::create(H.B, L, H.PtrTy, LenArrTy, One, /*alignment=*/0);
    for (unsigned i = 0; i < N; ++i) {
      Value Idx = H.i64Const(L, (int64_t)i);
      Value PSlot = LLVM::GEPOp::create(H.B, L, H.PtrTy, H.PtrTy, PtrArr, ValueRange{Idx});
      Value LSlot = LLVM::GEPOp::create(H.B, L, H.PtrTy, H.I64,   LenArr, ValueRange{Idx});
      LLVM::StoreOp::create(H.B, L, Strs[i].first,                 PSlot);
      LLVM::StoreOp::create(H.B, L, H.i64Const(L, Strs[i].second), LSlot);
    }
    StringRef rt = Callee == "xticklabels" ? "matlab_xticklabels_strs"
                                            : "matlab_yticklabels_strs";
    H.callVoid(L, rt, {H.PtrTy, H.PtrTy, H.I64},
               {PtrArr, LenArr, H.i64Const(L, (int64_t)N)});
    return true;
  }

  // ---- xline(v) / yline(v) [+ label] ----
  if (Callee == "xline" || Callee == "yline") {
    if (N < 1) return false;
    auto V = H.toF64(L, Op->getOperand(0));
    if (!V) return false;
    if (N == 1) {
      H.B.setInsertionPoint(Op);
      H.callVoid(L, Callee == "xline" ? "matlab_xline" : "matlab_yline",
                 {H.F64}, {*V});
      return true;
    }
    if (N == 2) {
      auto Pair = materializeStringArg(Op->getOperand(1), H.B, S);
      if (!Pair) return false;
      auto [Ptr, Len] = *Pair;
      H.B.setInsertionPoint(Op);
      H.callVoid(L,
                 Callee == "xline" ? "matlab_xline_label" : "matlab_yline_label",
                 {H.F64, H.PtrTy, H.I64},
                 {*V, Ptr, H.i64Const(L, Len)});
      return true;
    }
    return false;
  }

  // ---- text(x, y, 's') — two scalars + string. ----
  if (Callee == "text" && N == 3) {
    auto Xv = H.toF64(L, Op->getOperand(0));
    auto Yv = H.toF64(L, Op->getOperand(1));
    auto Pair = materializeStringArg(Op->getOperand(2), H.B, S);
    if (!Xv || !Yv || !Pair) return false;
    auto [Ptr, Len] = *Pair;
    H.B.setInsertionPoint(Op);
    H.callVoid(L, "matlab_text", {H.F64, H.F64, H.PtrTy, H.I64},
               {*Xv, *Yv, Ptr, H.i64Const(L, Len)});
    return true;
  }

  // ---- legend('label1', 'label2', ...) varargs form ----
  // Stack-allocates two parallel arrays at the LLVM IR level:
  // ptrs[N] and lens[N], fills them from the materialized literal
  // globals, then calls matlab_legend_strs. The cell form
  // legend({'a','b'}) needs matlab_cell_get_mat + char-mat decoding and
  // is left for a follow-up.
  if (Callee == "legend") {
    if (N == 0) return false;
    /* Cell form: legend({'a', 'b'}). The cell flows in as a single
     * PtrTy operand pointing at a matlab_cell. The runtime entry walks
     * it via matlab_cell_get_mat. */
    if (N == 1 && Op->getOperand(0).getType() == H.PtrTy) {
      H.B.setInsertionPoint(Op);
      H.callVoid(L, "matlab_legend", {H.PtrTy}, {Op->getOperand(0)});
      return true;
    }
    /* Varargs form: legend('a', 'b', ...) — every operand is a literal. */
    llvm::SmallVector<std::pair<Value, int64_t>, 8> Strs;
    for (unsigned i = 0; i < N; ++i) {
      auto Pair = materializeStringArg(Op->getOperand(i), H.B, S);
      if (!Pair) return false;
      Strs.push_back(*Pair);
    }
    H.B.setInsertionPoint(Op);
    auto PtrArrTy = LLVM::LLVMArrayType::get(H.PtrTy, (unsigned)N);
    auto LenArrTy = LLVM::LLVMArrayType::get(H.I64,   (unsigned)N);
    Value One = H.i64Const(L, 1);
    Value PtrArr = LLVM::AllocaOp::create(H.B, L, H.PtrTy, PtrArrTy, One, /*alignment=*/0);
    Value LenArr = LLVM::AllocaOp::create(H.B, L, H.PtrTy, LenArrTy, One, /*alignment=*/0);
    for (unsigned i = 0; i < N; ++i) {
      Value Idx = H.i64Const(L, (int64_t)i);
      Value PSlot = LLVM::GEPOp::create(H.B, L, H.PtrTy, H.PtrTy, PtrArr, ValueRange{Idx});
      Value LSlot = LLVM::GEPOp::create(H.B, L, H.PtrTy, H.I64,   LenArr, ValueRange{Idx});
      LLVM::StoreOp::create(H.B, L, Strs[i].first,                 PSlot);
      LLVM::StoreOp::create(H.B, L, H.i64Const(L, Strs[i].second), LSlot);
    }
    H.callVoid(L, "matlab_legend_strs",
               {H.PtrTy, H.PtrTy, H.I64},
               {PtrArr, LenArr, H.i64Const(L, (int64_t)N)});
    return true;
  }

  // ---- plot family with arity dispatch ----
  // plot(y) / plot(x, y) / plot(x, y, 'fmt')
  if (Callee == "plot") {
    if (N == 1 && Op->getOperand(0).getType() == H.PtrTy)
      return rewriteMatArgs(Op, H, "matlab_plot1", 1);
    if (N == 2 && Op->getOperand(0).getType() == H.PtrTy &&
                  Op->getOperand(1).getType() == H.PtrTy)
      return rewriteMatArgs(Op, H, "matlab_plot2", 2);
    if (N == 3 && Op->getOperand(0).getType() == H.PtrTy &&
                  Op->getOperand(1).getType() == H.PtrTy) {
      auto Pair = materializeStringArg(Op->getOperand(2), H.B, S);
      if (!Pair) return false;
      auto [Ptr, Len] = *Pair;
      H.B.setInsertionPoint(Op);
      H.callVoid(L, "matlab_plot_fmt",
                 {H.PtrTy, H.PtrTy, H.PtrTy, H.I64},
                 {Op->getOperand(0), Op->getOperand(1), Ptr, H.i64Const(L, Len)});
      return true;
    }
    return false;
  }

  // bar / scatter / stem / stairs / area — same arity dispatch as plot
  // (no linespec variant for now).
  auto rewritePair12 = [&](StringRef name1, StringRef name2) -> bool {
    if (N == 1 && Op->getOperand(0).getType() == H.PtrTy)
      return rewriteMatArgs(Op, H, name1, 1);
    if (N == 2 && Op->getOperand(0).getType() == H.PtrTy &&
                  Op->getOperand(1).getType() == H.PtrTy)
      return rewriteMatArgs(Op, H, name2, 2);
    return false;
  };
  if (Callee == "bar") return rewritePair12("matlab_bar1", "matlab_bar2");
  if (Callee == "scatter") {
    /* scatter only accepts (x, y) — no scatter(y) shortcut. */
    if (N == 2 && Op->getOperand(0).getType() == H.PtrTy &&
                  Op->getOperand(1).getType() == H.PtrTy)
      return rewriteMatArgs(Op, H, "matlab_scatter", 2);
    return false;
  }
  if (Callee == "stem")    return rewritePair12("matlab_stem1",    "matlab_stem2");
  if (Callee == "stairs")  return rewritePair12("matlab_stairs1",  "matlab_stairs2");
  if (Callee == "area")    return rewritePair12("matlab_area1",    "matlab_area2");

  // errorbar(x, y, e) / errorbar(x, y, neg, pos)
  if (Callee == "errorbar") {
    if (N == 3) return rewriteMatArgs(Op, H, "matlab_errorbar",  3);
    if (N == 4) return rewriteMatArgs(Op, H, "matlab_errorbar2", 4);
    return false;
  }

  // histogram(data) / histogram(data, n)
  if (Callee == "histogram") {
    if (N == 1 && Op->getOperand(0).getType() == H.PtrTy) {
      H.B.setInsertionPoint(Op);
      H.callVoid(L, "matlab_histogram", {H.PtrTy, H.F64},
                 {Op->getOperand(0), H.f64Const(L, 0.0)});
      return true;
    }
    if (N == 2 && Op->getOperand(0).getType() == H.PtrTy) {
      auto Nb = H.toF64(L, Op->getOperand(1));
      if (!Nb) return false;
      H.B.setInsertionPoint(Op);
      H.callVoid(L, "matlab_histogram", {H.PtrTy, H.F64},
                 {Op->getOperand(0), *Nb});
      return true;
    }
    return false;
  }

  // imshow / imagesc / pcolor / contour — single matrix.
  if (Callee == "imshow")   return rewriteMatArgs(Op, H, "matlab_imshow",   1);
  if (Callee == "imagesc")  return rewriteMatArgs(Op, H, "matlab_imagesc",  1);
  if (Callee == "pcolor")   return rewriteMatArgs(Op, H, "matlab_pcolor",   1);
  if (Callee == "contourf") return rewriteMatArgs(Op, H, "matlab_contourf", 1);
  if (Callee == "contour") {
    if (N == 1) return rewriteMatArgs(Op, H, "matlab_contour", 1);
    if (N == 2) return rewriteMatArgs(Op, H, "matlab_contour_levels", 2);
    return false;
  }
  if (Callee == "quiver" && N == 4) return rewriteMatArgs(Op, H, "matlab_quiver", 4);

  // surf(Z) / surf(X, Y, Z); mesh same.
  if (Callee == "surf") {
    if (N == 1) return rewriteMatArgs(Op, H, "matlab_surf1", 1);
    if (N == 3) return rewriteMatArgs(Op, H, "matlab_surf3", 3);
    return false;
  }
  if (Callee == "mesh") {
    if (N == 1) return rewriteMatArgs(Op, H, "matlab_mesh1", 1);
    if (N == 3) return rewriteMatArgs(Op, H, "matlab_mesh3", 3);
    return false;
  }

  /* pdeplot — 2-D version, accepts (nodes, triangles) or
   * (nodes, triangles, data). */
  if (Callee == "pdeplot") {
    if (N == 2) {
      H.B.setInsertionPoint(Op);
      auto NullPtr = mlir::LLVM::ZeroOp::create(H.B, L, H.PtrTy);
      H.callVoid(L, "matlab_pdeplot", {H.PtrTy, H.PtrTy, H.PtrTy},
                 {Op->getOperand(0), Op->getOperand(1), NullPtr.getRes()});
      return true;
    }
    if (N == 3) return rewriteMatArgs(Op, H, "matlab_pdeplot", 3);
    return false;
  }

  /* pdeplot3D — accepts (nodes, triangles) or (nodes, triangles, data).
   * Lower-case `pdeplot3d` alias kept for ergonomics. */
  if (Callee == "pdeplot3d" || Callee == "pdeplot3D") {
    if (N == 2) {
      H.B.setInsertionPoint(Op);
      /* Pass a NULL data pointer when no colour vector is supplied. */
      auto NullPtr = mlir::LLVM::ZeroOp::create(H.B, L, H.PtrTy);
      H.callVoid(L, "matlab_pdeplot3d", {H.PtrTy, H.PtrTy, H.PtrTy},
                 {Op->getOperand(0), Op->getOperand(1), NullPtr.getRes()});
      return true;
    }
    if (N == 3) return rewriteMatArgs(Op, H, "matlab_pdeplot3d", 3);
    return false;
  }
  if (Callee == "pdeplot3d_deformation" && N == 1)
    return rewriteMatArgs(Op, H, "matlab_pdeplot3d_deformation", 1);
  if (Callee == "pdeplot3d_deform_scale" && N == 1) {
    /* Scalar f64 arg → matlab_pdeplot3d_deform_scale(double). */
    H.B.setInsertionPoint(Op);
    H.callVoid(L, "matlab_pdeplot3d_deform_scale",
               {H.F64}, {Op->getOperand(0)});
    return true;
  }

  // plot3(x, y, z) / plot3(x, y, z, 'fmt')
  if (Callee == "plot3") {
    if (N == 3) return rewriteMatArgs(Op, H, "matlab_plot3", 3);
    if (N == 4) {
      auto Pair = materializeStringArg(Op->getOperand(3), H.B, S);
      if (!Pair) return false;
      auto [Ptr, Len] = *Pair;
      H.B.setInsertionPoint(Op);
      H.callVoid(L, "matlab_plot3_fmt",
                 {H.PtrTy, H.PtrTy, H.PtrTy, H.PtrTy, H.I64},
                 {Op->getOperand(0), Op->getOperand(1), Op->getOperand(2),
                  Ptr, H.i64Const(L, Len)});
      return true;
    }
    return false;
  }

  return false;
}

}  // namespace

bool runLowerPlot(ModuleOp M) {
  OpBuilder B(M.getContext());
  StringGlobals Strings(M, B);
  Helper H(M, B, Strings);

  llvm::SmallVector<Operation *, 16> ToRewrite;
  M.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto Callee = Op->getAttrOfType<StringAttr>("callee");
    if (!Callee) return;
    if (plotBuiltins().contains(Callee.getValue()))
      ToRewrite.push_back(Op);
  });

  /* Series-producing callees that accept Name/Value tail pairs. The
   * dispatch here strips the pairs, runs the regular rewriter on the
   * shortened operand list, then emits the setter follow-ups. */
  static const llvm::StringSet<> AcceptsProps = {
    "plot", "plot3", "scatter", "bar", "stem", "stairs", "area",
    "errorbar", "histogram", "surf", "mesh",
    "imshow", "imagesc", "pcolor", "contour", "contourf",
    "quiver",
  };

  bool Changed = false;
  for (Operation *Op : ToRewrite) {
    auto Callee = Op->getAttrOfType<StringAttr>("callee");
    StringRef Name = Callee.getValue();

    /* Strip trailing (string-literal-name, value) pairs. We do this by
     * temporarily setting fewer operands on the op, calling the regular
     * rewriter, then restoring + emitting setters after the new call. */
    unsigned pairs = AcceptsProps.contains(Name)
                         ? countTrailingPropPairs(Op) : 0;
    unsigned full_n = Op->getNumOperands();
    unsigned pos_n  = full_n - pairs * 2;

    /* Save the prop operands, clip the op to positional only. */
    llvm::SmallVector<Value, 8> KeptProps;
    for (unsigned i = pos_n; i < full_n; ++i)
      KeptProps.push_back(Op->getOperand(i));
    if (pairs) Op->setOperands(Op->getOperands().take_front(pos_n));

    bool ok = rewriteCallee(Op, Name, H, Strings);
    if (ok) {
      /* Emit setters after the synthesized main call. The handler set
       * the insertion point right before Op when it ran; subsequent
       * inserts go *just before* Op too. To place the setters AFTER the
       * main call but still before Op, set IP to immediately before Op. */
      if (pairs) {
        /* Restore operands so emitPropSetters can read them. */
        llvm::SmallVector<Value, 16> All(Op->getOperands().begin(),
                                          Op->getOperands().end());
        for (Value V : KeptProps) All.push_back(V);
        Op->setOperands(All);

        H.B.setInsertionPoint(Op);
        emitPropSetters(Op, pos_n, H, Strings);
      }
      Op->erase();
      Changed = true;
    } else if (pairs) {
      /* Rewrite failed — restore original operands so other passes can
       * still see the call as written. */
      llvm::SmallVector<Value, 16> All(Op->getOperands().begin(),
                                        Op->getOperands().end());
      for (Value V : KeptProps) All.push_back(V);
      Op->setOperands(All);
    }
  }

  /* After erasing call_builtin ops, three matlab.* op kinds may be left
   * dangling without a downstream LLVM translation interface and would
   * crash conversion later:
   *   - matlab.const_str  (toggle path: `grid on` / `hold off` / `box on`)
   *   - matlab.const_char (string args we inspected directly)
   *   - matlab.make_handle (e.g. `saveas(gcf, 'path')` — the bare `gcf`
   *     in argument position parses as a function-handle creation; we
   *     don't consume the figure handle in saveas, so it's unused.)
   * Erase any of these that have no users. */
  llvm::SmallVector<Operation *, 8> DeadOps;
  M.walk([&](Operation *Op) {
    if ((isMatlabOp(Op, "matlab.const_str")  ||
         isMatlabOp(Op, "matlab.const_char") ||
         isMatlabOp(Op, "matlab.make_handle")) &&
        Op->use_empty()) {
      DeadOps.push_back(Op);
    }
  });
  for (Operation *Op : DeadOps) Op->erase();

  return Changed;
}

}  // namespace mlirgen
}  // namespace matlab
