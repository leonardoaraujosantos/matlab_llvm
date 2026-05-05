// Phase 5.6.1 — apply `% hdl: port(...)` pragmas to function
// signatures so a function-only `.m` file emits synthesizable
// SystemVerilog without a separate typed driver.
//
// `ScanHWPragmas` parses `% hdl: port(<name>, <kind>, ...)`
// comments and attaches them as an `hdl.ports` ArrayAttr on the
// func.func, with one DictionaryAttr per port (fields: name,
// kind, signed, width, frac). This pass walks every such function
// and rewrites the FunctionType + entry-block arg types to the
// declared types, matching by `matlab.name` arg/result attr.
//
// Runs BEFORE the user-call refinement iteration loop so the
// existing pattern set (BinArithToArith, slot-type inference,
// etc.) sees the typed args naturally.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#include <string>

namespace matlab {
namespace mlirgen {

namespace {

/// Build the MLIR type for a parsed port DictionaryAttr.
mlir::Type typeForPortAttr(mlir::MLIRContext &Ctx,
                           mlir::DictionaryAttr Entry) {
  auto KindAttr = Entry.getAs<mlir::StringAttr>("kind");
  auto WidthAttr = Entry.getAs<mlir::IntegerAttr>("width");
  if (!KindAttr || !WidthAttr) return {};
  unsigned W = (unsigned)WidthAttr.getInt();
  if (W == 0 || W > 64) return {};
  // For `bool` we always emit i1; everything else gets the declared
  // width as a (signless) integer. MLIR integers don't carry sign;
  // signedness is preserved on the DictionaryAttr and read back by
  // downstream passes / emitters when they need it.
  if (KindAttr.getValue() == "bool" || KindAttr.getValue() == "i1")
    return mlir::IntegerType::get(&Ctx, 1);
  return mlir::IntegerType::get(&Ctx, W);
}

} // namespace

bool runApplyPortTypePragmas(mlir::ModuleOp M) {
  bool Failed = false;
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;
    auto PortsAttr = F->getAttrOfType<mlir::ArrayAttr>("hdl.ports");
    if (!PortsAttr || PortsAttr.empty()) return;

    auto &Ctx = *F.getContext();
    auto FT = F.getFunctionType();

    // Build mutable type lists seeded with the existing signature.
    llvm::SmallVector<mlir::Type> InTys(FT.getInputs().begin(),
                                        FT.getInputs().end());
    llvm::SmallVector<mlir::Type> OutTys(FT.getResults().begin(),
                                         FT.getResults().end());

    // Index args + results by their `matlab.name` attribute. Args
    // that have no name attr are skipped (they cannot be addressed
    // by a `port(...)` pragma).
    auto findArgByName = [&](llvm::StringRef Name) -> int {
      for (unsigned I = 0; I < FT.getNumInputs(); ++I)
        if (auto S = F.getArgAttrOfType<mlir::StringAttr>(I, "matlab.name"))
          if (S.getValue() == Name) return (int)I;
      return -1;
    };
    auto findResByName = [&](llvm::StringRef Name) -> int {
      for (unsigned I = 0; I < FT.getNumResults(); ++I)
        if (auto S = F.getResultAttrOfType<mlir::StringAttr>(I, "matlab.name"))
          if (S.getValue() == Name) return (int)I;
      return -1;
    };

    for (auto Attr : PortsAttr) {
      auto Entry = mlir::dyn_cast<mlir::DictionaryAttr>(Attr);
      if (!Entry) continue;
      auto NameAttr = Entry.getAs<mlir::StringAttr>("name");
      if (!NameAttr) continue;
      auto T = typeForPortAttr(Ctx, Entry);
      if (!T) {
        F.emitWarning() << "ignoring `port(" << NameAttr.getValue()
                        << ", ...)` pragma with malformed type fields";
        continue;
      }

      int Idx = findArgByName(NameAttr.getValue());
      if (Idx >= 0) {
        // Reject the silent case where a typed caller already
        // refined this arg to something incompatible — surfacing
        // the conflict early is friendlier than emitting wrong RTL.
        auto Existing = InTys[Idx];
        bool IsAny = mlir::isa<mlir::NoneType>(Existing);
        if (!IsAny && Existing != T) {
          F.emitError() << "`port(" << NameAttr.getValue()
                        << ", ...)` pragma declares type " << T
                        << " but arg already has incompatible type "
                        << Existing;
          Failed = true;
          continue;
        }
        InTys[Idx] = T;
        // Thread the pragma's signedness onto the arg as
        // `matlab.fi_signed` so the SV port-list emitter can render
        // unsigned ports as `logic [W-1:0]` instead of the default
        // `logic signed [W-1:0]`. MLIR's IntegerType is signless;
        // signedness lives on attrs. Same convention the vector
        // (`!llvm.ptr`) port branch already follows.
        if (auto SignedA = Entry.getAs<mlir::BoolAttr>("signed")) {
          auto I32 = mlir::IntegerType::get(&Ctx, 32);
          F.setArgAttr((unsigned)Idx, "matlab.fi_signed",
              mlir::IntegerAttr::get(I32, SignedA.getValue() ? 1 : 0));
        }
        continue;
      }
      int RIdx = findResByName(NameAttr.getValue());
      if (RIdx >= 0) {
        // Result-port pragma. Tag the result with `matlab.fi_signed`
        // and `matlab.fi_wl` so the SV emitter renders the output
        // port at the user-declared signedness/width — overriding
        // the default-signed-multi-bit rule for `% hdl: port(_,
        // fi, unsigned, _, _)` outputs. The result type itself
        // stays as inferred by RefineFuncSigs from the typed body
        // (matching the pragma if the user got it right; a later
        // verify catches conflicts).
        if (auto SignedA = Entry.getAs<mlir::BoolAttr>("signed")) {
          auto I32 = mlir::IntegerType::get(&Ctx, 32);
          F.setResultAttr((unsigned)RIdx, "matlab.fi_signed",
              mlir::IntegerAttr::get(I32, SignedA.getValue() ? 1 : 0));
        }
        if (auto WAttr = Entry.getAs<mlir::IntegerAttr>("width")) {
          auto I32 = mlir::IntegerType::get(&Ctx, 32);
          F.setResultAttr((unsigned)RIdx, "matlab.fi_wl",
              mlir::IntegerAttr::get(I32, WAttr.getInt()));
        }
        (void)OutTys[RIdx];
        continue;
      }
      F.emitWarning() << "`port(" << NameAttr.getValue()
                      << ", ...)` pragma names no known arg or result; "
                         "ignored";
    }

    if (Failed) return;

    // Update the function type + entry-block arg types so downstream
    // passes see typed values. Only inputs change here; result types
    // get inferred by RefineFuncSigs from the typed body.
    auto NewFT = mlir::FunctionType::get(&Ctx, InTys, OutTys);
    F.setType(NewFT);
    auto &Entry = F.front();
    for (unsigned I = 0; I < InTys.size(); ++I) {
      Entry.getArgument(I).setType(InTys[I]);
    }
  });
  return !Failed;
}

} // namespace mlirgen
} // namespace matlab
