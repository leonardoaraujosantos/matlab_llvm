#pragma once

// Forward-declare MLIR types to keep this header cheap. Clients that actually
// need the full types include <mlir/IR/MLIRContext.h> themselves.
namespace mlir { class MLIRContext; }

namespace matlab {
namespace mlirgen {

/// RAII wrapper around mlir::MLIRContext preloaded with the dialects we use
/// (func, arith, scf, tensor, memref, cf) and configured to allow our
/// unregistered `matlab.*` operations.
class Context {
public:
  Context();
  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  mlir::MLIRContext &get() { return *Ctx; }
  const mlir::MLIRContext &get() const { return *Ctx; }

private:
  mlir::MLIRContext *Ctx; // owned
};

} // namespace mlirgen
} // namespace matlab
