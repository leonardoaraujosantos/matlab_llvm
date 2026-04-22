#include "matlab/MIR/MIR.h"

namespace matlab {
namespace mir {

const char *opKindName(OpKind K) {
  switch (K) {
#define C(N) case OpKind::N: return #N;
  C(ModuleOp) C(FuncOp)
  C(ConstInt) C(ConstFloat) C(ConstComplex) C(ConstString) C(ConstChar)
  C(ConstLogical) C(ConstColon) C(ConstEnd)
  C(Alloc) C(Load) C(Store)
  C(Neg) C(UPlus) C(Not)
  C(Add) C(Sub)
  C(MatMul) C(MatDiv) C(MatLDiv) C(MatPow)
  C(EMul) C(EDiv) C(ELDiv) C(EPow)
  C(Eq) C(Ne) C(Lt) C(Le) C(Gt) C(Ge)
  C(BitAnd) C(BitOr) C(ShortAnd) C(ShortOr)
  C(Transpose) C(CTranspose)
  C(Range) C(ConcatRow) C(ConcatCol)
  C(Subscript) C(CellSubscript) C(FieldAccess)
  C(Call) C(CallBuiltin) C(CallIndirect)
  C(MakeHandle) C(MakeAnon)
  C(IfOp) C(ForOp) C(WhileOp)
  C(Yield) C(Return) C(Break) C(Continue)
  C(MakeMatrix) C(MakeCell)
  C(DynamicField)
#undef C
  }
  return "?";
}

//===----------------------------------------------------------------------===//
// MIRContext
//===----------------------------------------------------------------------===//

MIRContext::MIRContext() = default;
MIRContext::~MIRContext() = default;

Op *MIRContext::newOp(OpKind K) {
  auto O = std::make_unique<Op>();
  O->K = K;
  Op *P = O.get();
  Ops.push_back(std::move(O));
  return P;
}

Value *MIRContext::newValue(const Type *Ty) {
  auto V = std::make_unique<Value>();
  V->Ty = Ty;
  V->Id = NextValueId++;
  Value *P = V.get();
  Values.push_back(std::move(V));
  return P;
}

Block *MIRContext::newBlock() {
  auto B = std::make_unique<Block>();
  B->Id = NextBlockId++;
  Block *P = B.get();
  Blocks.push_back(std::move(B));
  return P;
}

Region *MIRContext::newRegion() {
  auto R = std::make_unique<Region>();
  Region *P = R.get();
  Regions.push_back(std::move(R));
  return P;
}

Value *MIRContext::addBlockArgument(Block *B, const Type *Ty) {
  Value *V = newValue(Ty);
  V->DefiningBlock = B;
  V->Index = static_cast<uint32_t>(B->Arguments.size());
  B->Arguments.push_back(V);
  return V;
}

Module createModule(MIRContext &Ctx) {
  Module M;
  M.ModuleOp = Ctx.newOp(OpKind::ModuleOp);
  M.Body = Ctx.newRegion();
  M.Body->Parent = M.ModuleOp;
  M.ModuleOp->Regions.push_back(M.Body);
  M.EntryBlock = Ctx.newBlock();
  M.EntryBlock->Parent = M.Body;
  M.Body->Blocks.push_back(M.EntryBlock);
  return M;
}

} // namespace mir
} // namespace matlab
