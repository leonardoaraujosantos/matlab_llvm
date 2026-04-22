#include "matlab/MIR/Builder.h"

#include <cassert>

namespace matlab {
namespace mir {

//===----------------------------------------------------------------------===//
// Low-level create
//===----------------------------------------------------------------------===//

Op *Builder::create(OpKind K, std::vector<Value *> Operands,
                    std::vector<const Type *> ResultTypes, SourceRange Loc) {
  Op *O = Ctx.newOp(K);
  O->Operands = std::move(Operands);
  O->Loc = Loc;
  for (const Type *T : ResultTypes) {
    Value *R = Ctx.newValue(T);
    R->DefiningOp = O;
    R->Index = static_cast<uint32_t>(O->Results.size());
    R->Loc = Loc.Begin;
    O->Results.push_back(R);
  }
  if (InsertBlock) {
    O->ParentBlock = InsertBlock;
    InsertBlock->Ops.push_back(O);
  }
  return O;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

Value *Builder::constInt(int64_t V, const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::ConstInt, {}, {Ty}, Loc);
  O->setAttr("value", Attribute::ofInt(V));
  return O->result();
}
Value *Builder::constFloat(double V, const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::ConstFloat, {}, {Ty}, Loc);
  O->setAttr("value", Attribute::ofFloat(V));
  return O->result();
}
Value *Builder::constString(std::string V, const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::ConstString, {}, {Ty}, Loc);
  O->setAttr("value", Attribute::ofStr(std::move(V)));
  return O->result();
}
Value *Builder::constChar(std::string V, const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::ConstChar, {}, {Ty}, Loc);
  O->setAttr("value", Attribute::ofStr(std::move(V)));
  return O->result();
}
Value *Builder::constLogical(bool V, SourceRange Loc) {
  Op *O = create(OpKind::ConstLogical, {}, {TC.scalar(Dtype::Logical)}, Loc);
  O->setAttr("value", Attribute::ofBool(V));
  return O->result();
}
Value *Builder::constColon(SourceRange Loc) {
  Op *O = create(OpKind::ConstColon, {}, {TC.any()}, Loc);
  return O->result();
}
Value *Builder::constEnd(SourceRange Loc) {
  Op *O = create(OpKind::ConstEnd, {}, {TC.scalar(Dtype::Double)}, Loc);
  return O->result();
}

//===----------------------------------------------------------------------===//
// Slot operations
//===----------------------------------------------------------------------===//

Value *Builder::alloc(const Type *Ty, std::string Name, SourceRange Loc) {
  // Slot value's type is the value type it holds. A separate "SlotType"
  // layer would make the distinction explicit; for now the slot and the value
  // share the same Type* (load/store wrap it).
  Op *O = create(OpKind::Alloc, {}, {Ty}, Loc);
  if (!Name.empty()) O->setAttr("name", Attribute::ofSym(std::move(Name)));
  return O->result();
}

Value *Builder::load(Value *Slot, const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::Load, {Slot}, {Ty}, Loc);
  return O->result();
}

void Builder::store(Value *V, Value *Slot, SourceRange Loc) {
  (void)create(OpKind::Store, {V, Slot}, {}, Loc);
}

//===----------------------------------------------------------------------===//
// Arithmetic / comparison
//===----------------------------------------------------------------------===//

Value *Builder::unary(OpKind K, Value *A, const Type *Ty, SourceRange Loc) {
  Op *O = create(K, {A}, {Ty}, Loc);
  return O->result();
}

Value *Builder::binary(OpKind K, Value *A, Value *B, const Type *Ty,
                       SourceRange Loc) {
  Op *O = create(K, {A, B}, {Ty}, Loc);
  return O->result();
}

//===----------------------------------------------------------------------===//
// Array ops
//===----------------------------------------------------------------------===//

Value *Builder::range(Value *Start, Value *Step, Value *End, const Type *Ty,
                      SourceRange Loc) {
  std::vector<Value *> Os;
  Os.push_back(Start);
  if (Step) Os.push_back(Step);
  Os.push_back(End);
  Op *O = create(OpKind::Range, std::move(Os), {Ty}, Loc);
  O->setAttr("has_step", Attribute::ofBool(Step != nullptr));
  return O->result();
}

Value *Builder::concatRow(std::vector<Value *> Elts, const Type *Ty,
                          SourceRange Loc) {
  Op *O = create(OpKind::ConcatRow, std::move(Elts), {Ty}, Loc);
  return O->result();
}

Value *Builder::concatCol(std::vector<Value *> Elts, const Type *Ty,
                          SourceRange Loc) {
  Op *O = create(OpKind::ConcatCol, std::move(Elts), {Ty}, Loc);
  return O->result();
}

Value *Builder::subscript(Value *Arr, std::vector<Value *> Idx, const Type *Ty,
                          SourceRange Loc) {
  std::vector<Value *> Os;
  Os.push_back(Arr);
  for (auto *I : Idx) Os.push_back(I);
  Op *O = create(OpKind::Subscript, std::move(Os), {Ty}, Loc);
  O->setAttr("nindices", Attribute::ofInt(static_cast<int64_t>(Idx.size())));
  return O->result();
}

Value *Builder::cellSubscript(Value *Arr, std::vector<Value *> Idx,
                              const Type *Ty, SourceRange Loc) {
  std::vector<Value *> Os;
  Os.push_back(Arr);
  for (auto *I : Idx) Os.push_back(I);
  Op *O = create(OpKind::CellSubscript, std::move(Os), {Ty}, Loc);
  O->setAttr("nindices", Attribute::ofInt(static_cast<int64_t>(Idx.size())));
  return O->result();
}

Value *Builder::fieldAccess(Value *Base, std::string Field, const Type *Ty,
                            SourceRange Loc) {
  Op *O = create(OpKind::FieldAccess, {Base}, {Ty}, Loc);
  O->setAttr("field", Attribute::ofSym(std::move(Field)));
  return O->result();
}

Value *Builder::dynamicField(Value *Base, Value *Name, const Type *Ty,
                             SourceRange Loc) {
  Op *O = create(OpKind::DynamicField, {Base, Name}, {Ty}, Loc);
  return O->result();
}

//===----------------------------------------------------------------------===//
// Calls
//===----------------------------------------------------------------------===//

Value *Builder::call(std::string Callee, std::vector<Value *> Args,
                     const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::Call, std::move(Args), {Ty}, Loc);
  O->setAttr("callee", Attribute::ofSym(std::move(Callee)));
  return O->result();
}

Value *Builder::callBuiltin(std::string Callee, std::vector<Value *> Args,
                            const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::CallBuiltin, std::move(Args), {Ty}, Loc);
  O->setAttr("callee", Attribute::ofSym(std::move(Callee)));
  return O->result();
}

Value *Builder::callIndirect(Value *Handle, std::vector<Value *> Args,
                             const Type *Ty, SourceRange Loc) {
  std::vector<Value *> Os;
  Os.push_back(Handle);
  for (auto *A : Args) Os.push_back(A);
  Op *O = create(OpKind::CallIndirect, std::move(Os), {Ty}, Loc);
  return O->result();
}

//===----------------------------------------------------------------------===//
// Handles / anonymous functions
//===----------------------------------------------------------------------===//

Value *Builder::makeHandle(std::string Name, const Type *Ty, SourceRange Loc) {
  Op *O = create(OpKind::MakeHandle, {}, {Ty}, Loc);
  O->setAttr("callee", Attribute::ofSym(std::move(Name)));
  return O->result();
}

Op *Builder::makeAnon(std::vector<std::string> Params, const Type *Ty,
                      SourceRange Loc) {
  Op *O = create(OpKind::MakeAnon, {}, {Ty}, Loc);
  // Join params into a single string attribute; printer will split for display.
  std::string Joined;
  for (size_t i = 0; i < Params.size(); ++i) {
    if (i) Joined += ",";
    Joined += Params[i];
  }
  O->setAttr("params", Attribute::ofStr(std::move(Joined)));
  Region *Body = Ctx.newRegion();
  Body->Parent = O;
  O->Regions.push_back(Body);
  Block *Entry = Ctx.newBlock();
  Entry->Parent = Body;
  Body->Blocks.push_back(Entry);
  return O;
}

//===----------------------------------------------------------------------===//
// Structured control flow
//===----------------------------------------------------------------------===//

static Region *makeEmptyRegion(MIRContext &Ctx, Op *Parent) {
  Region *R = Ctx.newRegion();
  R->Parent = Parent;
  Block *B = Ctx.newBlock();
  B->Parent = R;
  R->Blocks.push_back(B);
  return R;
}

Op *Builder::ifOp(Value *Cond, SourceRange Loc) {
  Op *O = create(OpKind::IfOp, {Cond}, {}, Loc);
  O->Regions.push_back(makeEmptyRegion(Ctx, O)); // then
  O->Regions.push_back(makeEmptyRegion(Ctx, O)); // else (may remain empty)
  return O;
}

Op *Builder::forOp(std::string VarName, Value *Iter, SourceRange Loc) {
  Op *O = create(OpKind::ForOp, {Iter}, {}, Loc);
  if (!VarName.empty())
    O->setAttr("var", Attribute::ofSym(std::move(VarName)));
  O->Regions.push_back(makeEmptyRegion(Ctx, O)); // body
  return O;
}

Op *Builder::whileOp(SourceRange Loc) {
  Op *O = create(OpKind::WhileOp, {}, {}, Loc);
  O->Regions.push_back(makeEmptyRegion(Ctx, O)); // cond
  O->Regions.push_back(makeEmptyRegion(Ctx, O)); // body
  return O;
}

void Builder::yield(std::vector<Value *> Vs, SourceRange Loc) {
  (void)create(OpKind::Yield, std::move(Vs), {}, Loc);
}

void Builder::retOp(std::vector<Value *> Vs, SourceRange Loc) {
  (void)create(OpKind::Return, std::move(Vs), {}, Loc);
}

void Builder::breakOp(SourceRange Loc) {
  (void)create(OpKind::Break, {}, {}, Loc);
}
void Builder::continueOp(SourceRange Loc) {
  (void)create(OpKind::Continue, {}, {}, Loc);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

Op *Builder::funcOp(std::string Name, std::vector<const Type *> InTys,
                    std::vector<const Type *> OutTys, SourceRange Loc) {
  Op *O = create(OpKind::FuncOp, {}, {}, Loc);
  O->setAttr("name", Attribute::ofSym(std::move(Name)));
  // Encode arities as attributes; actual types go via block arguments / result
  // types on the FuncOp itself (stored in Attrs as joined strings for print).
  O->setAttr("num_inputs", Attribute::ofInt((int64_t)InTys.size()));
  O->setAttr("num_outputs", Attribute::ofInt((int64_t)OutTys.size()));
  // Create body region with entry block; add one argument per input.
  Region *Body = Ctx.newRegion();
  Body->Parent = O;
  O->Regions.push_back(Body);
  Block *Entry = Ctx.newBlock();
  Entry->Parent = Body;
  Body->Blocks.push_back(Entry);
  for (const Type *T : InTys) Ctx.addBlockArgument(Entry, T);
  // Record output types on the op via a side-vector-like attr encoding.
  // We don't emit a real "return type" list in the printer; we rely on the
  // Return op's operand types instead. OutTys is still carried for possible
  // use in future passes.
  (void)OutTys;
  return O;
}

} // namespace mir
} // namespace matlab
