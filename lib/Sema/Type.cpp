#include "matlab/Sema/Type.h"

#include "matlab/AST/AST.h" // ClassDef, for ObjectType::Class->Name

#include <sstream>

namespace matlab {

//===----------------------------------------------------------------------===//
// Dtype helpers
//===----------------------------------------------------------------------===//

const char *dtypeName(Dtype D) {
  switch (D) {
  case Dtype::Unknown: return "?";
  case Dtype::Logical: return "logical";
  case Dtype::Char:    return "char";
  case Dtype::Double:  return "double";
  case Dtype::Single:  return "single";
  case Dtype::Complex: return "complex";
  case Dtype::Int8:    return "int8";
  case Dtype::Int16:   return "int16";
  case Dtype::Int32:   return "int32";
  case Dtype::Int64:   return "int64";
  case Dtype::UInt8:   return "uint8";
  case Dtype::UInt16:  return "uint16";
  case Dtype::UInt32:  return "uint32";
  case Dtype::UInt64:  return "uint64";
  case Dtype::Fixed:   return "fi"; // spec rendered separately by fixedSpecName
  }
  return "?";
}

std::string fixedSpecName(const FixedSpec &S) {
  std::ostringstream OS;
  OS << "numerictype(" << (S.Signed ? 1 : 0) << ',' << int(S.WordLength)
     << ',' << int(S.FractionLength) << ')';
  return OS.str();
}

bool isInteger(Dtype D) {
  switch (D) {
  case Dtype::Int8: case Dtype::Int16: case Dtype::Int32: case Dtype::Int64:
  case Dtype::UInt8: case Dtype::UInt16: case Dtype::UInt32: case Dtype::UInt64:
  case Dtype::Fixed:
    return true;
  default: return false;
  }
}

bool isFloating(Dtype D) {
  return D == Dtype::Double || D == Dtype::Single || D == Dtype::Complex;
}

bool isNumeric(Dtype D) { return isInteger(D) || isFloating(D); }

// MATLAB arithmetic promotion (approximate):
//   - Unknown + anything -> Unknown
//   - If either is Complex -> Complex
//   - If either is Double -> Double
//   - If either is Single -> Single
//   - If same integer kind -> that kind
//   - Logical + numeric -> numeric (scalar rules are complicated in MATLAB;
//     we approximate by taking the numeric side)
//   - Mixed integer kinds -> Unknown (MATLAB errors at runtime)
Dtype promoteDtype(Dtype A, Dtype B) {
  if (A == Dtype::Unknown || B == Dtype::Unknown) return Dtype::Unknown;
  // Fixed beats double in the MATLAB Fixed-Point Designer rules: a
  // fi+double expression casts the double to the fi's numerictype. The
  // *spec* must be resolved by the caller (TypeInference / LowerFixedPoint);
  // the dtype lattice only reports that the result is Fixed.
  if (A == Dtype::Fixed || B == Dtype::Fixed) return Dtype::Fixed;
  if (A == Dtype::Logical) A = Dtype::Double; // logical-as-numeric
  if (B == Dtype::Logical) B = Dtype::Double;
  if (A == Dtype::Char)    A = Dtype::Double; // char arithmetic yields double
  if (B == Dtype::Char)    B = Dtype::Double;
  if (A == Dtype::Complex || B == Dtype::Complex) return Dtype::Complex;
  if (A == Dtype::Double  || B == Dtype::Double)  return Dtype::Double;
  if (A == Dtype::Single  || B == Dtype::Single)  return Dtype::Single;
  if (A == B) return A;
  return Dtype::Unknown;
}

//===----------------------------------------------------------------------===//
// Shape helpers
//===----------------------------------------------------------------------===//

std::string Shape::toString() const {
  switch (K) {
  case Rank::Unknown: return "?";
  case Rank::Scalar:  return "scalar";
  case Rank::Vector: {
    std::ostringstream OS;
    OS << "vec[" << (Dims.empty() ? int64_t(-1) : Dims[0]) << "]";
    return OS.str();
  }
  case Rank::Matrix: {
    std::ostringstream OS;
    auto at = [&](size_t i) -> int64_t {
      return i < Dims.size() ? Dims[i] : -1;
    };
    OS << "mat[" << at(0) << "," << at(1) << "]";
    return OS.str();
  }
  case Rank::NDArray: {
    std::ostringstream OS;
    OS << "nd[";
    for (size_t i = 0; i < Dims.size(); ++i) {
      if (i) OS << ",";
      OS << Dims[i];
    }
    OS << "]";
    return OS.str();
  }
  }
  return "?";
}

static int64_t joinDim(int64_t A, int64_t B) {
  if (A == B) return A;
  return -1; // dynamic
}

Shape joinShape(const Shape &A, const Shape &B) {
  if (A == B) return A;
  if (A.K == Shape::Rank::Unknown) return B;
  if (B.K == Shape::Rank::Unknown) return A;
  if (A.K != B.K) {
    // Differing ranks merge to Unknown.
    return Shape::unknown();
  }
  Shape R;
  R.K = A.K;
  size_t N = std::max(A.Dims.size(), B.Dims.size());
  R.Dims.resize(N);
  for (size_t i = 0; i < N; ++i) {
    int64_t a = i < A.Dims.size() ? A.Dims[i] : -1;
    int64_t b = i < B.Dims.size() ? B.Dims[i] : -1;
    R.Dims[i] = joinDim(a, b);
  }
  return R;
}

// MATLAB implicit expansion (broadcasting) for element-wise ops.
// - A scalar broadcasts to any shape.
// - Equal shapes match.
// - Differing non-scalar shapes: result rank is the max; each dim is the max
//   where one side is 1 or dynamic, else dynamic if unclear.
Shape broadcastShape(const Shape &A, const Shape &B) {
  if (A.K == Shape::Rank::Scalar) return B;
  if (B.K == Shape::Rank::Scalar) return A;
  if (A.K == Shape::Rank::Unknown || B.K == Shape::Rank::Unknown)
    return Shape::unknown();
  if (A == B) return A;

  // Promote to the higher rank.
  Shape R;
  R.K = (A.K >= B.K) ? A.K : B.K;
  size_t N = std::max(A.Dims.size(), B.Dims.size());
  R.Dims.resize(N);
  for (size_t i = 0; i < N; ++i) {
    int64_t a = i < A.Dims.size() ? A.Dims[i] : 1;
    int64_t b = i < B.Dims.size() ? B.Dims[i] : 1;
    if (a == b)          R.Dims[i] = a;
    else if (a == 1)     R.Dims[i] = b;
    else if (b == 1)     R.Dims[i] = a;
    else                 R.Dims[i] = -1;
  }
  return R;
}

//===----------------------------------------------------------------------===//
// Type::toString
//===----------------------------------------------------------------------===//

std::string Type::toString() const {
  switch (K) {
  case Kind::Any: return "any";
  case Kind::Array: {
    auto &A = static_cast<const ArrayType &>(*this);
    std::string S;
    if (A.Elt == Dtype::Fixed && A.FxSpec)
      S = fixedSpecName(*A.FxSpec);
    else
      S = dtypeName(A.Elt);
    if (A.S.K == Shape::Rank::Scalar) return S;
    return S + ":" + A.S.toString();
  }
  case Kind::StringArray: {
    auto &A = static_cast<const StringArrayType &>(*this);
    if (A.S.K == Shape::Rank::Scalar) return "string";
    return "string:" + A.S.toString();
  }
  case Kind::Cell:        return "cell";
  case Kind::Struct:      return "struct";
  case Kind::FuncHandle:  return "@handle";
  case Kind::Numerictype: {
    auto &N = static_cast<const NumerictypeType &>(*this);
    std::ostringstream OS;
    OS << "numerictype(" << (N.Signed ? 1 : 0) << ',' << int(N.WordLength)
       << ',' << int(N.FractionLength) << ')';
    return OS.str();
  }
  case Kind::Fimath:      return "fimath";
  case Kind::Object: {
    auto &O = static_cast<const ObjectType &>(*this);
    return "object<" +
           (O.Class ? std::string(O.Class->Name) : std::string("?")) + ">";
  }
  }
  return "?";
}

//===----------------------------------------------------------------------===//
// TypeContext
//===----------------------------------------------------------------------===//

TypeContext::TypeContext() {
  AnyT = std::make_unique<AnyType>();
}
TypeContext::~TypeContext() = default;

bool TypeContext::ArrayKey::operator==(const ArrayKey &O) const {
  if (D != O.D || !(S == O.S)) return false;
  // FixedSpec equality only matters for Dtype::Fixed; for any other dtype
  // the field stays nullopt and compares equal trivially.
  return FxSpec == O.FxSpec;
}
size_t TypeContext::ArrayKeyHash::operator()(const ArrayKey &) const {
  return 0; // unused (linear scan)
}

template <typename T, typename... A>
T *TypeContext::own(A &&...as) {
  auto P = std::make_unique<T>(std::forward<A>(as)...);
  T *R = P.get();
  Owned.push_back(std::move(P));
  return R;
}

const ArrayType *TypeContext::scalar(Dtype D) {
  return arrayOf(D, Shape::scalar());
}

const ArrayType *TypeContext::arrayOf(Dtype D, Shape S) {
  for (auto &E : ArrayCache) {
    if (E.first.D == D && E.first.S == S && !E.first.FxSpec) return E.second;
  }
  auto *T = own<ArrayType>(D, S);
  ArrayCache.push_back({{D, std::move(S), std::nullopt}, T});
  return T;
}

const ArrayType *TypeContext::fixedScalar(FixedSpec Spec) {
  return fixedArray(Spec, Shape::scalar());
}

const ArrayType *TypeContext::fixedArray(FixedSpec Spec, Shape S) {
  for (auto &E : ArrayCache) {
    if (E.first.D == Dtype::Fixed && E.first.S == S && E.first.FxSpec &&
        *E.first.FxSpec == Spec)
      return E.second;
  }
  auto *T = own<ArrayType>(Dtype::Fixed, S, Spec);
  ArrayCache.push_back({{Dtype::Fixed, std::move(S), Spec}, T});
  return T;
}

const StringArrayType *TypeContext::stringScalar() {
  return stringArray(Shape::scalar());
}

const StringArrayType *TypeContext::stringArray(Shape S) {
  for (auto &E : StringCache) {
    if (E.first == S) return E.second;
  }
  auto *T = own<StringArrayType>(S);
  StringCache.push_back({std::move(S), T});
  return T;
}

const CellType *TypeContext::cellAny() {
  if (!CellAnyT) CellAnyT = own<CellType>();
  return CellAnyT;
}

const CellType *TypeContext::cellOf(const Type *Elt) {
  // No useful element type -> the shared singleton.
  if (!Elt || Elt->K == Type::Kind::Any) return cellAny();
  for (auto &E : CellOfCache)
    if (E.first == Elt) return E.second;
  auto *T = own<CellType>();
  T->ElementUpperBound = Elt;
  CellOfCache.push_back({Elt, T});
  return T;
}

const StructType *TypeContext::structAny() {
  if (!StructAnyT) StructAnyT = own<StructType>();
  return StructAnyT;
}

const FuncHandleType *TypeContext::funcHandle() {
  if (!FuncHandleT) FuncHandleT = own<FuncHandleType>();
  return FuncHandleT;
}

const ObjectType *TypeContext::objectOf(const ClassDef *CD) {
  for (auto &E : ObjectCache)
    if (E.first == CD) return E.second;
  auto *T = own<ObjectType>(CD);
  ObjectCache.push_back({CD, T});
  return T;
}

const NumerictypeType *TypeContext::numerictype(bool Signed, uint8_t WL,
                                                 int8_t FL) {
  /* Linear scan — these are tiny per-program (a handful at most). */
  for (auto &E : NumerictypeCache) {
    if (E->Signed == Signed && E->WordLength == WL && E->FractionLength == FL)
      return E;
  }
  auto *T = own<NumerictypeType>(Signed, WL, FL);
  NumerictypeCache.push_back(T);
  return T;
}

const FimathType *TypeContext::fimath(FixedSpec::Overflow OF,
                                       FixedSpec::Rounding RM) {
  for (auto &E : FimathCache) {
    if (E->OF == OF && E->RM == RM) return E;
  }
  auto *T = own<FimathType>(OF, RM);
  FimathCache.push_back(T);
  return T;
}

const Type *TypeContext::join(const Type *A, const Type *B) {
  if (!A) return B;
  if (!B) return A;
  if (A == B) return A;
  if (A->K == Type::Kind::Any || B->K == Type::Kind::Any) return any();
  if (A->K != B->K) return any();

  switch (A->K) {
  case Type::Kind::Array: {
    auto &AA = static_cast<const ArrayType &>(*A);
    auto &BB = static_cast<const ArrayType &>(*B);
    if (AA.Elt != BB.Elt) {
      // Promote numerically.
      Dtype P = promoteDtype(AA.Elt, BB.Elt);
      if (P == Dtype::Unknown) return any();
      // Fixed × non-Fixed mixes happen at binop sites where the non-Fixed
      // side is cast to the Fixed side's spec. Sema's join (used for
      // control-flow merges) doesn't have enough information to pick that
      // spec, so we report `any` and leave it to the binop visitor.
      if (P == Dtype::Fixed) return any();
      return arrayOf(P, joinShape(AA.S, BB.S));
    }
    if (AA.Elt == Dtype::Fixed) {
      // Both are fi: matching specs join shapes; mismatched specs fall back
      // to `any` (a real promotion needs §3.10 fimath rules — handled at
      // arithmetic sites, not here).
      if (AA.FxSpec && BB.FxSpec && *AA.FxSpec == *BB.FxSpec)
        return fixedArray(*AA.FxSpec, joinShape(AA.S, BB.S));
      if (!AA.FxSpec && !BB.FxSpec)
        return arrayOf(Dtype::Fixed, joinShape(AA.S, BB.S));
      return any();
    }
    return arrayOf(AA.Elt, joinShape(AA.S, BB.S));
  }
  case Type::Kind::StringArray: {
    auto &AA = static_cast<const StringArrayType &>(*A);
    auto &BB = static_cast<const StringArrayType &>(*B);
    return stringArray(joinShape(AA.S, BB.S));
  }
  case Type::Kind::Cell:       return cellAny();
  case Type::Kind::Struct:     return structAny();
  case Type::Kind::FuncHandle: return funcHandle();
  case Type::Kind::Numerictype:
  case Type::Kind::Fimath:
    /* Compile-time fi metadata objects don't meaningfully merge across
     * branches; defer to `any` (a control-flow merge on these almost
     * never happens — they're constructed once per use). */
    return any();
  case Type::Kind::Object: {
    /* Two object types of the SAME class are already pointer-equal (objectOf
     * interns per ClassDef) and caught by the `A == B` fast path above. A
     * merge of two DIFFERENT classes has no common static class — widen to
     * any (dispatch falls back to the runtime). */
    return any();
  }
  case Type::Kind::Any:        return any();
  }
  return any();
}

const Type *TypeContext::broadcastNumeric(const Type *A, const Type *B) {
  if (!A || !B) return any();
  if (A->K != Type::Kind::Array || B->K != Type::Kind::Array) return any();
  auto &AA = static_cast<const ArrayType &>(*A);
  auto &BB = static_cast<const ArrayType &>(*B);
  Dtype D = promoteDtype(AA.Elt, BB.Elt);
  /* MATLAB's native-int rule: when one side is a typed integer array and
   * the other is a double / single, the integer type wins (the f64 side
   * gets saturating-cast at the binop site). Apply only when at least
   * one side is non-scalar — scalar+scalar arithmetic stays on the f64
   * lane to match the existing scalar test fixtures. The Lowering layer
   * uses Bi.Ty (this result) plus operand expr types to pick the typed
   * runtime entry points (matlab_mat_i32_add_ms, etc.). */
  auto isIntElt = [](Dtype DD) {
    return DD == Dtype::Int8  || DD == Dtype::Int16 ||
           DD == Dtype::Int32 || DD == Dtype::Int64 ||
           DD == Dtype::UInt8 || DD == Dtype::UInt16 ||
           DD == Dtype::UInt32|| DD == Dtype::UInt64;
  };
  bool ANonScalar = AA.S.K != Shape::Rank::Scalar;
  bool BNonScalar = BB.S.K != Shape::Rank::Scalar;
  if ((ANonScalar || BNonScalar) &&
      (D == Dtype::Double || D == Dtype::Single)) {
    if (isIntElt(AA.Elt) && !isIntElt(BB.Elt)) D = AA.Elt;
    else if (isIntElt(BB.Elt) && !isIntElt(AA.Elt)) D = BB.Elt;
  }
  if (D == Dtype::Unknown) return any();
  Shape Out = broadcastShape(AA.S, BB.S);
  if (D == Dtype::Fixed) {
    // Pick a concrete FixedSpec when one side has it. The full §3.10–3.14
    // rule (KeepLSB sum: result FL = max(FL_a, FL_b), result WL grows by
    // 1 for add/sub) is applied at the binop site in TypeInference where
    // the operator is known. Here we only handle the trivial case "both
    // are fi with matching spec" — a wider mix returns the spec from the
    // fi operand when the other side is non-fi (mixed fi+double).
    const std::optional<FixedSpec> &La = AA.FxSpec;
    const std::optional<FixedSpec> &Rb = BB.FxSpec;
    if (La && Rb) {
      if (*La == *Rb) return fixedArray(*La, Out);
      // Differing specs: defer to arithmetic-site rules.
      return arrayOf(Dtype::Fixed, Out);
    }
    if (La) return fixedArray(*La, Out);
    if (Rb) return fixedArray(*Rb, Out);
    return arrayOf(Dtype::Fixed, Out);
  }
  return arrayOf(D, Out);
}

const Type *scalarOf(TypeContext &C, Dtype D) { return C.scalar(D); }

} // namespace matlab
