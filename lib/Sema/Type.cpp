#include "matlab/Sema/Type.h"

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
  }
  return "?";
}

bool isInteger(Dtype D) {
  switch (D) {
  case Dtype::Int8: case Dtype::Int16: case Dtype::Int32: case Dtype::Int64:
  case Dtype::UInt8: case Dtype::UInt16: case Dtype::UInt32: case Dtype::UInt64:
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
    std::string S = dtypeName(A.Elt);
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
  return D == O.D && S == O.S;
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
    if (E.first.D == D && E.first.S == S) return E.second;
  }
  auto *T = own<ArrayType>(D, S);
  ArrayCache.push_back({{D, std::move(S)}, T});
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

const StructType *TypeContext::structAny() {
  if (!StructAnyT) StructAnyT = own<StructType>();
  return StructAnyT;
}

const FuncHandleType *TypeContext::funcHandle() {
  if (!FuncHandleT) FuncHandleT = own<FuncHandleType>();
  return FuncHandleT;
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
      return arrayOf(P, joinShape(AA.S, BB.S));
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
  if (D == Dtype::Unknown) return any();
  return arrayOf(D, broadcastShape(AA.S, BB.S));
}

const Type *scalarOf(TypeContext &C, Dtype D) { return C.scalar(D); }

} // namespace matlab
