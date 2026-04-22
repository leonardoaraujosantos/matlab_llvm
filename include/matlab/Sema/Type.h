#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace matlab {

//===----------------------------------------------------------------------===//
// Dtype — element type of a numeric/logical/char array.
//===----------------------------------------------------------------------===//

enum class Dtype : uint8_t {
  Unknown,
  Logical,
  Char,
  Double,
  Single,
  Complex, // complex double
  Int8,  Int16,  Int32,  Int64,
  UInt8, UInt16, UInt32, UInt64,
};

const char *dtypeName(Dtype D);
bool isInteger(Dtype D);
bool isFloating(Dtype D);
bool isNumeric(Dtype D);
Dtype promoteDtype(Dtype A, Dtype B); // arithmetic promotion; Unknown if unresolvable

//===----------------------------------------------------------------------===//
// Shape — rank + per-dim extents. -1 denotes dynamic / unknown extent.
//===----------------------------------------------------------------------===//

struct Shape {
  enum class Rank : uint8_t { Unknown, Scalar, Vector, Matrix, NDArray };
  Rank K = Rank::Unknown;
  std::vector<int64_t> Dims; // empty for Scalar; size() >= 1 otherwise

  bool operator==(const Shape &O) const {
    return K == O.K && Dims == O.Dims;
  }
  static Shape unknown() { return {}; }
  static Shape scalar()  { Shape S; S.K = Rank::Scalar; return S; }
  static Shape vector(int64_t N) {
    Shape S; S.K = Rank::Vector; S.Dims = {N}; return S;
  }
  static Shape matrix(int64_t R, int64_t C) {
    Shape S; S.K = Rank::Matrix; S.Dims = {R, C}; return S;
  }

  std::string toString() const;
};

Shape joinShape(const Shape &A, const Shape &B);       // LUB for merges
Shape broadcastShape(const Shape &A, const Shape &B);  // element-wise ops

//===----------------------------------------------------------------------===//
// Type hierarchy
//===----------------------------------------------------------------------===//

class Type {
public:
  enum class Kind : uint8_t {
    Any,          // top element
    Array,        // numeric / logical / char array with Dtype + Shape
    StringArray,  // "..." string array (element = string scalar)
    Cell,         // cell array (heterogeneous)
    Struct,       // named-field record
    FuncHandle,   // function handle (signature left opaque for now)
  };
  Kind K;
  explicit Type(Kind K) : K(K) {}
  virtual ~Type() = default;

  std::string toString() const;
};

class AnyType : public Type {
public:
  AnyType() : Type(Kind::Any) {}
};

class ArrayType : public Type {
public:
  Dtype Elt = Dtype::Unknown;
  Shape S;
  ArrayType(Dtype D, Shape Sh) : Type(Kind::Array), Elt(D), S(std::move(Sh)) {}
};

class StringArrayType : public Type {
public:
  Shape S; // shape of the outer string array
  explicit StringArrayType(Shape Sh = Shape::scalar())
      : Type(Kind::StringArray), S(std::move(Sh)) {}
};

class CellType : public Type {
public:
  Shape S;                         // outer cell-array shape
  const Type *ElementUpperBound = nullptr; // join of known element types
  CellType() : Type(Kind::Cell) {}
};

class StructType : public Type {
public:
  std::map<std::string, const Type *> Fields; // known fields
  bool OpenSet = true; // true => may have additional dynamic fields
  StructType() : Type(Kind::Struct) {}
};

class FuncHandleType : public Type {
public:
  // For now: opaque. Later we'll track parameter arity / types.
  int NumInputs = -1;  // -1 = unknown
  int NumOutputs = -1;
  FuncHandleType() : Type(Kind::FuncHandle) {}
};

//===----------------------------------------------------------------------===//
// TypeContext — owns Type objects and provides interned singletons.
//===----------------------------------------------------------------------===//

class TypeContext {
public:
  TypeContext();
  ~TypeContext();

  const AnyType *any() const { return AnyT.get(); }

  // Commonly-used interned types.
  const ArrayType *scalar(Dtype D);
  const ArrayType *arrayOf(Dtype D, Shape S);
  const StringArrayType *stringScalar();
  const StringArrayType *stringArray(Shape S);
  const CellType *cellAny();
  const StructType *structAny();
  const FuncHandleType *funcHandle();

  // Join two types (control-flow merge).
  const Type *join(const Type *A, const Type *B);

  // Element-wise broadcast typing for numeric/logical/char ops.
  const Type *broadcastNumeric(const Type *A, const Type *B);

private:
  std::unique_ptr<AnyType> AnyT;
  std::vector<std::unique_ptr<Type>> Owned;
  // Simple memoization.
  struct ArrayKey { Dtype D; Shape S; bool operator==(const ArrayKey&) const; };
  struct ArrayKeyHash { size_t operator()(const ArrayKey &K) const; };
  // We keep a vector here because Shape::Dims order matters; linear scan is
  // fine for our sizes.
  std::vector<std::pair<ArrayKey, const ArrayType *>> ArrayCache;
  std::vector<std::pair<Shape, const StringArrayType *>> StringCache;
  const CellType *CellAnyT = nullptr;
  const StructType *StructAnyT = nullptr;
  const FuncHandleType *FuncHandleT = nullptr;

  template <typename T, typename... A> T *own(A &&...as);
};

const Type *scalarOf(TypeContext &C, Dtype D);

} // namespace matlab
