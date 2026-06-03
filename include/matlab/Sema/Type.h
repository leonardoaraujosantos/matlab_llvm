#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace matlab {

class ClassDef; // user classdef AST node; ObjectType references it by pointer.

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
  Fixed,  // MATLAB Fixed-Point Designer `fi` — see FixedSpec on ArrayType.
};

const char *dtypeName(Dtype D);
bool isInteger(Dtype D);
bool isFloating(Dtype D);
bool isNumeric(Dtype D);
Dtype promoteDtype(Dtype A, Dtype B); // arithmetic promotion; Unknown if unresolvable

//===----------------------------------------------------------------------===//
// FixedSpec — Fixed-Point Designer numerictype + fimath parameters.
//
// Only meaningful when the parent ArrayType has Elt == Dtype::Fixed. The
// rendered name is `numerictype(s,WL,FL)`; arithmetic / overflow / rounding
// modes ride alongside on the same struct so binop promotion and the
// LowerFixedPoint pass can read them in one place.
//===----------------------------------------------------------------------===//

struct FixedSpec {
  bool Signed = true;
  uint8_t WordLength = 16;     // 1..64; sub-native widths are legal in the
                               // type but emitted code stores in the smallest
                               // containing native int.
  int8_t  FractionLength = 15; // 0..WL; negative deferred (see plan §3.5).
  // Wrap = 0, Saturate = 1 — aligned with the runtime ABI (matlab_fi_*).
  enum class Overflow : uint8_t { Wrap = 0, Saturate = 1 } OF = Overflow::Saturate;
  enum class Rounding : uint8_t {
    Floor, Nearest, Zero, Convergent, Ceiling
  } RM = Rounding::Floor;

  bool operator==(const FixedSpec &O) const {
    return Signed == O.Signed && WordLength == O.WordLength &&
           FractionLength == O.FractionLength && OF == O.OF && RM == O.RM;
  }
  bool operator!=(const FixedSpec &O) const { return !(*this == O); }

  // Number of integer-magnitude bits (== WL - FL - Signed). May be negative
  // for "scaling smaller than 1 LSB"; we don't ship that case yet but the
  // arithmetic rules still need the value as signed.
  int  integerLength() const {
    return int(WordLength) - int(FractionLength) - (Signed ? 1 : 0);
  }
  // Smallest native storage class that can hold WL bits with this signedness.
  // Returns 8 / 16 / 32 / 64; 0 if WL == 0 (not a legal fi but defensively
  // handled).
  uint8_t storageBits() const {
    if (WordLength == 0) return 0;
    if (WordLength <= 8)  return 8;
    if (WordLength <= 16) return 16;
    if (WordLength <= 32) return 32;
    return 64;
  }
};

// Render `numerictype(1,16,8)` form. Used by dtype names + toString.
std::string fixedSpecName(const FixedSpec &S);

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
  // Rank >= 3 array. Dims carries one entry per dimension (-1 = unknown
  // extent). A rank-N constructor result (e.g. zeros(a,b,c)) must carry a
  // *positive* NDArray rank so the clone-on-assign gate in Lowering keys off
  // it to deep-copy the matlab_matN buffer on `B = A` (issue #102).
  static Shape ndarray(std::vector<int64_t> Dims) {
    Shape S; S.K = Rank::NDArray; S.Dims = std::move(Dims); return S;
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
    Numerictype,  // Fixed-Point Designer numerictype object — compile-time only
    Fimath,       // Fixed-Point Designer fimath object — compile-time only
    Object,       // instance of a user classdef (lowers to a matlab_obj* ptr)
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
  /// Only meaningful when Elt == Dtype::Fixed. Carries the WL/FL/signedness/
  /// overflow/rounding parameters of the fi value. nullopt is the
  /// "don't-care" state used by joins where the spec is unresolved.
  std::optional<FixedSpec> FxSpec;

  ArrayType(Dtype D, Shape Sh) : Type(Kind::Array), Elt(D), S(std::move(Sh)) {}
  ArrayType(Dtype D, Shape Sh, FixedSpec Spec)
      : Type(Kind::Array), Elt(D), S(std::move(Sh)), FxSpec(Spec) {}
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

/// Instance of a user classdef (e.g. the result of a `tf(...)` constructor
/// call, an operator overload, or a method returning the class). Carries the
/// owning ClassDef so the monomorphizer can bucket constructor call sites by
/// concrete class and the type mapper can lower it to `matlab_obj*` (ptr).
/// This is what lets class-valued expressions carry a static type instead of
/// the old `Any` + `Binding::PinnedClass` side channel — see docs/sema.md §4.
class ObjectType : public Type {
public:
  const ClassDef *Class = nullptr;
  explicit ObjectType(const ClassDef *CD) : Type(Kind::Object), Class(CD) {}
};

/// Fixed-Point Designer `numerictype(s, WL, FL)` object — pure compile-time
/// metadata. Carries only the WL/FL/signedness; overflow + rounding live on
/// FimathType. Has no runtime representation: any consumer (`fi(value, T)`,
/// property access) reads the spec out at type-inference time and folds.
class NumerictypeType : public Type {
public:
  bool Signed = true;
  uint8_t WordLength = 16;
  int8_t  FractionLength = 15;
  NumerictypeType(bool S, uint8_t WL, int8_t FL)
      : Type(Kind::Numerictype), Signed(S), WordLength(WL),
        FractionLength(FL) {}
};

/// Fixed-Point Designer `fimath` object — pure compile-time metadata.
/// Holds overflow + rounding overrides that fi() consumers apply on top
/// of a numerictype.
class FimathType : public Type {
public:
  FixedSpec::Overflow OF = FixedSpec::Overflow::Saturate;
  FixedSpec::Rounding RM = FixedSpec::Rounding::Floor;
  FimathType(FixedSpec::Overflow O, FixedSpec::Rounding R)
      : Type(Kind::Fimath), OF(O), RM(R) {}
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
  // Fixed-point variants — Elt is Dtype::Fixed and the FixedSpec is interned
  // alongside the shape so two requests with the same spec hash-equal.
  const ArrayType *fixedScalar(FixedSpec Spec);
  const ArrayType *fixedArray(FixedSpec Spec, Shape S);
  // Compile-time fi metadata objects.
  const NumerictypeType *numerictype(bool Signed, uint8_t WL, int8_t FL);
  const FimathType *fimath(FixedSpec::Overflow OF, FixedSpec::Rounding RM);

  const StringArrayType *stringScalar();
  const StringArrayType *stringArray(Shape S);
  const CellType *cellAny();
  const StructType *structAny();
  const FuncHandleType *funcHandle();
  // Instance of a user classdef, interned per ClassDef so two requests for
  // the same class hash-equal (constructor-call bucketing relies on this).
  const ObjectType *objectOf(const ClassDef *CD);

  // Join two types (control-flow merge).
  const Type *join(const Type *A, const Type *B);

  // Element-wise broadcast typing for numeric/logical/char ops.
  const Type *broadcastNumeric(const Type *A, const Type *B);

private:
  std::unique_ptr<AnyType> AnyT;
  std::vector<std::unique_ptr<Type>> Owned;
  // Simple memoization.
  struct ArrayKey {
    Dtype D;
    Shape S;
    std::optional<FixedSpec> FxSpec;
    bool operator==(const ArrayKey&) const;
  };
  struct ArrayKeyHash { size_t operator()(const ArrayKey &K) const; };
  // We keep a vector here because Shape::Dims order matters; linear scan is
  // fine for our sizes.
  std::vector<std::pair<ArrayKey, const ArrayType *>> ArrayCache;
  std::vector<std::pair<Shape, const StringArrayType *>> StringCache;
  const CellType *CellAnyT = nullptr;
  const StructType *StructAnyT = nullptr;
  const FuncHandleType *FuncHandleT = nullptr;
  std::vector<const NumerictypeType *> NumerictypeCache;
  std::vector<const FimathType *> FimathCache;
  std::vector<std::pair<const ClassDef *, const ObjectType *>> ObjectCache;

  template <typename T, typename... A> T *own(A &&...as);
};

const Type *scalarOf(TypeContext &C, Dtype D);

} // namespace matlab
