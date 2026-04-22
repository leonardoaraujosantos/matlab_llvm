#pragma once

#include "matlab/Basic/SourceManager.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace matlab {

// Forward-declared Sema types. Non-owning pointers on AST nodes are populated
// by the resolver / type inference passes. Parsing alone leaves them null.
class Type;
struct Binding;

//===----------------------------------------------------------------------===//
// Node kinds
//===----------------------------------------------------------------------===//

enum class NodeKind : uint16_t {
  // Expressions
  IntegerLiteral,
  FPLiteral,
  ImagLiteral,
  StringLiteral,
  CharLiteral,
  NameExpr,
  EndExpr,
  ColonExpr,
  BinaryOp,
  UnaryOp,
  PostfixOp,        // postfix ' and .'
  RangeExpr,
  CallOrIndex,
  CellIndex,
  FieldAccess,
  DynamicField,
  MatrixLiteral,
  CellLiteral,
  AnonFunction,
  FuncHandle,

  // Statements
  ExprStmt,
  AssignStmt,
  IfStmt,
  ForStmt,
  WhileStmt,
  SwitchStmt,
  TryStmt,
  ReturnStmt,
  BreakStmt,
  ContinueStmt,
  GlobalDecl,
  PersistentDecl,
  ImportStmt,
  CommandStmt,
  Block,

  // Top-level
  Function,
  Script,
  TranslationUnit,
};

//===----------------------------------------------------------------------===//
// Binary and unary operators
//===----------------------------------------------------------------------===//

enum class BinOp {
  // arithmetic
  Add, Sub, Mul, Div, LeftDiv, Pow,
  ElemMul, ElemDiv, ElemLeftDiv, ElemPow,
  // comparison
  Eq, Ne, Lt, Le, Gt, Ge,
  // logical
  And, Or, ShortAnd, ShortOr,
  // colon range is a special node, not a BinOp
};

enum class UnOp {
  Plus, Minus, Not,
};

enum class PostfixOp {
  CTranspose,    // '
  Transpose,     // .'
};

const char *binOpName(BinOp);
const char *unOpName(UnOp);
const char *postfixName(PostfixOp);
const char *nodeKindName(NodeKind);

//===----------------------------------------------------------------------===//
// Node hierarchy
//===----------------------------------------------------------------------===//

class ASTContext;

class Node {
public:
  NodeKind Kind;
  SourceRange Range;

  explicit Node(NodeKind K) : Kind(K) {}
  virtual ~Node() = default;
};

class Expr : public Node {
public:
  using Node::Node;
  const Type *Ty = nullptr; // inferred by Sema; nullptr if not yet inferred
};

class Stmt : public Node {
public:
  using Node::Node;
};

//===----------------------------------------------------------------------===//
// Literals / names
//===----------------------------------------------------------------------===//

class IntegerLiteral : public Expr {
public:
  std::string_view Text;          // raw spelling; value conversion in Sema
  IntegerLiteral() : Expr(NodeKind::IntegerLiteral) {}
};

class FPLiteral : public Expr {
public:
  std::string_view Text;
  FPLiteral() : Expr(NodeKind::FPLiteral) {}
};

class ImagLiteral : public Expr {
public:
  std::string_view Text;
  ImagLiteral() : Expr(NodeKind::ImagLiteral) {}
};

class StringLiteral : public Expr {
public:
  std::string Value;               // unescaped contents
  StringLiteral() : Expr(NodeKind::StringLiteral) {}
};

class CharLiteral : public Expr {
public:
  std::string Value;
  CharLiteral() : Expr(NodeKind::CharLiteral) {}
};

class NameExpr : public Expr {
public:
  std::string_view Name;
  Binding *Ref = nullptr; // resolved by the Resolver
  NameExpr() : Expr(NodeKind::NameExpr) {}
};

class EndExpr : public Expr {
public:
  EndExpr() : Expr(NodeKind::EndExpr) {}
};

class ColonExpr : public Expr { // bare `:` used as argument, e.g. a(:)
public:
  ColonExpr() : Expr(NodeKind::ColonExpr) {}
};

//===----------------------------------------------------------------------===//
// Operators
//===----------------------------------------------------------------------===//

class BinaryOpExpr : public Expr {
public:
  BinOp Op;
  Expr *LHS = nullptr;
  Expr *RHS = nullptr;
  BinaryOpExpr() : Expr(NodeKind::BinaryOp) {}
};

class UnaryOpExpr : public Expr {
public:
  UnOp Op;
  Expr *Operand = nullptr;
  UnaryOpExpr() : Expr(NodeKind::UnaryOp) {}
};

class PostfixOpExpr : public Expr {
public:
  PostfixOp Op;
  Expr *Operand = nullptr;
  PostfixOpExpr() : Expr(NodeKind::PostfixOp) {}
};

class RangeExpr : public Expr {
public:
  Expr *Start = nullptr;
  Expr *Step = nullptr; // nullable (two-operand range)
  Expr *End = nullptr;
  RangeExpr() : Expr(NodeKind::RangeExpr) {}
};

//===----------------------------------------------------------------------===//
// Postfix suffixes: calls / indexing / fields
//===----------------------------------------------------------------------===//

// Parser emits every `expr(args)` as CallOrIndex. The resolver sets Resolved
// to Call (static function dispatch) or Index (array subscript).
enum class CallKind : uint8_t { Unresolved, Call, Index };

class CallOrIndex : public Expr {
public:
  Expr *Callee = nullptr;
  std::vector<Expr *> Args;
  CallKind Resolved = CallKind::Unresolved;
  CallOrIndex() : Expr(NodeKind::CallOrIndex) {}
};

class CellIndex : public Expr {
public:
  Expr *Callee = nullptr;
  std::vector<Expr *> Args;
  CellIndex() : Expr(NodeKind::CellIndex) {}
};

class FieldAccess : public Expr {
public:
  Expr *Base = nullptr;
  std::string_view Field;
  FieldAccess() : Expr(NodeKind::FieldAccess) {}
};

class DynamicField : public Expr {
public:
  Expr *Base = nullptr;
  Expr *Name = nullptr;
  DynamicField() : Expr(NodeKind::DynamicField) {}
};

//===----------------------------------------------------------------------===//
// Aggregate literals
//===----------------------------------------------------------------------===//

class MatrixLiteral : public Expr {
public:
  std::vector<std::vector<Expr *>> Rows;
  MatrixLiteral() : Expr(NodeKind::MatrixLiteral) {}
};

class CellLiteral : public Expr {
public:
  std::vector<std::vector<Expr *>> Rows;
  CellLiteral() : Expr(NodeKind::CellLiteral) {}
};

//===----------------------------------------------------------------------===//
// Function handles / anonymous functions
//===----------------------------------------------------------------------===//

class FuncHandle : public Expr {
public:
  std::string_view Name;            // e.g. @sin
  Binding *Ref = nullptr;           // resolved by the Resolver
  FuncHandle() : Expr(NodeKind::FuncHandle) {}
};

class AnonFunction : public Expr {
public:
  std::vector<std::string_view> Params;
  Expr *Body = nullptr;
  AnonFunction() : Expr(NodeKind::AnonFunction) {}
};

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

class ExprStmt : public Stmt {
public:
  Expr *E = nullptr;
  bool Suppressed = false;
  ExprStmt() : Stmt(NodeKind::ExprStmt) {}
};

class AssignStmt : public Stmt {
public:
  std::vector<Expr *> LHS; // one element (scalar) or many (multi-assign)
  Expr *RHS = nullptr;
  bool Suppressed = false;
  AssignStmt() : Stmt(NodeKind::AssignStmt) {}
};

class Block : public Stmt {
public:
  std::vector<Stmt *> Stmts;
  Block() : Stmt(NodeKind::Block) {}
};

struct ElseIf {
  Expr *Cond = nullptr;
  Block *Body = nullptr;
};

class IfStmt : public Stmt {
public:
  Expr *Cond = nullptr;
  Block *Then = nullptr;
  std::vector<ElseIf> Elseifs;
  Block *Else = nullptr; // nullable
  IfStmt() : Stmt(NodeKind::IfStmt) {}
};

class ForStmt : public Stmt {
public:
  std::string_view Var;
  Expr *Iter = nullptr;
  Block *Body = nullptr;
  bool IsParfor = false;
  ForStmt() : Stmt(NodeKind::ForStmt) {}
};

class WhileStmt : public Stmt {
public:
  Expr *Cond = nullptr;
  Block *Body = nullptr;
  WhileStmt() : Stmt(NodeKind::WhileStmt) {}
};

struct SwitchCase {
  Expr *Value = nullptr; // nullable for `otherwise`
  Block *Body = nullptr;
};

class SwitchStmt : public Stmt {
public:
  Expr *Discriminant = nullptr;
  std::vector<SwitchCase> Cases;
  SwitchStmt() : Stmt(NodeKind::SwitchStmt) {}
};

class TryStmt : public Stmt {
public:
  Block *TryBody = nullptr;
  std::string_view CatchVar;  // empty if no binding
  Block *CatchBody = nullptr; // nullable
  TryStmt() : Stmt(NodeKind::TryStmt) {}
};

class ReturnStmt : public Stmt {
public:
  ReturnStmt() : Stmt(NodeKind::ReturnStmt) {}
};
class BreakStmt : public Stmt {
public:
  BreakStmt() : Stmt(NodeKind::BreakStmt) {}
};
class ContinueStmt : public Stmt {
public:
  ContinueStmt() : Stmt(NodeKind::ContinueStmt) {}
};

class GlobalDecl : public Stmt {
public:
  std::vector<std::string_view> Names;
  GlobalDecl() : Stmt(NodeKind::GlobalDecl) {}
};

class PersistentDecl : public Stmt {
public:
  std::vector<std::string_view> Names;
  PersistentDecl() : Stmt(NodeKind::PersistentDecl) {}
};

class ImportStmt : public Stmt {
public:
  std::vector<std::string_view> Path;
  bool Wildcard = false;
  ImportStmt() : Stmt(NodeKind::ImportStmt) {}
};

class CommandStmt : public Stmt {
public:
  std::string_view Name;
  std::vector<std::string> Args; // bare-word arguments
  bool Suppressed = false;
  CommandStmt() : Stmt(NodeKind::CommandStmt) {}
};

//===----------------------------------------------------------------------===//
// Top-level
//===----------------------------------------------------------------------===//

class Scope; // Sema forward decl

class Function : public Node {
public:
  std::string_view Name;
  std::vector<std::string_view> Inputs;
  std::vector<std::string_view> Outputs;
  Block *Body = nullptr;
  std::vector<Function *> Nested;

  // Populated by Sema.
  Scope *FnScope = nullptr;          // owns Bindings for this function
  Binding *Self = nullptr;           // this function's own binding
  std::vector<Binding *> ParamRefs;  // one per Inputs, in order
  std::vector<Binding *> OutputRefs; // one per Outputs, in order

  Function() : Node(NodeKind::Function) {}
};

class Script : public Node {
public:
  Block *Body = nullptr;
  Script() : Node(NodeKind::Script) {}
};

class TranslationUnit : public Node {
public:
  // Exactly one of Script or top-level Functions is populated in typical files.
  Script *ScriptNode = nullptr;
  std::vector<Function *> Functions;
  TranslationUnit() : Node(NodeKind::TranslationUnit) {}
};

//===----------------------------------------------------------------------===//
// ASTContext — owns all nodes via a bump allocator.
//===----------------------------------------------------------------------===//

class ASTContext {
public:
  ASTContext();
  ~ASTContext();
  ASTContext(const ASTContext &) = delete;
  ASTContext &operator=(const ASTContext &) = delete;

  template <typename T, typename... Args>
  T *make(Args &&...As) {
    void *Mem = allocate(sizeof(T), alignof(T));
    T *N = new (Mem) T(std::forward<Args>(As)...);
    registerNode(N);
    return N;
  }

  // Intern a string into a stable buffer and return a view.
  std::string_view intern(std::string_view S);

private:
  void *allocate(size_t Size, size_t Align);
  void registerNode(Node *N);

  struct Slab {
    std::unique_ptr<char[]> Data;
    size_t Size = 0;
    size_t Used = 0;
  };
  std::vector<Slab> Slabs;
  std::vector<Node *> Nodes;            // for destructor dispatch
  std::vector<std::unique_ptr<std::string>> Strings;
};

} // namespace matlab
