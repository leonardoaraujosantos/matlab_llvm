#pragma once

#include "matlab/Basic/SourceManager.h"

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace matlab {

class Function;
class Type;

enum class BindingKind : uint8_t {
  Var,        // assigned somewhere in this function's body
  Param,      // input parameter
  Output,     // output parameter (must be assigned before function exits)
  Global,     // declared `global`
  Persistent, // declared `persistent`
  Function,   // TU-level or nested function (user-defined)
  Builtin,    // known MATLAB builtin (extern)
  Import,     // `import pkg.name` alias (unused for now)
};

const char *bindingKindName(BindingKind K);

struct Binding {
  BindingKind Kind = BindingKind::Var;
  std::string_view Name;
  SourceLocation FirstUse;
  matlab::Function *FuncDef = nullptr; // for Kind=Function; null for Builtin
  const Type *DeclaredType = nullptr;  // optional, e.g. for builtins
  // Type inference fills in the current inferred type here after analysis.
  const Type *InferredType = nullptr;
  bool WrittenTo = false;
  bool ReadFrom = false;
};

class Scope {
public:
  explicit Scope(Scope *Parent = nullptr, std::string Name = "")
      : Parent(Parent), Name(std::move(Name)) {}

  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

  // Lookup walks up the parent chain.
  Binding *lookup(std::string_view N);
  Binding *lookupLocal(std::string_view N);

  // Declare a new binding in this scope (no-op if already present with same
  // kind — returns existing binding in that case, promoting Var to keep Kind).
  Binding *declare(std::string_view N, BindingKind K, Binding *Owned);

  // Retrieve or auto-declare as Var (used at first assignment).
  Binding *getOrDeclareVar(std::string_view N, Binding *Owned);

  Scope *parent() const { return Parent; }
  const std::string &name() const { return Name; }

  // Iteration for the dumper.
  const std::unordered_map<std::string, Binding *> &locals() const {
    return Bindings;
  }

private:
  Scope *Parent;
  std::string Name;
  std::unordered_map<std::string, Binding *> Bindings;
};

// Owns all Sema-created objects (Scopes, Bindings, Types).
class SemaContext {
public:
  SemaContext();
  ~SemaContext();
  SemaContext(const SemaContext &) = delete;
  SemaContext &operator=(const SemaContext &) = delete;

  Scope *newScope(Scope *Parent, std::string Name = "");
  Binding *newBinding();

  // String interning for names whose storage must outlive the source buffer
  // (generally unnecessary since source outlives Sema, but available).
  std::string_view intern(std::string_view S);

private:
  std::vector<std::unique_ptr<Scope>> Scopes;
  std::vector<std::unique_ptr<Binding>> Bindings;
  std::vector<std::unique_ptr<std::string>> Strings;
};

} // namespace matlab
