#include "matlab/Sema/Scope.h"

#include <string>

namespace matlab {

const char *bindingKindName(BindingKind K) {
  switch (K) {
  case BindingKind::Var:        return "var";
  case BindingKind::Param:      return "param";
  case BindingKind::Output:     return "output";
  case BindingKind::Global:     return "global";
  case BindingKind::Persistent: return "persistent";
  case BindingKind::Function:   return "function";
  case BindingKind::Builtin:    return "builtin";
  case BindingKind::Import:     return "import";
  }
  return "?";
}

Binding *Scope::lookupLocal(std::string_view N) {
  auto It = Bindings.find(std::string(N));
  return It == Bindings.end() ? nullptr : It->second;
}

Binding *Scope::lookup(std::string_view N) {
  for (Scope *S = this; S; S = S->Parent) {
    if (auto *B = S->lookupLocal(N)) return B;
  }
  return nullptr;
}

Binding *Scope::declare(std::string_view N, BindingKind K, Binding *Owned) {
  auto [It, Inserted] = Bindings.emplace(std::string(N), Owned);
  if (Inserted) {
    Owned->Kind = K;
    Owned->Name = N;
    return Owned;
  }
  // Already declared — preserve existing binding's kind (first declaration wins
  // for Function/Param etc.). If the prior was Var and caller is re-declaring
  // as something stronger (Param/Global/Persistent/Function), promote.
  Binding *Prev = It->second;
  if (Prev->Kind == BindingKind::Var && K != BindingKind::Var)
    Prev->Kind = K;
  return Prev;
}

Binding *Scope::getOrDeclareVar(std::string_view N, Binding *Owned) {
  if (auto *Prev = lookupLocal(N)) return Prev;
  return declare(N, BindingKind::Var, Owned);
}

//===----------------------------------------------------------------------===//
// SemaContext
//===----------------------------------------------------------------------===//

SemaContext::SemaContext() = default;
SemaContext::~SemaContext() = default;

Scope *SemaContext::newScope(Scope *Parent, std::string Name) {
  auto S = std::make_unique<Scope>(Parent, std::move(Name));
  Scope *P = S.get();
  Scopes.push_back(std::move(S));
  return P;
}

Binding *SemaContext::newBinding() {
  auto B = std::make_unique<Binding>();
  Binding *P = B.get();
  Bindings.push_back(std::move(B));
  return P;
}

std::string_view SemaContext::intern(std::string_view S) {
  auto Str = std::make_unique<std::string>(S);
  std::string_view V = *Str;
  Strings.push_back(std::move(Str));
  return V;
}

} // namespace matlab
