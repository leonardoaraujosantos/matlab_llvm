#pragma once

#include "matlab/Basic/SourceManager.h"

#include <string>
#include <vector>

namespace matlab {

enum class DiagLevel { Note, Warning, Error };

struct Diagnostic {
  DiagLevel Level;
  SourceLocation Loc;
  SourceRange Range; // optional; Range.Begin may be invalid
  std::string Message;
};

class DiagnosticEngine {
public:
  explicit DiagnosticEngine(const SourceManager &SM) : SM(SM) {}

  void report(DiagLevel Level, SourceLocation Loc, std::string Message);
  void report(DiagLevel Level, SourceRange Range, std::string Message);

  void error(SourceLocation Loc, std::string Message) {
    report(DiagLevel::Error, Loc, std::move(Message));
  }
  void error(SourceRange Range, std::string Message) {
    report(DiagLevel::Error, Range, std::move(Message));
  }
  void warning(SourceLocation Loc, std::string Message) {
    report(DiagLevel::Warning, Loc, std::move(Message));
  }

  bool hasErrors() const { return ErrorCount > 0; }
  unsigned errorCount() const { return ErrorCount; }

  const std::vector<Diagnostic> &diagnostics() const { return Diags; }

  // Print all buffered diagnostics to stderr in clang-style.
  void printAll() const;

private:
  const SourceManager &SM;
  std::vector<Diagnostic> Diags;
  unsigned ErrorCount = 0;

  void printOne(const Diagnostic &D) const;
};

} // namespace matlab
