#include "matlab/Basic/Diagnostic.h"

#include <iostream>

namespace matlab {

void DiagnosticEngine::report(DiagLevel Level, SourceLocation Loc,
                              std::string Message) {
  if (Level == DiagLevel::Error)
    ++ErrorCount;
  Diags.push_back({Level, Loc, {}, std::move(Message)});
}

void DiagnosticEngine::report(DiagLevel Level, SourceRange Range,
                              std::string Message) {
  if (Level == DiagLevel::Error)
    ++ErrorCount;
  Diags.push_back({Level, Range.Begin, Range, std::move(Message)});
}

static const char *levelName(DiagLevel L) {
  switch (L) {
  case DiagLevel::Note:    return "note";
  case DiagLevel::Warning: return "warning";
  case DiagLevel::Error:   return "error";
  }
  return "?";
}

void DiagnosticEngine::printOne(const Diagnostic &D) const {
  if (D.Loc.isValid()) {
    auto LC = SM.getLineColumn(D.Loc);
    std::cerr << SM.getName(D.Loc.File) << ':' << LC.Line << ':' << LC.Column
              << ": ";
  }
  std::cerr << levelName(D.Level) << ": " << D.Message << '\n';

  if (D.Loc.isValid()) {
    auto LC = SM.getLineColumn(D.Loc);
    auto Line = SM.getLineText(D.Loc.File, LC.Line);
    std::cerr << "  " << Line << '\n';
    std::cerr << "  ";
    for (uint32_t i = 1; i < LC.Column; ++i)
      std::cerr << ' ';
    std::cerr << '^';

    // Underline the range if available and on the same line.
    if (D.Range.Begin.isValid() && D.Range.End.isValid() &&
        D.Range.End.File == D.Range.Begin.File) {
      auto EndLC = SM.getLineColumn(D.Range.End);
      if (EndLC.Line == LC.Line && D.Range.End.Offset > D.Loc.Offset) {
        uint32_t Len = D.Range.End.Offset - D.Loc.Offset;
        for (uint32_t i = 1; i < Len; ++i)
          std::cerr << '~';
      }
    }
    std::cerr << '\n';
  }
}

void DiagnosticEngine::printAll() const {
  for (const auto &D : Diags)
    printOne(D);
}

} // namespace matlab
