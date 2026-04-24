#include "matlab/Basic/SourceManager.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <sstream>

namespace matlab {

void SourceManager::computeLineStarts(Entry &E) {
  E.LineStarts.clear();
  E.LineStarts.push_back(0);
  for (uint32_t i = 0; i < E.Contents.size(); ++i) {
    if (E.Contents[i] == '\n')
      E.LineStarts.push_back(i + 1);
  }
}

FileID SourceManager::loadFile(const std::string &Path) {
  std::ifstream In(Path, std::ios::binary);
  if (!In)
    return 0;
  std::ostringstream Buf;
  Buf << In.rdbuf();
  return addBuffer(Path, Buf.str());
}

FileID SourceManager::addBuffer(std::string Name, std::string Contents) {
  auto E = std::make_unique<Entry>();
  E->Name = std::move(Name);
  E->Contents = std::move(Contents);
  computeLineStarts(*E);
  Entries.push_back(std::move(E));
  return static_cast<FileID>(Entries.size()); // 1-based
}

const SourceManager::Entry *SourceManager::get(FileID File) const {
  if (File == 0 || File > Entries.size())
    return nullptr;
  return Entries[File - 1].get();
}

std::string_view SourceManager::getBuffer(FileID File) const {
  const Entry *E = get(File);
  assert(E && "invalid FileID");
  return E->Contents;
}

const std::string &SourceManager::getName(FileID File) const {
  const Entry *E = get(File);
  assert(E && "invalid FileID");
  return E->Name;
}

LineColumn SourceManager::getLineColumn(SourceLocation Loc) const {
  const Entry *E = get(Loc.File);
  if (!E)
    return {};
  const auto &Ls = E->LineStarts;
  auto It = std::upper_bound(Ls.begin(), Ls.end(), Loc.Offset);
  uint32_t LineIdx = static_cast<uint32_t>((It - Ls.begin()) - 1);
  LineColumn R;
  R.Line = LineIdx + 1;
  R.Column = Loc.Offset - Ls[LineIdx] + 1;
  return R;
}

FileID SourceManager::findFileByName(std::string_view Name) const {
  for (uint32_t i = 0; i < Entries.size(); ++i) {
    if (Entries[i]->Name == Name)
      return static_cast<FileID>(i + 1);
  }
  return 0;
}

std::string_view SourceManager::getLineText(FileID File, uint32_t Line) const {
  const Entry *E = get(File);
  if (!E || Line == 0 || Line > E->LineStarts.size())
    return {};
  uint32_t Start = E->LineStarts[Line - 1];
  uint32_t End = Line < E->LineStarts.size() ? E->LineStarts[Line] - 1
                                             : static_cast<uint32_t>(E->Contents.size());
  // strip trailing \r
  while (End > Start && E->Contents[End - 1] == '\r')
    --End;
  return std::string_view(E->Contents).substr(Start, End - Start);
}

} // namespace matlab
