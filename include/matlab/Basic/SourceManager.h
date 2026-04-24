#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace matlab {

using FileID = uint32_t;

struct SourceLocation {
  FileID File = 0;
  uint32_t Offset = 0;

  bool isValid() const { return File != 0; }
  bool operator==(const SourceLocation &O) const {
    return File == O.File && Offset == O.Offset;
  }
};

struct SourceRange {
  SourceLocation Begin;
  SourceLocation End;
};

struct LineColumn {
  uint32_t Line = 0;   // 1-based
  uint32_t Column = 0; // 1-based
};

class SourceManager {
public:
  // Load a file from disk. Returns 0 on failure.
  FileID loadFile(const std::string &Path);

  // Register an in-memory buffer (e.g. for tests).
  FileID addBuffer(std::string Name, std::string Contents);

  std::string_view getBuffer(FileID File) const;
  const std::string &getName(FileID File) const;

  LineColumn getLineColumn(SourceLocation Loc) const;
  std::string_view getLineText(FileID File, uint32_t Line) const;

  // Look up a registered file by the name it was loaded with. Returns 0
  // when no file matches. O(N) over the entry count — fine for the tiny
  // file counts we deal with (one or two .m files per run).
  FileID findFileByName(std::string_view Name) const;

private:
  struct Entry {
    std::string Name;
    std::string Contents;
    std::vector<uint32_t> LineStarts; // offsets of the first char of each line
  };
  std::vector<std::unique_ptr<Entry>> Entries; // index = FileID - 1

  void computeLineStarts(Entry &E);
  const Entry *get(FileID File) const;
};

} // namespace matlab
