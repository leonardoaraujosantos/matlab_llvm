#include "matlab/Flowchart/Loader.h"

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"

#include <algorithm>
#include <cctype>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace matlab::flowchart {

namespace {

//===----------------------------------------------------------------------===//
// Small recursive-descent JSON reader.
//
// Hand-rolled to avoid pulling a third-party header into the build (the
// rest of the project follows the same "no JSON dep" rule). Recognizes
// the JSON subset that .mflow uses: objects, arrays, strings (with
// standard \-escapes), numbers (kept as raw text), booleans, null.
// Tracks a byte offset for every value so diagnostics can point at the
// exact field.
//===----------------------------------------------------------------------===//

enum class JKind {
  Null, Bool, Number, String, Array, Object
};

struct JValue {
  JKind Kind = JKind::Null;
  uint32_t Offset = 0;                       // byte offset of first char
  // The variant payload — flat fields keep this header-free.
  bool BoolVal = false;
  std::string StrVal;                        // for String / Number (raw text)
  std::vector<JValue> ArrVal;
  std::vector<std::pair<std::string, JValue>> ObjVal;

  const JValue *find(std::string_view K) const {
    for (auto &P : ObjVal)
      if (P.first == K) return &P.second;
    return nullptr;
  }
};

class JsonReader {
public:
  JsonReader(std::string_view Src, FileID File, DiagnosticEngine &Diag)
      : Src_(Src), File_(File), Diag_(Diag) {}

  std::optional<JValue> parse() {
    skipWs();
    JValue Root = parseValue();
    if (Failed_) return std::nullopt;
    skipWs();
    if (Pos_ != Src_.size()) {
      err(Pos_, "trailing content after top-level JSON value");
      return std::nullopt;
    }
    return Root;
  }

private:
  std::string_view Src_;
  FileID File_;
  DiagnosticEngine &Diag_;
  size_t Pos_ = 0;
  bool Failed_ = false;

  SourceLocation loc(size_t P) const {
    SourceLocation L;
    L.File = File_;
    L.Offset = static_cast<uint32_t>(P);
    return L;
  }

  void err(size_t P, std::string Msg) {
    if (Failed_) return;
    Failed_ = true;
    Diag_.error(loc(P), std::move(Msg));
  }

  void skipWs() {
    while (Pos_ < Src_.size()) {
      char C = Src_[Pos_];
      if (C == ' ' || C == '\t' || C == '\n' || C == '\r') ++Pos_;
      else break;
    }
  }

  bool consume(char C) {
    skipWs();
    if (Pos_ < Src_.size() && Src_[Pos_] == C) { ++Pos_; return true; }
    return false;
  }

  JValue parseValue() {
    skipWs();
    if (Failed_ || Pos_ >= Src_.size()) {
      err(Pos_, "unexpected end of input");
      return {};
    }
    JValue V;
    V.Offset = static_cast<uint32_t>(Pos_);
    char C = Src_[Pos_];
    if (C == '{') return parseObject(V);
    if (C == '[') return parseArray(V);
    if (C == '"') return parseString(V);
    if (C == 't' || C == 'f') return parseBool(V);
    if (C == 'n') return parseNull(V);
    if (C == '-' || (C >= '0' && C <= '9')) return parseNumber(V);
    err(Pos_, std::string("unexpected character '") + C + "'");
    return V;
  }

  JValue parseObject(JValue V) {
    V.Kind = JKind::Object;
    ++Pos_;                  // consume '{'
    skipWs();
    if (consume('}')) return V;
    while (!Failed_) {
      skipWs();
      if (Pos_ >= Src_.size() || Src_[Pos_] != '"') {
        err(Pos_, "expected string key in object");
        return V;
      }
      JValue K;
      K.Offset = static_cast<uint32_t>(Pos_);
      K = parseString(K);
      if (Failed_) return V;
      skipWs();
      if (!consume(':')) {
        err(Pos_, "expected ':' after object key");
        return V;
      }
      JValue Val = parseValue();
      if (Failed_) return V;
      V.ObjVal.emplace_back(std::move(K.StrVal), std::move(Val));
      skipWs();
      if (consume(',')) continue;
      if (consume('}')) return V;
      err(Pos_, "expected ',' or '}' in object");
      return V;
    }
    return V;
  }

  JValue parseArray(JValue V) {
    V.Kind = JKind::Array;
    ++Pos_;                  // consume '['
    skipWs();
    if (consume(']')) return V;
    while (!Failed_) {
      JValue E = parseValue();
      if (Failed_) return V;
      V.ArrVal.push_back(std::move(E));
      skipWs();
      if (consume(',')) continue;
      if (consume(']')) return V;
      err(Pos_, "expected ',' or ']' in array");
      return V;
    }
    return V;
  }

  JValue parseString(JValue V) {
    V.Kind = JKind::String;
    if (Src_[Pos_] != '"') {
      err(Pos_, "expected string");
      return V;
    }
    ++Pos_;                  // opening quote
    std::string Out;
    while (Pos_ < Src_.size()) {
      char C = Src_[Pos_++];
      if (C == '"') { V.StrVal = std::move(Out); return V; }
      if (C == '\\') {
        if (Pos_ >= Src_.size()) {
          err(Pos_, "unterminated escape sequence");
          return V;
        }
        char E = Src_[Pos_++];
        switch (E) {
        case '"':  Out += '"';  break;
        case '\\': Out += '\\'; break;
        case '/':  Out += '/';  break;
        case 'b':  Out += '\b'; break;
        case 'f':  Out += '\f'; break;
        case 'n':  Out += '\n'; break;
        case 'r':  Out += '\r'; break;
        case 't':  Out += '\t'; break;
        case 'u': {
          if (Pos_ + 4 > Src_.size()) {
            err(Pos_, "truncated \\u escape");
            return V;
          }
          unsigned CP = 0;
          for (int I = 0; I < 4; ++I) {
            char H = Src_[Pos_++];
            CP <<= 4;
            if (H >= '0' && H <= '9') CP |= unsigned(H - '0');
            else if (H >= 'a' && H <= 'f') CP |= unsigned(H - 'a' + 10);
            else if (H >= 'A' && H <= 'F') CP |= unsigned(H - 'A' + 10);
            else { err(Pos_, "invalid hex in \\u escape"); return V; }
          }
          // Encode codepoint as UTF-8. Surrogate pairs aren't needed for
          // the .mflow content we expect (ASCII identifiers + light
          // punctuation); fall back to '?' if we somehow hit one.
          if (CP < 0x80) Out += static_cast<char>(CP);
          else if (CP < 0x800) {
            Out += static_cast<char>(0xC0 | (CP >> 6));
            Out += static_cast<char>(0x80 | (CP & 0x3F));
          } else if (CP < 0xD800 || CP >= 0xE000) {
            Out += static_cast<char>(0xE0 | (CP >> 12));
            Out += static_cast<char>(0x80 | ((CP >> 6) & 0x3F));
            Out += static_cast<char>(0x80 | (CP & 0x3F));
          } else {
            Out += '?';
          }
          break;
        }
        default:
          err(Pos_ - 1, std::string("unknown escape '\\") + E + "'");
          return V;
        }
      } else {
        Out += C;
      }
    }
    err(V.Offset, "unterminated string literal");
    return V;
  }

  JValue parseNumber(JValue V) {
    V.Kind = JKind::Number;
    size_t Start = Pos_;
    if (Src_[Pos_] == '-') ++Pos_;
    while (Pos_ < Src_.size() && std::isdigit(static_cast<unsigned char>(Src_[Pos_]))) ++Pos_;
    if (Pos_ < Src_.size() && Src_[Pos_] == '.') {
      ++Pos_;
      while (Pos_ < Src_.size() && std::isdigit(static_cast<unsigned char>(Src_[Pos_]))) ++Pos_;
    }
    if (Pos_ < Src_.size() && (Src_[Pos_] == 'e' || Src_[Pos_] == 'E')) {
      ++Pos_;
      if (Pos_ < Src_.size() && (Src_[Pos_] == '+' || Src_[Pos_] == '-')) ++Pos_;
      while (Pos_ < Src_.size() && std::isdigit(static_cast<unsigned char>(Src_[Pos_]))) ++Pos_;
    }
    V.StrVal.assign(Src_.substr(Start, Pos_ - Start));
    return V;
  }

  JValue parseBool(JValue V) {
    V.Kind = JKind::Bool;
    if (Src_.substr(Pos_, 4) == "true")   { Pos_ += 4; V.BoolVal = true;  return V; }
    if (Src_.substr(Pos_, 5) == "false")  { Pos_ += 5; V.BoolVal = false; return V; }
    err(Pos_, "expected 'true' or 'false'");
    return V;
  }

  JValue parseNull(JValue V) {
    V.Kind = JKind::Null;
    if (Src_.substr(Pos_, 4) == "null")   { Pos_ += 4; return V; }
    err(Pos_, "expected 'null'");
    return V;
  }
};

//===----------------------------------------------------------------------===//
// Small typed accessors over JValue.
//===----------------------------------------------------------------------===//

const std::string *asString(const JValue *V) {
  return (V && V->Kind == JKind::String) ? &V->StrVal : nullptr;
}

const std::vector<JValue> *asArray(const JValue *V) {
  return (V && V->Kind == JKind::Array) ? &V->ArrVal : nullptr;
}

const std::vector<std::pair<std::string, JValue>> *asObject(const JValue *V) {
  return (V && V->Kind == JKind::Object) ? &V->ObjVal : nullptr;
}

std::optional<bool> asBool(const JValue *V) {
  if (V && V->Kind == JKind::Bool) return V->BoolVal;
  return std::nullopt;
}

SourceLocation locOf(const JValue &V, FileID File) {
  SourceLocation L;
  L.File = File;
  L.Offset = V.Offset;
  return L;
}

//===----------------------------------------------------------------------===//
// JSON → FlowDoc translation.
//===----------------------------------------------------------------------===//

class Builder {
public:
  Builder(FileID File, DiagnosticEngine &Diag) : File_(File), Diag_(Diag) {}

  std::optional<FlowDoc> build(const JValue &Root) {
    if (Root.Kind != JKind::Object) {
      Diag_.error(locOf(Root, File_), "top-level value must be an object");
      return std::nullopt;
    }
    FlowDoc Doc;
    Doc.File = File_;

    if (auto *S = asString(Root.find("schema"))) Doc.Schema = *S;
    if (auto *S = asString(Root.find("version"))) Doc.Version = *S;
    if (auto *S = asString(Root.find("entry"))) Doc.Entry = *S;
    if (auto *Settings = asObject(Root.find("settings"))) {
      for (auto &P : *Settings) {
        if (P.first == "columnMajor") {
          if (auto B = asBool(&P.second)) Doc.Settings.ColumnMajor = *B;
        } else if (P.first == "defaultNumericType") {
          if (auto *S = asString(&P.second)) Doc.Settings.DefaultNumericType = *S;
        } else if (P.first == "sourceLanguage") {
          if (auto *S = asString(&P.second)) Doc.Settings.SourceLanguage = *S;
        }
      }
    }

    if (Doc.Schema != "matforge.flowchart") {
      Diag_.error(locOf(Root, File_),
                  "expected schema \"matforge.flowchart\", got \"" +
                      Doc.Schema + "\"");
      return std::nullopt;
    }

    auto *Flows = asArray(Root.find("flows"));
    if (!Flows) {
      Diag_.error(locOf(Root, File_), "missing or non-array \"flows\"");
      return std::nullopt;
    }

    std::set<std::string> SeenFlowIds;
    for (auto &FJ : *Flows) {
      if (auto F = buildFlow(FJ)) {
        if (!SeenFlowIds.insert(F->Id).second) {
          Diag_.error(F->Loc, "duplicate flow id \"" + F->Id + "\"");
          return std::nullopt;
        }
        Doc.Flows.push_back(std::move(*F));
      } else {
        return std::nullopt;
      }
    }

    if (Doc.Flows.empty()) {
      Diag_.error(locOf(Root, File_), "\"flows\" is empty");
      return std::nullopt;
    }

    if (!Doc.Entry.empty()) {
      bool FoundEntry = false;
      for (auto &F : Doc.Flows)
        if (F.Name == Doc.Entry) { FoundEntry = true; break; }
      if (!FoundEntry) {
        Diag_.error(locOf(Root, File_),
                    "entry \"" + Doc.Entry + "\" not found in flows");
        return std::nullopt;
      }
    }

    // Validate edges, ports, start/end counts. Fatal errors short-circuit.
    for (auto &F : Doc.Flows)
      if (!validateFlow(F)) return std::nullopt;

    // Reachability is best-effort (warnings only); never fails the load.
    for (auto &F : Doc.Flows)
      reportUnreachable(F);

    return Doc;
  }

private:
  FileID File_;
  DiagnosticEngine &Diag_;

  std::optional<Flow> buildFlow(const JValue &FJ) {
    if (FJ.Kind != JKind::Object) {
      Diag_.error(locOf(FJ, File_), "flow entry must be an object");
      return std::nullopt;
    }
    Flow F;
    F.Loc = locOf(FJ, File_);
    if (auto *S = asString(FJ.find("id")))   F.Id = *S;
    if (auto *S = asString(FJ.find("kind"))) F.Kind = *S;
    if (auto *S = asString(FJ.find("name"))) F.Name = *S;

    if (F.Id.empty()) {
      Diag_.error(F.Loc, "flow missing \"id\"");
      return std::nullopt;
    }
    if (F.Name.empty()) {
      Diag_.error(F.Loc, "flow \"" + F.Id + "\" missing \"name\"");
      return std::nullopt;
    }
    if (F.Kind.empty()) F.Kind = "program";
    if (F.Kind != "program" && F.Kind != "function") {
      Diag_.error(F.Loc,
                  "flow \"" + F.Id + "\" has unknown kind \"" + F.Kind + "\"");
      return std::nullopt;
    }

    if (auto *Sig = asObject(FJ.find("signature"))) {
      auto loadList = [&](std::string_view Key, std::vector<std::string> &Out) {
        for (auto &P : *Sig) {
          if (P.first == Key) {
            if (auto *Arr = asArray(&P.second)) {
              for (auto &E : *Arr)
                if (auto *S = asString(&E)) Out.push_back(*S);
            }
          }
        }
      };
      loadList("inputs", F.Sig.Inputs);
      loadList("outputs", F.Sig.Outputs);
    }

    auto *Nodes = asArray(FJ.find("nodes"));
    if (!Nodes) {
      Diag_.error(F.Loc, "flow \"" + F.Id + "\" missing \"nodes\" array");
      return std::nullopt;
    }
    for (auto &NJ : *Nodes) {
      if (auto N = buildNode(NJ)) F.Nodes.push_back(std::move(*N));
      else return std::nullopt;
    }

    auto *Edges = asArray(FJ.find("edges"));
    if (!Edges) {
      Diag_.error(F.Loc, "flow \"" + F.Id + "\" missing \"edges\" array");
      return std::nullopt;
    }
    for (auto &EJ : *Edges) {
      if (auto E = buildEdge(EJ)) F.Edges.push_back(std::move(*E));
      else return std::nullopt;
    }
    return F;
  }

  std::optional<Node> buildNode(const JValue &NJ) {
    if (NJ.Kind != JKind::Object) {
      Diag_.error(locOf(NJ, File_), "node entry must be an object");
      return std::nullopt;
    }
    Node N;
    N.Loc = locOf(NJ, File_);
    if (auto *S = asString(NJ.find("id"))) N.Id = *S;
    if (auto *S = asString(NJ.find("kind"))) N.Kind = *S;
    if (auto *S = asString(NJ.find("label"))) N.Label = *S;
    if (N.Id.empty()) {
      Diag_.error(N.Loc, "node missing \"id\"");
      return std::nullopt;
    }
    if (N.Kind.empty()) {
      Diag_.error(N.Loc, "node \"" + N.Id + "\" missing \"kind\"");
      return std::nullopt;
    }

    if (auto *Data = asObject(NJ.find("data"))) {
      for (auto &P : *Data) {
        if (auto *S = asString(&P.second)) {
          N.Data[P.first] = *S;
          N.DataLocs[P.first] = locOf(P.second, File_);
        } else if (P.second.Kind == JKind::Number) {
          N.Data[P.first] = P.second.StrVal;
          N.DataLocs[P.first] = locOf(P.second, File_);
        } else if (P.second.Kind == JKind::Bool) {
          N.Data[P.first] = P.second.BoolVal ? "true" : "false";
          N.DataLocs[P.first] = locOf(P.second, File_);
        } else if (auto *Arr = asArray(&P.second)) {
          // Arrays of strings: record them under DataArrays. Used
          // by `custom` blocks for `inputs` / `outputs` lists.
          // Non-string elements are coerced to text so a future
          // numeric-typed array (e.g. shape dims) round-trips
          // without losing the user's intent.
          std::vector<std::string> Elems;
          Elems.reserve(Arr->size());
          for (auto &E : *Arr) {
            if (auto *S = asString(&E)) Elems.push_back(*S);
            else if (E.Kind == JKind::Number) Elems.push_back(E.StrVal);
            else if (E.Kind == JKind::Bool)
              Elems.push_back(E.BoolVal ? "true" : "false");
          }
          N.DataArrays[P.first] = std::move(Elems);
          N.DataLocs[P.first] = locOf(P.second, File_);
        }
        // Nested objects in data are still reserved for future
        // additions; skipped silently so forward-compat files load.
      }
    }

    if (auto *Ports = asObject(NJ.find("ports"))) {
      for (auto &P : *Ports) {
        if (P.first == "in" || P.first == "out") {
          auto *Arr = asArray(&P.second);
          if (!Arr) continue;
          auto &Dst = (P.first == "in") ? N.InPorts : N.OutPorts;
          for (auto &PJ : *Arr) {
            if (PJ.Kind != JKind::Object) continue;
            Port Pt;
            Pt.Loc = locOf(PJ, File_);
            if (auto *S = asString(PJ.find("id"))) Pt.Id = *S;
            if (Pt.Id.empty()) {
              Diag_.error(Pt.Loc, "port missing \"id\" on node \"" +
                                      N.Id + "\"");
              return std::nullopt;
            }
            Dst.push_back(std::move(Pt));
          }
        }
      }
    }

    // Phase 8d: capture `ui.position.{x, y}` so `-emit-mflow
    // --preserve-layout` can restore IDE-set positions on re-emit.
    // Forgiving: numeric values are stored as the parser's raw text
    // (strtol-on-demand) so trailing decimals or large coordinates
    // don't error out.
    if (auto *Ui = asObject(NJ.find("ui"))) {
      if (auto *Pos = asObject(JValue{}.ObjVal.empty() ? nullptr : nullptr)) {
        (void)Pos;
      }
      for (auto &P : *Ui) {
        if (P.first != "position") continue;
        auto *PosObj = asObject(&P.second);
        if (!PosObj) continue;
        auto readCoord = [](const JValue *V, int &Out) {
          if (!V) return false;
          if (V->Kind == JKind::Number) {
            try { Out = std::stoi(V->StrVal); return true; }
            catch (...) { return false; }
          }
          if (V->Kind == JKind::String) {
            try { Out = std::stoi(V->StrVal); return true; }
            catch (...) { return false; }
          }
          return false;
        };
        bool GotX = false, GotY = false;
        for (auto &PP : *PosObj) {
          if (PP.first == "x") GotX = readCoord(&PP.second, N.UiX);
          else if (PP.first == "y") GotY = readCoord(&PP.second, N.UiY);
        }
        if (GotX || GotY) N.HasUiPosition = true;
      }
    }
    return N;
  }

  std::optional<Edge> buildEdge(const JValue &EJ) {
    if (EJ.Kind != JKind::Object) {
      Diag_.error(locOf(EJ, File_), "edge entry must be an object");
      return std::nullopt;
    }
    Edge E;
    E.Loc = locOf(EJ, File_);
    if (auto *S = asString(EJ.find("id"))) E.Id = *S;
    if (auto *S = asString(EJ.find("kind"))) E.Kind = *S;
    if (E.Kind.empty()) E.Kind = "control";

    auto loadEndpoint = [&](const JValue *EP, Endpoint &Out,
                            std::string_view Side) -> bool {
      if (!EP || EP->Kind != JKind::Object) {
        Diag_.error(E.Loc, std::string("edge missing \"") + std::string(Side) +
                               "\" endpoint object");
        return false;
      }
      Out.Loc = locOf(*EP, File_);
      if (auto *S = asString(EP->find("node"))) Out.Node = *S;
      if (auto *S = asString(EP->find("port"))) Out.Port = *S;
      if (Out.Node.empty() || Out.Port.empty()) {
        Diag_.error(Out.Loc,
                    "edge endpoint must have non-empty \"node\" and \"port\"");
        return false;
      }
      return true;
    };
    if (!loadEndpoint(EJ.find("from"), E.From, "from")) return std::nullopt;
    if (!loadEndpoint(EJ.find("to"), E.To, "to")) return std::nullopt;
    return E;
  }

  bool validateFlow(const Flow &F) {
    std::unordered_map<std::string, const Node *> NodeById;
    for (auto &N : F.Nodes) {
      auto [It, Inserted] = NodeById.emplace(N.Id, &N);
      if (!Inserted) {
        Diag_.error(N.Loc,
                    "duplicate node id \"" + N.Id + "\" in flow \"" + F.Id +
                        "\"");
        return false;
      }
    }

    auto portExists = [&](const Node &N, std::string_view Side,
                          const std::string &PortId) {
      const auto &List = (Side == "in") ? N.InPorts : N.OutPorts;
      for (auto &P : List)
        if (P.Id == PortId) return true;
      return false;
    };

    std::unordered_set<std::string> SeenEdgeIds;
    for (auto &E : F.Edges) {
      if (!E.Id.empty() && !SeenEdgeIds.insert(E.Id).second) {
        Diag_.error(E.Loc, "duplicate edge id \"" + E.Id + "\"");
        return false;
      }
      auto Fr = NodeById.find(E.From.Node);
      if (Fr == NodeById.end()) {
        Diag_.error(E.From.Loc,
                    "edge \"from\" references unknown node \"" +
                        E.From.Node + "\"");
        return false;
      }
      auto To = NodeById.find(E.To.Node);
      if (To == NodeById.end()) {
        Diag_.error(E.To.Loc,
                    "edge \"to\" references unknown node \"" + E.To.Node +
                        "\"");
        return false;
      }
      if (!portExists(*Fr->second, "out", E.From.Port)) {
        Diag_.error(E.From.Loc,
                    "edge \"from\" port \"" + E.From.Port +
                        "\" not declared on node \"" + E.From.Node + "\"");
        return false;
      }
      if (!portExists(*To->second, "in", E.To.Port)) {
        Diag_.error(E.To.Loc,
                    "edge \"to\" port \"" + E.To.Port +
                        "\" not declared on node \"" + E.To.Node + "\"");
        return false;
      }
    }

    if (F.Kind == "program") {
      int StartCount = 0, EndCount = 0;
      for (auto &N : F.Nodes) {
        if (N.Kind == "start") ++StartCount;
        if (N.Kind == "end")   ++EndCount;
      }
      if (StartCount != 1) {
        Diag_.error(F.Loc, "program flow \"" + F.Name + "\" must contain " +
                               "exactly one start node, found " +
                               std::to_string(StartCount));
        return false;
      }
      if (EndCount < 1) {
        Diag_.error(F.Loc, "program flow \"" + F.Name +
                               "\" must contain at least one end node");
        return false;
      }
    }
    return true;
  }

  void reportUnreachable(const Flow &F) {
    // Best-effort BFS from the start node. Disconnected palette nodes
    // (the IDE leaves them on the canvas without wiring them) emit a
    // single warning per node — they won't block compilation.
    const Node *Start = nullptr;
    for (auto &N : F.Nodes)
      if (N.Kind == "start") { Start = &N; break; }
    if (!Start) return;

    std::unordered_map<std::string, std::vector<const Edge *>> OutEdges;
    for (auto &E : F.Edges) OutEdges[E.From.Node].push_back(&E);

    std::unordered_set<std::string> Reachable;
    std::vector<const Node *> Stack{Start};
    while (!Stack.empty()) {
      const Node *N = Stack.back(); Stack.pop_back();
      if (!Reachable.insert(N->Id).second) continue;
      for (auto *E : OutEdges[N->Id]) {
        for (auto &M : F.Nodes) {
          if (M.Id == E->To.Node) { Stack.push_back(&M); break; }
        }
      }
    }
    for (auto &N : F.Nodes) {
      if (Reachable.count(N.Id)) continue;
      // `end` is reached *into*, not *out of*, so a flow with a single
      // unconnected end during early IDE editing isn't worth flagging.
      if (N.Kind == "end") continue;
      Diag_.warning(N.Loc, "node \"" + N.Id + "\" (kind \"" + N.Kind +
                               "\") is not reachable from start; ignored");
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

const Flow *FlowDoc::entryFlow() const {
  return findFlow(Entry);
}

const Flow *FlowDoc::findFlow(std::string_view Name) const {
  for (auto &F : Flows)
    if (F.Name == Name) return &F;
  return nullptr;
}

std::optional<FlowDoc> loadMflow(const SourceManager &SM, FileID File,
                                 DiagnosticEngine &Diag) {
  auto Buf = SM.getBuffer(File);
  JsonReader R(Buf, File, Diag);
  auto Root = R.parse();
  if (!Root) return std::nullopt;
  Builder B(File, Diag);
  return B.build(*Root);
}

std::optional<FlowDoc> loadMflowFromPath(SourceManager &SM,
                                         const std::string &Path,
                                         DiagnosticEngine &Diag) {
  FileID F = SM.loadFile(Path);
  if (!F) {
    SourceLocation Invalid;
    Diag.error(Invalid, "could not open .mflow file: " + Path);
    return std::nullopt;
  }
  return loadMflow(SM, F, Diag);
}

void dumpFlowDoc(std::ostream &OS, const FlowDoc &Doc) {
  OS << "FlowDoc schema=" << Doc.Schema << " version=" << Doc.Version
     << " entry=" << Doc.Entry << "\n";
  OS << "  settings.columnMajor=" << (Doc.Settings.ColumnMajor ? "true" : "false")
     << " defaultNumericType=" << Doc.Settings.DefaultNumericType
     << " sourceLanguage=" << Doc.Settings.SourceLanguage << "\n";
  for (auto &F : Doc.Flows) {
    OS << "Flow id=" << F.Id << " kind=" << F.Kind << " name=" << F.Name;
    if (!F.Sig.Inputs.empty() || !F.Sig.Outputs.empty()) {
      OS << " sig=(";
      for (size_t I = 0; I < F.Sig.Inputs.size(); ++I) {
        if (I) OS << ",";
        OS << F.Sig.Inputs[I];
      }
      OS << ")->(";
      for (size_t I = 0; I < F.Sig.Outputs.size(); ++I) {
        if (I) OS << ",";
        OS << F.Sig.Outputs[I];
      }
      OS << ")";
    }
    OS << "\n";
    for (auto &N : F.Nodes) {
      OS << "  Node id=" << N.Id << " kind=" << N.Kind;
      if (!N.Label.empty()) OS << " label=" << N.Label;
      OS << "\n";
      // Print data fields in sorted order so goldens are stable across
      // different std::map implementations / insertion orders.
      std::vector<std::string> Keys;
      Keys.reserve(N.Data.size());
      for (auto &P : N.Data) Keys.push_back(P.first);
      std::sort(Keys.begin(), Keys.end());
      for (auto &K : Keys) OS << "    data." << K << "=" << N.Data.at(K) << "\n";
      std::vector<std::string> ArrKeys;
      ArrKeys.reserve(N.DataArrays.size());
      for (auto &P : N.DataArrays) ArrKeys.push_back(P.first);
      std::sort(ArrKeys.begin(), ArrKeys.end());
      for (auto &K : ArrKeys) {
        OS << "    data." << K << "=[";
        const auto &Vs = N.DataArrays.at(K);
        for (size_t I = 0; I < Vs.size(); ++I) {
          if (I) OS << ",";
          OS << Vs[I];
        }
        OS << "]\n";
      }
      for (auto &P : N.InPorts)  OS << "    in:"  << P.Id << "\n";
      for (auto &P : N.OutPorts) OS << "    out:" << P.Id << "\n";
    }
    for (auto &E : F.Edges) {
      OS << "  Edge id=" << E.Id << " kind=" << E.Kind << " "
         << E.From.Node << ":" << E.From.Port << " -> "
         << E.To.Node << ":" << E.To.Port << "\n";
    }
  }
}

} // namespace matlab::flowchart
