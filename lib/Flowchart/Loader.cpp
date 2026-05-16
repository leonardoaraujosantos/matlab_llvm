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
  // Byte offset of the LAST character of the value (inclusive): the
  // closing `}` / `]` / `"`, the last digit of a number, or the last
  // letter of `true` / `false` / `null`. Used so callers can derive
  // a SourceLocation that points at the end of a JSON value — the
  // breakpoint plumbing relies on this to register every line a
  // .mflow block spans, not just the line of its opening `{`.
  uint32_t EndOffset = 0;
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

  // Stamp V.EndOffset with the position of the last character we just
  // consumed (Pos_ is one past it, so subtract one). Called from each
  // successful return path in the parse* helpers below.
  void setEnd(JValue &V) {
    V.EndOffset = Pos_ > 0 ? static_cast<uint32_t>(Pos_ - 1) : V.Offset;
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
    if (consume('}')) { setEnd(V); return V; }
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
      if (consume('}')) { setEnd(V); return V; }
      err(Pos_, "expected ',' or '}' in object");
      return V;
    }
    return V;
  }

  JValue parseArray(JValue V) {
    V.Kind = JKind::Array;
    ++Pos_;                  // consume '['
    skipWs();
    if (consume(']')) { setEnd(V); return V; }
    while (!Failed_) {
      JValue E = parseValue();
      if (Failed_) return V;
      V.ArrVal.push_back(std::move(E));
      skipWs();
      if (consume(',')) continue;
      if (consume(']')) { setEnd(V); return V; }
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
      if (C == '"') { V.StrVal = std::move(Out); setEnd(V); return V; }
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
    setEnd(V);
    return V;
  }

  JValue parseBool(JValue V) {
    V.Kind = JKind::Bool;
    if (Src_.substr(Pos_, 4) == "true")   { Pos_ += 4; V.BoolVal = true;  setEnd(V); return V; }
    if (Src_.substr(Pos_, 5) == "false")  { Pos_ += 5; V.BoolVal = false; setEnd(V); return V; }
    err(Pos_, "expected 'true' or 'false'");
    return V;
  }

  JValue parseNull(JValue V) {
    V.Kind = JKind::Null;
    if (Src_.substr(Pos_, 4) == "null")   { Pos_ += 4; setEnd(V); return V; }
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

// Numeric JSON value → double. Numbers are kept as raw text by the
// reader, so this is the one place that actually evaluates them.
std::optional<double> asNumber(const JValue *V) {
  if (!V || V->Kind != JKind::Number) return std::nullopt;
  try {
    return std::stod(V->StrVal);
  } catch (...) {
    return std::nullopt;
  }
}

// A "step size" field (`settings.solver.maxStep` / `minStep`): the IDE
// always writes it as a string ("auto" or a numeric literal), but a
// bare JSON number is accepted defensively and kept as its raw text.
std::optional<std::string> asNumberOrString(const JValue *V) {
  if (!V) return std::nullopt;
  if (V->Kind == JKind::String || V->Kind == JKind::Number) return V->StrVal;
  return std::nullopt;
}

// Parse a `solver` JSON object into a SolverConfig. §17.5 #7
// extracts this from the top-level settings.solver and per-flow
// solver paths so the two share one keyset. The returned config
// has every field at its SolverConfig default unless explicitly
// overridden by the JSON.
SolverConfig parseSolverConfig(const std::vector<std::pair<std::string, JValue>>
                                   &Obj) {
  SolverConfig SC;
  for (auto &Q : Obj) {
    if (Q.first == "type") {
      if (auto *S = asString(&Q.second)) SC.Type = *S;
    } else if (Q.first == "algorithm") {
      if (auto *S = asString(&Q.second)) SC.Algorithm = *S;
    } else if (Q.first == "startTime") {
      if (auto N = asNumber(&Q.second)) SC.StartTime = *N;
    } else if (Q.first == "stopTime") {
      if (auto N = asNumber(&Q.second)) SC.StopTime = *N;
    } else if (Q.first == "maxStep") {
      if (auto S = asNumberOrString(&Q.second)) SC.MaxStep = *S;
    } else if (Q.first == "minStep") {
      if (auto S = asNumberOrString(&Q.second)) SC.MinStep = *S;
    } else if (Q.first == "relTol") {
      if (auto N = asNumber(&Q.second)) SC.RelTol = *N;
    } else if (Q.first == "absTol") {
      if (auto N = asNumber(&Q.second)) SC.AbsTol = *N;
    } else if (Q.first == "zeroCrossing") {
      if (auto B = asBool(&Q.second)) SC.ZeroCrossing = *B;
    } else if (Q.first == "algebraicLoopMethod") {
      if (auto *S = asString(&Q.second)) SC.AlgebraicLoopMethod = *S;
    }
  }
  return SC;
}

SourceLocation locOf(const JValue &V, FileID File) {
  SourceLocation L;
  L.File = File;
  L.Offset = V.Offset;
  return L;
}

// Location pointing AT the last character of the JSON value (the
// closing `}` / `]` / `"` for compound types, the last digit/letter
// for scalars). Used to derive `Node::LocEnd` so the Stmt range a
// .mflow block synthesises spans every line of its JSON object — the
// AST walker in matlabc registers each line as breakpoint-eligible.
SourceLocation endLocOf(const JValue &V, FileID File) {
  SourceLocation L;
  L.File = File;
  L.Offset = V.EndOffset != 0 ? V.EndOffset : V.Offset;
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
        } else if (P.first == "kind") {
          // mflowLink dialect selector. An absent `kind` keeps the
          // historical "control_flow" default.
          if (auto *S = asString(&P.second)) Doc.Settings.Kind = *S;
        } else if (P.first == "solver") {
          if (auto *Obj = asObject(&P.second))
            Doc.Settings.Solver = parseSolverConfig(*Obj);
        } else if (P.first == "snapshot") {
          if (auto *Obj = asObject(&P.second)) {
            SnapshotConfig SN;
            for (auto &Q : *Obj) {
              if (Q.first == "enabled") {
                if (auto B = asBool(&Q.second)) SN.Enabled = *B;
              } else if (Q.first == "depth") {
                if (auto N = asNumber(&Q.second))
                  SN.Depth = static_cast<int>(*N);
              } else if (Q.first == "fields") {
                if (auto *S = asString(&Q.second)) SN.Fields = *S;
              }
            }
            Doc.Settings.Snapshot = std::move(SN);
          }
        }
      }
    }

    if (Doc.Settings.Kind != "control_flow" &&
        Doc.Settings.Kind != "signal_flow" &&
        Doc.Settings.Kind != "state_chart") {
      Diag_.error(locOf(Root, File_),
                  "unknown settings.kind \"" + Doc.Settings.Kind +
                      "\"; expected \"control_flow\", \"signal_flow\", or "
                      "\"state_chart\"");
      return std::nullopt;
    }
    SignalFlow_ = Doc.isSignalFlow();
    StateChart_ = Doc.isStateChart();

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

    // mStateflow: per-flow hierarchy + decomposition + default-
    // transition validation. Runs only for state_chart documents;
    // signal-flow and control-flow files take the empty default path.
    if (StateChart_) {
      for (auto &F : Doc.Flows)
        if (!validateStateChart(F)) return std::nullopt;
    }

    // Reachability is best-effort (warnings only); never fails the load.
    for (auto &F : Doc.Flows)
      reportUnreachable(F);

    return Doc;
  }

private:
  FileID File_;
  DiagnosticEngine &Diag_;
  // Set once `settings.kind` is parsed: a signal-flow document is a
  // block diagram, so it has no `start` / `end` control nodes — the
  // program-shape validation in `validateFlow` is skipped for it.
  bool SignalFlow_ = false;
  // Set once `settings.kind` is parsed: a state-chart document has
  // no `start` / `end` either; its top-level entry is encoded as a
  // `junction_default` node or a `params.isInitial=true` substate.
  bool StateChart_ = false;

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
    // Item-3 — `library` flows are templates: they're never the
    // entry, never run on their own, and only participate in a
    // simulation when a `signal_subsystem` with a `data.mask`
    // resolves their id. The loader accepts them as a recognised
    // flow kind; SignalFlowLowering skips them unless mask-cloned.
    if (F.Kind != "program" && F.Kind != "function" &&
        F.Kind != "library") {
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

    // §17.5 #7 — optional per-flow solver. Same JSON shape as the
    // top-level settings.solver.
    if (auto *Obj = asObject(FJ.find("solver")))
      F.Solver = parseSolverConfig(*Obj);

    // mStateflow: per-flow symbol table (data / events / messages).
    // Shape:
    //   "symbols": {
    //     "data":     [{"name": ..., "scope": ..., "type": ..., ...}, ...],
    //     "events":   [{"name": ..., "trigger": ...}, ...],
    //     "messages": [{"name": ..., "type": ...}, ...]
    //   }
    // The loader keeps every field as raw text; MATLAB-type resolution
    // and semantic checks happen in the chart IR builder.
    if (auto *Obj = asObject(FJ.find("symbols"))) {
      auto loadSymbols = [&](std::string_view Key, std::vector<Symbol> &Out) {
        for (auto &P : *Obj) {
          if (P.first != Key) continue;
          auto *Arr = asArray(&P.second);
          if (!Arr) continue;
          for (auto &E : *Arr) {
            if (E.Kind != JKind::Object) continue;
            Symbol S;
            S.Loc = locOf(E, File_);
            for (auto &F2 : E.ObjVal) {
              const std::string *V = asString(&F2.second);
              if (!V) continue;
              if (F2.first == "name") S.Name = *V;
              else if (F2.first == "scope") S.Scope = *V;
              else if (F2.first == "type") S.Type = *V;
              else if (F2.first == "units") S.Units = *V;
              else if (F2.first == "initial") S.Initial = *V;
              else if (F2.first == "trigger") S.Trigger = *V;
            }
            if (S.Name.empty()) {
              Diag_.error(S.Loc,
                          "symbol entry under \"" + std::string(Key) +
                              "\" missing \"name\"");
              continue;
            }
            Out.push_back(std::move(S));
          }
        }
      };
      loadSymbols("data",     F.Symbols.Data);
      loadSymbols("events",   F.Symbols.Events);
      loadSymbols("messages", F.Symbols.Messages);
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
    N.LocEnd = endLocOf(NJ, File_);
    if (auto *S = asString(NJ.find("id"))) N.Id = *S;
    if (auto *S = asString(NJ.find("kind"))) N.Kind = *S;
    if (auto *S = asString(NJ.find("label"))) N.Label = *S;
    // mStateflow: optional id of the containing state. Resolved
    // against sibling node ids in validateStateChart.
    if (const auto *PJ = NJ.find("parent")) {
      if (auto *S = asString(PJ)) {
        N.Parent = *S;
        N.ParentLoc = locOf(*PJ, File_);
      }
    }
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
        // mStateflow: `data.onEventActions` is a nested object of
        // event-name → MATLAB-source pairs. Pull it out before the
        // generic scalar/array branch (which silently skips nested
        // objects). Declaration order preserved.
        if (P.first == "onEventActions") {
          if (auto *Obj = asObject(&P.second)) {
            for (auto &OE : *Obj) {
              if (auto *S = asString(&OE.second)) {
                N.OnEventActions.emplace_back(OE.first, *S);
                N.OnEventActionLocs[OE.first] = locOf(OE.second, File_);
              }
            }
          }
          continue;
        }
        // mflowLink: `data.params` is a nested object of scalar block
        // parameters (per the IDE's SignalFlowParamSpec). The generic
        // scalar/array handling below skips nested objects, so pull
        // params out into Node::Params first. Scalars kept as raw
        // text — the lowering pass (§6) reparses per the catalogue.
        if (P.first == "params") {
          if (auto *PObj = asObject(&P.second)) {
            for (auto &PP : *PObj) {
              if (auto *S = asString(&PP.second))
                N.Params[PP.first] = *S;
              else if (PP.second.Kind == JKind::Number)
                N.Params[PP.first] = PP.second.StrVal;
              else if (PP.second.Kind == JKind::Bool)
                N.Params[PP.first] = PP.second.BoolVal ? "true" : "false";
              else
                continue; // arrays / nested objects not used by params
              N.ParamLocs[PP.first] = locOf(PP.second, File_);
            }
          }
          continue;
        }
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
      for (auto &P : *Ui) {
        if (P.first == "position") {
          auto *PosObj = asObject(&P.second);
          if (!PosObj) continue;
          bool GotX = false, GotY = false;
          for (auto &PP : *PosObj) {
            if (PP.first == "x") GotX = readCoord(&PP.second, N.UiX);
            else if (PP.first == "y") GotY = readCoord(&PP.second, N.UiY);
          }
          if (GotX || GotY) N.HasUiPosition = true;
        } else if (P.first == "size") {
          // mStateflow: compound states carry visible bounds.
          auto *SzObj = asObject(&P.second);
          if (!SzObj) continue;
          bool GotW = false, GotH = false;
          for (auto &PP : *SzObj) {
            if (PP.first == "w") GotW = readCoord(&PP.second, N.UiW);
            else if (PP.first == "h") GotH = readCoord(&PP.second, N.UiH);
          }
          if (GotW || GotH) N.HasUiSize = true;
        }
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

    // mStateflow: transition label (raw text). The four-field
    // decomposition `event[guard]{condAction}/transAction` is parsed
    // lazily by the chart IR builder.
    if (const auto *LV = EJ.find("label")) {
      if (auto *S = asString(LV)) {
        E.Label = *S;
        E.LabelLoc = locOf(*LV, File_);
      }
    }

    // mStateflow: `data.params` on a transition (priority / kind /
    // ...). Mirrors the node-side params bag.
    if (auto *Data = asObject(EJ.find("data"))) {
      for (auto &P : *Data) {
        if (P.first != "params") continue;
        auto *PObj = asObject(&P.second);
        if (!PObj) continue;
        for (auto &PP : *PObj) {
          if (auto *S = asString(&PP.second))
            E.Params[PP.first] = *S;
          else if (PP.second.Kind == JKind::Number)
            E.Params[PP.first] = PP.second.StrVal;
          else if (PP.second.Kind == JKind::Bool)
            E.Params[PP.first] = PP.second.BoolVal ? "true" : "false";
          else
            continue;
          E.ParamLocs[PP.first] = locOf(PP.second, File_);
        }
      }
    }

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

    // A signal-flow / state-chart flow is a diagram, not a statement
    // program: it has no `start` / `end` control nodes. Skip the
    // program-shape check entirely — execution-order validation
    // happens later in SignalFlowLowering (§6) for signal-flow, and
    // in lib/StateChart for state-chart flows.
    if (F.Kind == "program" && !SignalFlow_ && !StateChart_) {
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

  // mStateflow per-flow validator. Runs only for `state_chart` docs.
  //
  //   1. Every `Node.Parent` resolves to a sibling whose kind == "state".
  //   2. No parent cycles (DFS from each node, no node revisits itself).
  //   3. Every OR parent (a `state` whose `params.decomposition` is
  //      "or" or absent) has at most one default-transition source —
  //      counting both `junction_default` children and substates with
  //      `params.isInitial=true`.
  //   4. Every direct substate of an AND parent declares a numeric
  //      `params.executionOrder`.
  //
  // Each failure produces a single diagnostic and short-circuits.
  // Semantic / type checks against `Flow.Symbols` live in the chart
  // IR builder (Tier 4) — this is structural only.
  bool validateStateChart(const Flow &F) {
    std::unordered_map<std::string, const Node *> NodeById;
    for (auto &N : F.Nodes) NodeById[N.Id] = &N;

    auto isState = [](const Node &N) { return N.Kind == "state"; };
    auto decomposition = [&](const Node *Parent) -> std::string {
      // Chart-root is OR by default; an explicit "or"/"and"/"leaf"
      // override lives on the parent state's `params.decomposition`.
      if (!Parent) return "or";
      if (auto *S = Parent->getParam("decomposition")) return *S;
      return "or";
    };

    for (auto &N : F.Nodes) {
      if (N.Parent.empty()) continue;
      auto It = NodeById.find(N.Parent);
      if (It == NodeById.end()) {
        Diag_.error(N.ParentLoc.Offset ? N.ParentLoc : N.Loc,
                    "node \"" + N.Id + "\" references unknown parent \"" +
                        N.Parent + "\" in flow \"" + F.Id + "\"");
        return false;
      }
      if (!isState(*It->second)) {
        Diag_.error(N.ParentLoc.Offset ? N.ParentLoc : N.Loc,
                    "node \"" + N.Id + "\" parent \"" + N.Parent +
                        "\" must be a state node (got kind \"" +
                        It->second->Kind + "\")");
        return false;
      }
    }

    // Parent cycle: walk the parent chain for every node, bounded by
    // node count. A cycle re-visits its head; an out-of-bounds depth
    // signals a malformed chain (defensive — the resolution check
    // above already rejects dangling parents, so this is for cycles).
    for (auto &N : F.Nodes) {
      std::unordered_set<std::string> Seen{N.Id};
      const Node *Cur = &N;
      while (!Cur->Parent.empty()) {
        if (!Seen.insert(Cur->Parent).second) {
          Diag_.error(N.Loc,
                      "parent chain cycle detected at node \"" + N.Id +
                          "\" in flow \"" + F.Id + "\"");
          return false;
        }
        Cur = NodeById.at(Cur->Parent);
      }
    }

    // Default-transition + execution-order checks need a "children of
    // X" view. Bucket once.
    std::unordered_map<std::string, std::vector<const Node *>> ChildrenOf;
    for (auto &N : F.Nodes) ChildrenOf[N.Parent].push_back(&N);

    auto isInitialFlag = [](const Node &N) {
      auto *S = N.getParam("isInitial");
      return S && (*S == "true" || *S == "1");
    };

    auto checkDefaults = [&](const std::string &ParentId,
                             const std::vector<const Node *> &Kids) -> bool {
      int Defaults = 0;
      const Node *FirstDefault = nullptr;
      auto bump = [&](const Node &N) {
        ++Defaults;
        if (!FirstDefault) FirstDefault = &N;
      };
      for (auto *K : Kids) {
        if (K->Kind == "junction_default") bump(*K);
        else if (K->Kind == "state" && isInitialFlag(*K)) bump(*K);
      }
      if (Defaults > 1) {
        Diag_.error(FirstDefault->Loc,
                    "OR region under \"" +
                        (ParentId.empty() ? std::string("<root>") : ParentId) +
                        "\" has " + std::to_string(Defaults) +
                        " default transitions; expected at most one");
        return false;
      }
      return true;
    };

    // OR / AND checks. The chart-root is always OR.
    for (auto &Bucket : ChildrenOf) {
      const Node *Parent =
          Bucket.first.empty() ? nullptr : NodeById.at(Bucket.first);
      std::string Decomp = decomposition(Parent);
      if (Decomp == "or") {
        if (!checkDefaults(Bucket.first, Bucket.second)) return false;
      } else if (Decomp == "and") {
        for (auto *K : Bucket.second) {
          if (K->Kind != "state") continue;
          auto *OrderS = K->getParam("executionOrder");
          if (!OrderS) {
            Diag_.error(K->Loc,
                        "AND-region child state \"" + K->Id +
                            "\" missing data.params.executionOrder");
            return false;
          }
          try { (void)std::stoi(*OrderS); }
          catch (...) {
            Diag_.error(K->Loc,
                        "AND-region child state \"" + K->Id +
                            "\" has non-numeric data.params.executionOrder=\"" +
                            *OrderS + "\"");
            return false;
          }
        }
      } else if (Decomp != "leaf") {
        Diag_.error(Parent ? Parent->Loc : F.Loc,
                    "unknown decomposition \"" + Decomp +
                        "\" (expected \"or\", \"and\", or \"leaf\")");
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
  // mflowLink lines are emitted only when present, so control-flow
  // `.mflow` dumps stay byte-for-byte identical to before.
  if (Doc.Settings.Kind != "control_flow")
    OS << "  settings.kind=" << Doc.Settings.Kind << "\n";
  if (Doc.Settings.Solver) {
    const auto &S = *Doc.Settings.Solver;
    OS << "  settings.solver type=" << S.Type << " algorithm=" << S.Algorithm
       << " startTime=" << S.StartTime << " stopTime=" << S.StopTime
       << " maxStep=" << S.MaxStep << " minStep=" << S.MinStep
       << " relTol=" << S.RelTol << " absTol=" << S.AbsTol
       << " zeroCrossing=" << (S.ZeroCrossing ? "true" : "false")
       << " algebraicLoopMethod=" << S.AlgebraicLoopMethod << "\n";
  }
  if (Doc.Settings.Snapshot) {
    const auto &S = *Doc.Settings.Snapshot;
    OS << "  settings.snapshot enabled=" << (S.Enabled ? "true" : "false")
       << " depth=" << S.Depth << " fields=" << S.Fields << "\n";
  }
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
    // mStateflow: symbol table. Printed in declaration order (the
    // loader preserves it); inner fields are emitted only when
    // non-empty so chart fixtures dump tightly.
    auto dumpSymbol = [&](const char *Bucket, const Symbol &S) {
      OS << "  symbol." << Bucket << " name=" << S.Name;
      if (!S.Scope.empty())   OS << " scope=" << S.Scope;
      if (!S.Type.empty())    OS << " type=" << S.Type;
      if (!S.Units.empty())   OS << " units=" << S.Units;
      if (!S.Initial.empty()) OS << " initial=" << S.Initial;
      if (!S.Trigger.empty()) OS << " trigger=" << S.Trigger;
      OS << "\n";
    };
    for (auto &S : F.Symbols.Data)     dumpSymbol("data",    S);
    for (auto &S : F.Symbols.Events)   dumpSymbol("event",   S);
    for (auto &S : F.Symbols.Messages) dumpSymbol("message", S);
    for (auto &N : F.Nodes) {
      OS << "  Node id=" << N.Id << " kind=" << N.Kind;
      if (!N.Label.empty()) OS << " label=" << N.Label;
      if (!N.Parent.empty()) OS << " parent=" << N.Parent;
      OS << "\n";
      if (N.HasUiSize)
        OS << "    ui.size=" << N.UiW << "x" << N.UiH << "\n";
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
      // mflowLink block parameters (`data.params`), sorted for stable
      // goldens. Only present on signal-flow nodes.
      std::vector<std::string> ParamKeys;
      ParamKeys.reserve(N.Params.size());
      for (auto &P : N.Params) ParamKeys.push_back(P.first);
      std::sort(ParamKeys.begin(), ParamKeys.end());
      for (auto &K : ParamKeys)
        OS << "    param." << K << "=" << N.Params.at(K) << "\n";
      // mStateflow on-event handlers — preserved in declaration order
      // so chart codegen can iterate deterministically.
      for (auto &P : N.OnEventActions)
        OS << "    onEvent." << P.first << "=" << P.second << "\n";
      for (auto &P : N.InPorts)  OS << "    in:"  << P.Id << "\n";
      for (auto &P : N.OutPorts) OS << "    out:" << P.Id << "\n";
    }
    for (auto &E : F.Edges) {
      OS << "  Edge id=" << E.Id << " kind=" << E.Kind << " "
         << E.From.Node << ":" << E.From.Port << " -> "
         << E.To.Node << ":" << E.To.Port;
      if (!E.Label.empty()) OS << " label=" << E.Label;
      OS << "\n";
      // mStateflow: transition params (priority / kind / ...). Sorted
      // for stable goldens; mirrors Node-side `param.X=...` lines.
      std::vector<std::string> EParamKeys;
      EParamKeys.reserve(E.Params.size());
      for (auto &P : E.Params) EParamKeys.push_back(P.first);
      std::sort(EParamKeys.begin(), EParamKeys.end());
      for (auto &K : EParamKeys)
        OS << "    param." << K << "=" << E.Params.at(K) << "\n";
    }
  }
}

} // namespace matlab::flowchart
