// runtime_onnx.cpp — Deep Learning Toolbox H5: minimal ONNX inference-graph
// importer + builder.
//
// Scope (matches the shipped matrix-lane DL surface):
//   - Hand-rolled Protocol-Buffers wire-format reader + writer, no external
//     dep (no libprotobuf / no abseil / no schema codegen).  Closes the
//     PyTorch / TensorFlow / ONNX-native user stories — `torch.onnx.export`
//     and `tf2onnx` both produce the same wire format we read here.
//   - ONNX schema decode / encode for ModelProto / GraphProto / NodeProto /
//     TensorProto / AttributeProto / ValueInfoProto.
//   - Op-execution dispatcher mapping ONNX op_type -> our shipped runtime
//     (matlab_matmul_mm, matlab_conv2d_batch_full, relu / sigmoid / tanh /
//     softmax / batchnorm / layernorm / pool / reshape / cat / ...).
//   - MATLAB-level entry points:
//        - reader/runner:  onnxRead, onnxRun
//        - programmatic builder for tests:  onnxNewModel /
//          onnxAddInit / onnxBeginNode / onnxNodeInput / onnxNodeOutput /
//          onnxNodeAttrInt / onnxNodeAttrFloat / onnxNodeAttrInts /
//          onnxEndNode / onnxSetInput / onnxSetOutput / onnxSave.
//
// Tensors flow through the runner as `matlab_mat *` (2-D matrix lane).  4-D
// SSCB tensors are NOT supported by the executor — same carve as the rest of
// the DL toolbox — but the *reader* accepts arbitrary rank and stores the
// raw values; per-op handlers reject ranks they can't handle.
//
// No external dependency: the only file I/O is std::ifstream / std::ofstream
// reading the .onnx bytes directly.  Single global builder + single global
// loaded model, mirroring the ident lsqnonlin / imageDatastore precedent.

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

// Shipped runtime entries the executor reuses.
extern "C" matlab_mat *matlab_matmul_mm(matlab_mat *, matlab_mat *);
extern "C" matlab_mat *matlab_conv2(matlab_mat *, matlab_mat *);
extern "C" void       *matlab_conv2d_batch_full(void *, void *, void *,
                                                double, double, double, double);
extern "C" matlab_mat *matlab_rand(double, double);

// File-scope helpers (mirrors the dlnet runtime_dlnet.cpp).
namespace {

struct onnx_string_s { char *data; int64_t len; };

static std::string read_str_arg(const void *s) {
    if (!s) return std::string();
    const onnx_string_s *p = reinterpret_cast<const onnx_string_s *>(s);
    if (!p->data || p->len <= 0) return std::string();
    return std::string(p->data, p->data + p->len);
}

// matlab_mat helpers (rows/cols/numel/clone).  Local to avoid pulling
// dlnet:: namespace from runtime_dlnet.cpp.
static int64_t mat_nelem(const matlab_mat *m) {
    return m ? m->rows * m->cols : 0;
}
static matlab_mat *mat_make(int64_t r, int64_t c) { return mat_alloc(r, c); }
static matlab_mat *mat_zeros(int64_t r, int64_t c) {
    matlab_mat *m = mat_alloc(r, c);
    for (int64_t i = 0; i < r * c; ++i) m->data[i] = 0.0;
    return m;
}
static matlab_mat *mat_clone(const matlab_mat *src) {
    if (!src) return mat_zeros(0, 0);
    matlab_mat *o = mat_alloc(src->rows, src->cols);
    int64_t n = src->rows * src->cols;
    for (int64_t i = 0; i < n; ++i) o->data[i] = src->data[i];
    return o;
}

/* =====================================================================
 * Protocol-Buffers wire-format primitives.
 *
 * Tag layout: (field_number << 3) | wire_type.  Wire types we support:
 *   0  varint            (varlen, 7 bits per byte, MSB = continuation)
 *   1  fixed64           (8 bytes little-endian)
 *   2  length-delimited  (varint length + payload bytes)
 *   5  fixed32           (4 bytes little-endian)
 *
 * No protoc — every ONNX field is decoded by hand against onnx.proto3
 * field numbers documented inline at each call site.
 * =================================================================== */

struct PbReader {
    const uint8_t *cur;
    const uint8_t *end;
    bool ok = true;
    bool eof() const { return !ok || cur >= end; }
};

static uint64_t pb_read_varint(PbReader &r) {
    uint64_t v = 0;
    int shift = 0;
    while (r.cur < r.end) {
        uint8_t b = *r.cur++;
        v |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if ((b & 0x80u) == 0) return v;
        shift += 7;
        if (shift > 63) { r.ok = false; return 0; }
    }
    r.ok = false;
    return 0;
}
static uint32_t pb_read_fixed32(PbReader &r) {
    if (r.cur + 4 > r.end) { r.ok = false; return 0; }
    uint32_t v = static_cast<uint32_t>(r.cur[0]) |
                 (static_cast<uint32_t>(r.cur[1]) << 8) |
                 (static_cast<uint32_t>(r.cur[2]) << 16) |
                 (static_cast<uint32_t>(r.cur[3]) << 24);
    r.cur += 4;
    return v;
}
static uint64_t pb_read_fixed64(PbReader &r) {
    if (r.cur + 8 > r.end) { r.ok = false; return 0; }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(r.cur[i]) << (8 * i);
    r.cur += 8;
    return v;
}
static float pb_read_float(PbReader &r) {
    uint32_t bits = pb_read_fixed32(r);
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}
static double pb_read_double(PbReader &r) {
    uint64_t bits = pb_read_fixed64(r);
    double d;
    std::memcpy(&d, &bits, sizeof(double));
    return d;
}
static std::string pb_read_string(PbReader &r) {
    uint64_t n = pb_read_varint(r);
    if (!r.ok || r.cur + n > r.end) { r.ok = false; return {}; }
    std::string s(reinterpret_cast<const char *>(r.cur),
                  reinterpret_cast<const char *>(r.cur + n));
    r.cur += n;
    return s;
}
static void pb_read_bytes(PbReader &r, std::vector<uint8_t> &out) {
    uint64_t n = pb_read_varint(r);
    if (!r.ok || r.cur + n > r.end) { r.ok = false; return; }
    out.assign(r.cur, r.cur + n);
    r.cur += n;
}
static PbReader pb_subreader(PbReader &r) {
    uint64_t n = pb_read_varint(r);
    PbReader sub;
    if (!r.ok || r.cur + n > r.end) { sub.ok = false; sub.cur = sub.end = nullptr; return sub; }
    sub.cur = r.cur;
    sub.end = r.cur + n;
    r.cur += n;
    return sub;
}
static void pb_skip_field(PbReader &r, uint32_t wire) {
    if (wire == 0) {
        pb_read_varint(r);
    } else if (wire == 1) {
        pb_read_fixed64(r);
    } else if (wire == 2) {
        uint64_t n = pb_read_varint(r);
        if (r.cur + n > r.end) { r.ok = false; return; }
        r.cur += n;
    } else if (wire == 5) {
        pb_read_fixed32(r);
    } else {
        r.ok = false;
    }
}

/* Packed-repeated decoders — ONNX serialises repeated int64/float arrays as
 * one length-delimited field containing back-to-back varints / fixed values
 * (proto3's "packed" rule). */
static void pb_read_packed_varint_i64(PbReader &r, std::vector<int64_t> &out) {
    PbReader sub = pb_subreader(r);
    while (sub.ok && sub.cur < sub.end) {
        out.push_back(static_cast<int64_t>(pb_read_varint(sub)));
    }
}
static void pb_read_packed_float(PbReader &r, std::vector<double> &out) {
    PbReader sub = pb_subreader(r);
    while (sub.ok && sub.cur + 4 <= sub.end) {
        out.push_back(static_cast<double>(pb_read_float(sub)));
    }
}
static void pb_read_packed_double(PbReader &r, std::vector<double> &out) {
    PbReader sub = pb_subreader(r);
    while (sub.ok && sub.cur + 8 <= sub.end) {
        out.push_back(pb_read_double(sub));
    }
}

/* Writer. */
struct PbWriter {
    std::vector<uint8_t> buf;
};

static void pbw_varint(PbWriter &w, uint64_t v) {
    while (v >= 0x80) {
        w.buf.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
        v >>= 7;
    }
    w.buf.push_back(static_cast<uint8_t>(v));
}
static void pbw_fixed32(PbWriter &w, uint32_t v) {
    w.buf.push_back(static_cast<uint8_t>(v & 0xFF));
    w.buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    w.buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    w.buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
static void pbw_fixed64(PbWriter &w, uint64_t v) {
    for (int i = 0; i < 8; ++i) w.buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
static void pbw_float(PbWriter &w, float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    pbw_fixed32(w, bits);
}
static void pbw_double(PbWriter &w, double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, sizeof(uint64_t));
    pbw_fixed64(w, bits);
}
static void pbw_tag(PbWriter &w, uint32_t field, uint32_t wire) {
    pbw_varint(w, (static_cast<uint64_t>(field) << 3) | wire);
}
static void pbw_field_varint(PbWriter &w, uint32_t field, uint64_t v) {
    pbw_tag(w, field, 0);
    pbw_varint(w, v);
}
static void pbw_field_sint64(PbWriter &w, uint32_t field, int64_t v) {
    pbw_field_varint(w, field, static_cast<uint64_t>(v));
}
static void pbw_field_bytes(PbWriter &w, uint32_t field, const uint8_t *data, size_t n) {
    pbw_tag(w, field, 2);
    pbw_varint(w, n);
    w.buf.insert(w.buf.end(), data, data + n);
}
static void pbw_field_string(PbWriter &w, uint32_t field, const std::string &s) {
    pbw_field_bytes(w, field, reinterpret_cast<const uint8_t *>(s.data()), s.size());
}
static void pbw_field_message(PbWriter &w, uint32_t field, const PbWriter &body) {
    pbw_field_bytes(w, field, body.buf.data(), body.buf.size());
}
static void pbw_field_packed_i64(PbWriter &w, uint32_t field, const std::vector<int64_t> &vs) {
    PbWriter sub;
    for (auto v : vs) pbw_varint(sub, static_cast<uint64_t>(v));
    pbw_field_message(w, field, sub);
}
static void pbw_field_packed_float(PbWriter &w, uint32_t field, const std::vector<double> &vs) {
    PbWriter sub;
    for (auto v : vs) pbw_float(sub, static_cast<float>(v));
    pbw_field_message(w, field, sub);
}

/* =====================================================================
 * ONNX schema types (subset that covers shipped DL ops).
 * =================================================================== */

enum OnnxAttrType : int32_t {
    OA_UNDEFINED = 0, OA_FLOAT = 1, OA_INT = 2, OA_STRING = 3,
    OA_TENSOR = 4, OA_GRAPH = 5,
    OA_FLOATS = 6, OA_INTS = 7, OA_STRINGS = 8, OA_TENSORS = 9
};
enum OnnxDataType : int32_t {
    ODT_UNDEFINED = 0, ODT_FLOAT = 1, ODT_UINT8 = 2, ODT_INT8 = 3,
    ODT_UINT16 = 4, ODT_INT16 = 5, ODT_INT32 = 6, ODT_INT64 = 7,
    ODT_STRING = 8, ODT_BOOL = 9, ODT_FLOAT16 = 10, ODT_DOUBLE = 11,
    ODT_UINT32 = 12, ODT_UINT64 = 13
};

struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = ODT_FLOAT;
    std::vector<double> data;     /* unified storage as double */
};

struct OnnxAttribute {
    std::string name;
    int32_t type = OA_UNDEFINED;
    double f = 0.0;
    int64_t i = 0;
    std::string s;
    OnnxTensor t;
    std::vector<double> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
};

struct OnnxNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> input;
    std::vector<std::string> output;
    std::vector<OnnxAttribute> attribute;
};

struct OnnxValueInfo {
    std::string name;
    std::vector<int64_t> dims;
};

struct OnnxGraph {
    std::string name = "graph";
    std::vector<OnnxNode> node;
    std::vector<OnnxTensor> initializer;
    std::vector<OnnxValueInfo> input;
    std::vector<OnnxValueInfo> output;
};

struct OnnxModel {
    int64_t ir_version = 7;
    std::string producer_name = "matlab_llvm";
    int64_t opset_version = 13;
    OnnxGraph graph;
};

/* =====================================================================
 * Decoders.  Field numbers from onnx.proto3 (R2026a uses the same
 * versioned schema since the spec is stable across opset 7-21).
 * =================================================================== */

static void decode_tensor(PbReader &r, OnnxTensor &t);

static void decode_attribute(PbReader &r, OnnxAttribute &a) {
    while (!r.eof()) {
        uint64_t tag = pb_read_varint(r);
        if (!r.ok) return;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        switch (field) {
            case 1: a.name = pb_read_string(r); break;                    /* name */
            case 20: a.type = static_cast<int32_t>(pb_read_varint(r)); break; /* type (enum) */
            case 2: a.f = static_cast<double>(pb_read_float(r)); break;   /* f */
            case 3: a.i = static_cast<int64_t>(pb_read_varint(r)); break; /* i */
            case 4: a.s = pb_read_string(r); break;                       /* s */
            case 5: { PbReader sub = pb_subreader(r); decode_tensor(sub, a.t); break; } /* t */
            case 7: pb_read_packed_float(r, a.floats); break;             /* floats (packed) */
            case 8: pb_read_packed_varint_i64(r, a.ints); break;          /* ints (packed) */
            case 9: a.strings.push_back(pb_read_string(r)); break;        /* strings (repeated) */
            default: pb_skip_field(r, wire); break;
        }
    }
}

static void decode_tensor(PbReader &r, OnnxTensor &t) {
    std::vector<uint8_t> raw;
    bool have_raw = false;
    while (!r.eof()) {
        uint64_t tag = pb_read_varint(r);
        if (!r.ok) return;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        switch (field) {
            case 1: pb_read_packed_varint_i64(r, t.dims); break;          /* dims */
            case 2: t.data_type = static_cast<int32_t>(pb_read_varint(r)); break; /* data_type */
            case 4: pb_read_packed_float(r, t.data); break;               /* float_data */
            case 7: pb_read_packed_varint_i64(r, *reinterpret_cast<std::vector<int64_t>*>(&t.data)); break; /* int64_data (we widen later) */
            case 8: t.name = pb_read_string(r); break;                    /* name */
            case 9: { pb_read_bytes(r, raw); have_raw = true; break; }    /* raw_data */
            case 10: pb_read_packed_double(r, t.data); break;             /* double_data */
            default: pb_skip_field(r, wire); break;
        }
    }
    /* Reinterpret raw_data per data_type. */
    if (have_raw && !raw.empty()) {
        t.data.clear();
        if (t.data_type == ODT_FLOAT) {
            size_t n = raw.size() / 4;
            const uint8_t *p = raw.data();
            for (size_t i = 0; i < n; ++i) {
                uint32_t bits = static_cast<uint32_t>(p[0]) |
                                (static_cast<uint32_t>(p[1]) << 8) |
                                (static_cast<uint32_t>(p[2]) << 16) |
                                (static_cast<uint32_t>(p[3]) << 24);
                float f;
                std::memcpy(&f, &bits, 4);
                t.data.push_back(static_cast<double>(f));
                p += 4;
            }
        } else if (t.data_type == ODT_DOUBLE) {
            size_t n = raw.size() / 8;
            const uint8_t *p = raw.data();
            for (size_t i = 0; i < n; ++i) {
                uint64_t bits = 0;
                for (int b = 0; b < 8; ++b) bits |= static_cast<uint64_t>(p[b]) << (8 * b);
                double d;
                std::memcpy(&d, &bits, 8);
                t.data.push_back(d);
                p += 8;
            }
        } else if (t.data_type == ODT_INT64) {
            size_t n = raw.size() / 8;
            const uint8_t *p = raw.data();
            for (size_t i = 0; i < n; ++i) {
                int64_t v = 0;
                for (int b = 0; b < 8; ++b) v |= static_cast<int64_t>(p[b]) << (8 * b);
                t.data.push_back(static_cast<double>(v));
                p += 8;
            }
        } else if (t.data_type == ODT_INT32) {
            size_t n = raw.size() / 4;
            const uint8_t *p = raw.data();
            for (size_t i = 0; i < n; ++i) {
                int32_t v = static_cast<int32_t>(p[0]) |
                            (static_cast<int32_t>(p[1]) << 8) |
                            (static_cast<int32_t>(p[2]) << 16) |
                            (static_cast<int32_t>(p[3]) << 24);
                t.data.push_back(static_cast<double>(v));
                p += 4;
            }
        }
    }
}

static void decode_node(PbReader &r, OnnxNode &n) {
    while (!r.eof()) {
        uint64_t tag = pb_read_varint(r);
        if (!r.ok) return;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        switch (field) {
            case 1: n.input.push_back(pb_read_string(r)); break;
            case 2: n.output.push_back(pb_read_string(r)); break;
            case 3: n.name = pb_read_string(r); break;
            case 4: n.op_type = pb_read_string(r); break;
            case 5: { PbReader sub = pb_subreader(r); OnnxAttribute a; decode_attribute(sub, a); n.attribute.push_back(std::move(a)); break; }
            default: pb_skip_field(r, wire); break;
        }
    }
}

static void decode_value_info(PbReader &r, OnnxValueInfo &v) {
    while (!r.eof()) {
        uint64_t tag = pb_read_varint(r);
        if (!r.ok) return;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        switch (field) {
            case 1: v.name = pb_read_string(r); break;
            case 2: pb_skip_field(r, wire); break;   /* type (we don't need full shape here) */
            default: pb_skip_field(r, wire); break;
        }
    }
}

static void decode_graph(PbReader &r, OnnxGraph &g) {
    while (!r.eof()) {
        uint64_t tag = pb_read_varint(r);
        if (!r.ok) return;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        switch (field) {
            case 1: { PbReader sub = pb_subreader(r); OnnxNode n; decode_node(sub, n); g.node.push_back(std::move(n)); break; }
            case 2: g.name = pb_read_string(r); break;
            case 5: { PbReader sub = pb_subreader(r); OnnxTensor t; decode_tensor(sub, t); g.initializer.push_back(std::move(t)); break; }
            case 11: { PbReader sub = pb_subreader(r); OnnxValueInfo v; decode_value_info(sub, v); g.input.push_back(std::move(v)); break; }
            case 12: { PbReader sub = pb_subreader(r); OnnxValueInfo v; decode_value_info(sub, v); g.output.push_back(std::move(v)); break; }
            default: pb_skip_field(r, wire); break;
        }
    }
}

static void decode_model(PbReader &r, OnnxModel &m) {
    while (!r.eof()) {
        uint64_t tag = pb_read_varint(r);
        if (!r.ok) return;
        uint32_t field = static_cast<uint32_t>(tag >> 3);
        uint32_t wire = static_cast<uint32_t>(tag & 7);
        switch (field) {
            case 1: m.ir_version = static_cast<int64_t>(pb_read_varint(r)); break;
            case 2: m.producer_name = pb_read_string(r); break;
            case 7: { PbReader sub = pb_subreader(r); decode_graph(sub, m.graph); break; }
            case 8: { /* opset_import (OperatorSetIdProto) — peek at version. */
                PbReader sub = pb_subreader(r);
                while (sub.ok && sub.cur < sub.end) {
                    uint64_t st = pb_read_varint(sub);
                    if (!sub.ok) break;
                    uint32_t sf = static_cast<uint32_t>(st >> 3);
                    uint32_t sw = static_cast<uint32_t>(st & 7);
                    if (sf == 2 && sw == 0) m.opset_version = static_cast<int64_t>(pb_read_varint(sub));
                    else pb_skip_field(sub, sw);
                }
                break;
            }
            default: pb_skip_field(r, wire); break;
        }
    }
}

/* =====================================================================
 * Encoders.
 * =================================================================== */

static void encode_tensor(PbWriter &w, const OnnxTensor &t) {
    /* dims (packed int64). */
    if (!t.dims.empty()) pbw_field_packed_i64(w, 1, t.dims);
    /* data_type. */
    pbw_field_varint(w, 2, static_cast<uint64_t>(t.data_type));
    /* name. */
    if (!t.name.empty()) pbw_field_string(w, 8, t.name);
    /* Storage: prefer raw_data (bytes) for compactness, matching what
     * torch/tf2onnx emit. */
    if (t.data_type == ODT_FLOAT) {
        PbWriter raw;
        for (double v : t.data) {
            uint32_t bits;
            float f = static_cast<float>(v);
            std::memcpy(&bits, &f, 4);
            raw.buf.push_back(static_cast<uint8_t>(bits & 0xFF));
            raw.buf.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
            raw.buf.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
            raw.buf.push_back(static_cast<uint8_t>((bits >> 24) & 0xFF));
        }
        pbw_field_bytes(w, 9, raw.buf.data(), raw.buf.size());
    } else if (t.data_type == ODT_DOUBLE) {
        PbWriter raw;
        for (double v : t.data) {
            uint64_t bits;
            std::memcpy(&bits, &v, 8);
            for (int b = 0; b < 8; ++b) raw.buf.push_back(static_cast<uint8_t>((bits >> (8 * b)) & 0xFF));
        }
        pbw_field_bytes(w, 9, raw.buf.data(), raw.buf.size());
    } else if (t.data_type == ODT_INT64) {
        PbWriter raw;
        for (double v : t.data) {
            int64_t iv = static_cast<int64_t>(v);
            for (int b = 0; b < 8; ++b) raw.buf.push_back(static_cast<uint8_t>((iv >> (8 * b)) & 0xFF));
        }
        pbw_field_bytes(w, 9, raw.buf.data(), raw.buf.size());
    }
}

static void encode_attribute(PbWriter &w, const OnnxAttribute &a) {
    if (!a.name.empty()) pbw_field_string(w, 1, a.name);
    pbw_field_varint(w, 20, static_cast<uint64_t>(a.type));
    switch (a.type) {
        case OA_FLOAT: pbw_tag(w, 2, 5); pbw_float(w, static_cast<float>(a.f)); break;
        case OA_INT: pbw_field_sint64(w, 3, a.i); break;
        case OA_STRING: pbw_field_string(w, 4, a.s); break;
        case OA_TENSOR: { PbWriter sub; encode_tensor(sub, a.t); pbw_field_message(w, 5, sub); break; }
        case OA_FLOATS: pbw_field_packed_float(w, 7, a.floats); break;
        case OA_INTS: pbw_field_packed_i64(w, 8, a.ints); break;
        case OA_STRINGS: for (auto &s : a.strings) pbw_field_string(w, 9, s); break;
        default: break;
    }
}

static void encode_node(PbWriter &w, const OnnxNode &n) {
    for (auto &s : n.input) pbw_field_string(w, 1, s);
    for (auto &s : n.output) pbw_field_string(w, 2, s);
    if (!n.name.empty()) pbw_field_string(w, 3, n.name);
    pbw_field_string(w, 4, n.op_type);
    for (auto &a : n.attribute) { PbWriter sub; encode_attribute(sub, a); pbw_field_message(w, 5, sub); }
}

static void encode_value_info(PbWriter &w, const OnnxValueInfo &v) {
    pbw_field_string(w, 1, v.name);
    /* Emit a minimal TypeProto with tensor_type containing FLOAT + shape. */
    PbWriter ttype;     /* TypeProto */
    PbWriter ttensor;   /* TypeProto.Tensor */
    pbw_field_varint(ttensor, 1, ODT_FLOAT);  /* elem_type */
    PbWriter shape;
    for (int64_t d : v.dims) {
        PbWriter dim;
        pbw_field_varint(dim, 1, static_cast<uint64_t>(d));  /* dim_value */
        pbw_field_message(shape, 1, dim);  /* dims (repeated Dimension) */
    }
    pbw_field_message(ttensor, 2, shape);
    pbw_field_message(ttype, 1, ttensor);  /* tensor_type (field 1) */
    pbw_field_message(w, 2, ttype);        /* type */
}

static void encode_graph(PbWriter &w, const OnnxGraph &g) {
    for (auto &n : g.node) { PbWriter sub; encode_node(sub, n); pbw_field_message(w, 1, sub); }
    pbw_field_string(w, 2, g.name);
    for (auto &t : g.initializer) { PbWriter sub; encode_tensor(sub, t); pbw_field_message(w, 5, sub); }
    for (auto &v : g.input) { PbWriter sub; encode_value_info(sub, v); pbw_field_message(w, 11, sub); }
    for (auto &v : g.output) { PbWriter sub; encode_value_info(sub, v); pbw_field_message(w, 12, sub); }
}

static void encode_model(PbWriter &w, const OnnxModel &m) {
    pbw_field_varint(w, 1, static_cast<uint64_t>(m.ir_version));
    pbw_field_string(w, 2, m.producer_name);
    PbWriter g;
    encode_graph(g, m.graph);
    pbw_field_message(w, 7, g);
    /* opset_import (repeated). */
    PbWriter os;
    pbw_field_varint(os, 2, static_cast<uint64_t>(m.opset_version));  /* version */
    pbw_field_message(w, 8, os);
}

/* =====================================================================
 * Op execution dispatcher.
 *
 * Tensors flow through as plain `matlab_mat *`.  Each op handler looks up
 * its inputs by name, performs the computation, and stores the output by
 * name into the execution state.  Initializers are pre-populated into
 * the state before the first node runs.
 *
 * Memory: handlers freely allocate fresh `matlab_mat`s via mat_alloc; the
 * runtime arena-frees on cleanup (we keep ownership simple — no explicit
 * frees).
 * =================================================================== */

struct ExecState {
    std::unordered_map<std::string, matlab_mat *> tensors;
    std::string error;
    bool ok() const { return error.empty(); }
};

/* Attribute lookup helpers. */
static const OnnxAttribute *find_attr(const OnnxNode &n, const std::string &name) {
    for (auto &a : n.attribute) if (a.name == name) return &a;
    return nullptr;
}
static int64_t attr_i(const OnnxNode &n, const std::string &name, int64_t dflt) {
    auto *a = find_attr(n, name);
    return a ? a->i : dflt;
}
static double attr_f(const OnnxNode &n, const std::string &name, double dflt) {
    auto *a = find_attr(n, name);
    return a ? a->f : dflt;
}
static std::vector<int64_t> attr_ints(const OnnxNode &n, const std::string &name) {
    auto *a = find_attr(n, name);
    return a ? a->ints : std::vector<int64_t>{};
}
static matlab_mat *get_in(const OnnxNode &n, ExecState &s, size_t idx) {
    if (idx >= n.input.size()) return nullptr;
    auto it = s.tensors.find(n.input[idx]);
    return (it != s.tensors.end()) ? it->second : nullptr;
}
static void put_out(const OnnxNode &n, ExecState &s, size_t idx, matlab_mat *m) {
    if (idx >= n.output.size()) return;
    s.tensors[n.output[idx]] = m;
}

/* ---- Op implementations -------------------------------------------- */
/* Each returns true on success, false on error (s.error filled). */

static bool op_Identity(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Identity: missing input"; return false; }
    put_out(n, s, 0, mat_clone(A));
    return true;
}

static bool op_MatMul(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "MatMul: missing input"; return false; }
    put_out(n, s, 0, matlab_matmul_mm(A, B));
    return true;
}

/* Y = α A^[transA] B^[transB] + β C.  ONNX-13 default α=β=1, transA=transB=0. */
static bool op_Gemm(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1), *C = get_in(n, s, 2);
    if (!A || !B) { s.error = "Gemm: missing input"; return false; }
    double alpha = attr_f(n, "alpha", 1.0);
    double beta  = attr_f(n, "beta",  1.0);
    int64_t tA = attr_i(n, "transA", 0);
    int64_t tB = attr_i(n, "transB", 0);
    /* Materialise transposes locally. */
    matlab_mat *Au = A, *Bu = B;
    matlab_mat *At = nullptr, *Bt = nullptr;
    if (tA) {
        At = mat_alloc(A->cols, A->rows);
        for (int64_t i = 0; i < A->rows; ++i)
            for (int64_t j = 0; j < A->cols; ++j)
                At->data[j * A->rows + i] = A->data[i * A->cols + j];
        Au = At;
    }
    if (tB) {
        Bt = mat_alloc(B->cols, B->rows);
        for (int64_t i = 0; i < B->rows; ++i)
            for (int64_t j = 0; j < B->cols; ++j)
                Bt->data[j * B->rows + i] = B->data[i * B->cols + j];
        Bu = Bt;
    }
    matlab_mat *Y = matlab_matmul_mm(Au, Bu);
    int64_t M = Y->rows, K = Y->cols, nY = M * K;
    if (alpha != 1.0) for (int64_t i = 0; i < nY; ++i) Y->data[i] *= alpha;
    if (C) {
        int64_t nC = mat_nelem(C);
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < K; ++j) {
                double c;
                if (nC == nY) c = C->data[i * K + j];
                else if (nC == K && C->cols == K) c = C->data[j];  /* row-broadcast 1xK */
                else if (nC == M && C->rows == M) c = C->data[i];  /* col-broadcast Mx1 */
                else c = C->data[i * K + j];                       /* size-matched fallback */
                Y->data[i * K + j] += beta * c;
            }
        }
    }
    put_out(n, s, 0, Y);
    return true;
}

/* Elementwise binary with numpy-style broadcasting on 2-D mats. */
enum BinKind { BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV };
static matlab_mat *bin_broadcast(matlab_mat *A, matlab_mat *B, BinKind k) {
    int64_t Ar = A->rows, Ac = A->cols, Br = B->rows, Bc = B->cols;
    int64_t Or = std::max(Ar, Br), Oc = std::max(Ac, Bc);
    matlab_mat *Y = mat_alloc(Or, Oc);
    for (int64_t i = 0; i < Or; ++i) {
        for (int64_t j = 0; j < Oc; ++j) {
            int64_t ai = (Ar == 1) ? 0 : i;
            int64_t aj = (Ac == 1) ? 0 : j;
            int64_t bi = (Br == 1) ? 0 : i;
            int64_t bj = (Bc == 1) ? 0 : j;
            double a = A->data[ai * Ac + aj];
            double b = B->data[bi * Bc + bj];
            double y;
            switch (k) {
                case BIN_ADD: y = a + b; break;
                case BIN_SUB: y = a - b; break;
                case BIN_MUL: y = a * b; break;
                case BIN_DIV: y = a / b; break;
            }
            Y->data[i * Oc + j] = y;
        }
    }
    return Y;
}
static bool op_Add(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Add: missing input"; return false; }
    put_out(n, s, 0, bin_broadcast(A, B, BIN_ADD)); return true;
}
static bool op_Sub(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Sub: missing input"; return false; }
    put_out(n, s, 0, bin_broadcast(A, B, BIN_SUB)); return true;
}
static bool op_Mul(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Mul: missing input"; return false; }
    put_out(n, s, 0, bin_broadcast(A, B, BIN_MUL)); return true;
}
static bool op_Div(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Div: missing input"; return false; }
    put_out(n, s, 0, bin_broadcast(A, B, BIN_DIV)); return true;
}

/* Elementwise unary. */
static matlab_mat *map_unary(matlab_mat *A, double (*f)(double)) {
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t n = mat_nelem(A);
    for (int64_t i = 0; i < n; ++i) Y->data[i] = f(A->data[i]);
    return Y;
}
static double u_relu(double x) { return x > 0 ? x : 0; }
static double u_sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double u_tanh(double x) { return std::tanh(x); }
static double u_exp(double x) { return std::exp(x); }
static double u_log(double x) { return std::log(x); }
static double u_sqrt(double x) { return std::sqrt(x); }
static double u_abs(double x) { return std::fabs(x); }
static double u_neg(double x) { return -x; }
static double u_floor(double x) { return std::floor(x); }
static double u_ceil(double x) { return std::ceil(x); }
static double u_round(double x) { return std::round(x); }
static double u_softplus(double x) { return std::log1p(std::exp(x)); }
static double u_softsign(double x) { return x / (1.0 + std::fabs(x)); }
static double u_reciprocal(double x) { return 1.0 / x; }

#define DEF_UN(NAME, FN) static bool op_##NAME(const OnnxNode &n, ExecState &s) { \
    matlab_mat *A = get_in(n, s, 0); if (!A) { s.error = #NAME ": missing input"; return false; } \
    put_out(n, s, 0, map_unary(A, FN)); return true; }
DEF_UN(Relu, u_relu)
DEF_UN(Sigmoid, u_sigmoid)
DEF_UN(Tanh, u_tanh)
DEF_UN(Exp, u_exp)
DEF_UN(Log, u_log)
DEF_UN(Sqrt, u_sqrt)
DEF_UN(Abs, u_abs)
DEF_UN(Neg, u_neg)
DEF_UN(Floor, u_floor)
DEF_UN(Ceil, u_ceil)
DEF_UN(Round, u_round)
DEF_UN(Softplus, u_softplus)
DEF_UN(Softsign, u_softsign)
DEF_UN(Reciprocal, u_reciprocal)
#undef DEF_UN

static bool op_LeakyRelu(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "LeakyRelu: missing input"; return false; }
    double alpha = attr_f(n, "alpha", 0.01);
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) Y->data[i] = A->data[i] > 0 ? A->data[i] : alpha * A->data[i];
    put_out(n, s, 0, Y);
    return true;
}
static bool op_Elu(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Elu: missing input"; return false; }
    double alpha = attr_f(n, "alpha", 1.0);
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) {
        double x = A->data[i];
        Y->data[i] = x >= 0 ? x : alpha * (std::exp(x) - 1.0);
    }
    put_out(n, s, 0, Y);
    return true;
}
static bool op_Selu(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Selu: missing input"; return false; }
    double alpha = attr_f(n, "alpha", 1.67326319217681884765625);
    double gamma = attr_f(n, "gamma", 1.05070102214813232421875);
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) {
        double x = A->data[i];
        Y->data[i] = gamma * (x > 0 ? x : alpha * (std::exp(x) - 1.0));
    }
    put_out(n, s, 0, Y);
    return true;
}
static double gelu_d(double x) {
    /* exact gelu via erf — same as ONNX-20's "Gelu" with approximate="none". */
    return 0.5 * x * (1.0 + std::erf(x / 1.41421356237309504880));
}
static double gelu_approx(double x) {
    /* tanh-approx Gelu (Hendrycks & Gimpel) — ONNX-20 approximate="tanh". */
    double a = 0.7978845608028654;   /* sqrt(2/pi) */
    double y = a * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std::tanh(y));
}
static bool op_Gelu(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Gelu: missing input"; return false; }
    auto *a = find_attr(n, "approximate");
    bool approx = a && a->s == "tanh";
    put_out(n, s, 0, map_unary(A, approx ? gelu_approx : gelu_d));
    return true;
}
static bool op_HardSigmoid(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "HardSigmoid: missing input"; return false; }
    double alpha = attr_f(n, "alpha", 0.2);
    double beta  = attr_f(n, "beta",  0.5);
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) {
        double v = alpha * A->data[i] + beta;
        if (v < 0) v = 0; else if (v > 1) v = 1;
        Y->data[i] = v;
    }
    put_out(n, s, 0, Y);
    return true;
}
static bool op_HardSwish(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "HardSwish: missing input"; return false; }
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) {
        double x = A->data[i];
        double s2 = (x + 3.0) / 6.0;
        if (s2 < 0) s2 = 0; else if (s2 > 1) s2 = 1;
        Y->data[i] = x * s2;
    }
    put_out(n, s, 0, Y);
    return true;
}

/* Softmax: opset-13 axis defaults to -1 (last).  For 2-D, axis=-1 / axis=1
 * normalises along columns (per row); axis=0 along rows. */
static bool op_Softmax(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Softmax: missing input"; return false; }
    int64_t axis = attr_i(n, "axis", -1);
    if (axis < 0) axis += 2;   /* 2-D fallback */
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    if (axis == 1) {
        /* per-row */
        for (int64_t i = 0; i < A->rows; ++i) {
            double mx = A->data[i * A->cols];
            for (int64_t j = 1; j < A->cols; ++j) if (A->data[i * A->cols + j] > mx) mx = A->data[i * A->cols + j];
            double sum = 0.0;
            for (int64_t j = 0; j < A->cols; ++j) { double e = std::exp(A->data[i * A->cols + j] - mx); Y->data[i * A->cols + j] = e; sum += e; }
            for (int64_t j = 0; j < A->cols; ++j) Y->data[i * A->cols + j] /= sum;
        }
    } else {
        /* per-column */
        for (int64_t j = 0; j < A->cols; ++j) {
            double mx = A->data[j];
            for (int64_t i = 1; i < A->rows; ++i) if (A->data[i * A->cols + j] > mx) mx = A->data[i * A->cols + j];
            double sum = 0.0;
            for (int64_t i = 0; i < A->rows; ++i) { double e = std::exp(A->data[i * A->cols + j] - mx); Y->data[i * A->cols + j] = e; sum += e; }
            for (int64_t i = 0; i < A->rows; ++i) Y->data[i * A->cols + j] /= sum;
        }
    }
    put_out(n, s, 0, Y);
    return true;
}
static bool op_LogSoftmax(const OnnxNode &n, ExecState &s) {
    /* LogSoftmax = log(softmax).  Reuse the kernel + log. */
    if (!op_Softmax(n, s)) return false;
    matlab_mat *Y = s.tensors[n.output[0]];
    int64_t nn = mat_nelem(Y);
    for (int64_t i = 0; i < nn; ++i) Y->data[i] = std::log(Y->data[i]);
    return true;
}

/* Reshape: input 0 is the data, input 1 is a 1-D int64 tensor of dims.
 * For 2-D we accept any [rows cols] target; -1 / 0 special values are
 * resolved against the input shape. */
static bool op_Reshape(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *shape = get_in(n, s, 1);
    if (!A || !shape) { s.error = "Reshape: missing input"; return false; }
    int64_t k = mat_nelem(shape);
    if (k < 1 || k > 2) { s.error = "Reshape: only 1-D or 2-D targets supported"; return false; }
    int64_t r = (k >= 1) ? static_cast<int64_t>(shape->data[0]) : -1;
    int64_t c = (k >= 2) ? static_cast<int64_t>(shape->data[1]) : 1;
    int64_t total = A->rows * A->cols;
    if (r == 0) r = A->rows;
    if (c == 0) c = A->cols;
    if (r == -1) r = (c > 0) ? total / c : total;
    if (c == -1) c = (r > 0) ? total / r : total;
    if (r * c != total) { s.error = "Reshape: dims don't match numel"; return false; }
    matlab_mat *Y = mat_alloc(r, c);
    for (int64_t i = 0; i < total; ++i) Y->data[i] = A->data[i];
    put_out(n, s, 0, Y);
    return true;
}
static bool op_Flatten(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Flatten: missing input"; return false; }
    int64_t axis = attr_i(n, "axis", 1);
    /* Flatten to (prod(dims<axis), prod(dims>=axis)).  For 2-D input the
     * only useful axis is 0 (-> 1xN) or 1 (-> already 2-D, the identity
     * grouping the rows axis).  We treat axis=0 as 1xN and any axis>=1 as
     * the input unchanged. */
    if (axis == 0) {
        matlab_mat *Y = mat_alloc(1, A->rows * A->cols);
        int64_t n2 = A->rows * A->cols;
        for (int64_t i = 0; i < n2; ++i) Y->data[i] = A->data[i];
        put_out(n, s, 0, Y);
    } else {
        put_out(n, s, 0, mat_clone(A));
    }
    return true;
}
static bool op_Transpose(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Transpose: missing input"; return false; }
    /* For 2-D the only meaningful perm is [1, 0]. */
    matlab_mat *Y = mat_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i)
        for (int64_t j = 0; j < A->cols; ++j)
            Y->data[j * A->rows + i] = A->data[i * A->cols + j];
    put_out(n, s, 0, Y);
    return true;
}
static bool op_Concat(const OnnxNode &n, ExecState &s) {
    if (n.input.empty()) { s.error = "Concat: no inputs"; return false; }
    int64_t axis = attr_i(n, "axis", 0);
    if (axis < 0) axis += 2;
    /* Gather inputs. */
    std::vector<matlab_mat *> As;
    for (size_t k = 0; k < n.input.size(); ++k) {
        matlab_mat *A = get_in(n, s, k);
        if (!A) { s.error = "Concat: missing input"; return false; }
        As.push_back(A);
    }
    if (axis == 0) {
        int64_t rsum = 0, c = As[0]->cols;
        for (auto *A : As) {
            if (A->cols != c) { s.error = "Concat axis=0: col mismatch"; return false; }
            rsum += A->rows;
        }
        matlab_mat *Y = mat_alloc(rsum, c);
        int64_t rofs = 0;
        for (auto *A : As) {
            for (int64_t i = 0; i < A->rows; ++i)
                for (int64_t j = 0; j < c; ++j)
                    Y->data[(rofs + i) * c + j] = A->data[i * c + j];
            rofs += A->rows;
        }
        put_out(n, s, 0, Y);
    } else {
        int64_t r = As[0]->rows, csum = 0;
        for (auto *A : As) {
            if (A->rows != r) { s.error = "Concat axis=1: row mismatch"; return false; }
            csum += A->cols;
        }
        matlab_mat *Y = mat_alloc(r, csum);
        int64_t cofs = 0;
        for (auto *A : As) {
            for (int64_t i = 0; i < r; ++i)
                for (int64_t j = 0; j < A->cols; ++j)
                    Y->data[i * csum + (cofs + j)] = A->data[i * A->cols + j];
            cofs += A->cols;
        }
        put_out(n, s, 0, Y);
    }
    return true;
}

/* Reductions.  Default axes = all; keepdims preserves a 1 dim. */
static matlab_mat *reduce_all(matlab_mat *A, double (*combine)(double, double), double init) {
    int64_t n2 = mat_nelem(A);
    double acc = init;
    for (int64_t i = 0; i < n2; ++i) acc = combine(acc, A->data[i]);
    matlab_mat *Y = mat_alloc(1, 1);
    Y->data[0] = acc;
    return Y;
}
static double comb_sum(double a, double b) { return a + b; }
static double comb_max(double a, double b) { return a > b ? a : b; }
static double comb_min(double a, double b) { return a < b ? a : b; }
static double comb_prod(double a, double b) { return a * b; }
static bool op_ReduceSum(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "ReduceSum: missing input"; return false; }
    put_out(n, s, 0, reduce_all(A, comb_sum, 0.0));
    return true;
}
static bool op_ReduceMean(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "ReduceMean: missing input"; return false; }
    int64_t nn = mat_nelem(A);
    matlab_mat *Y = reduce_all(A, comb_sum, 0.0);
    if (nn > 0) Y->data[0] /= static_cast<double>(nn);
    put_out(n, s, 0, Y);
    return true;
}
static bool op_ReduceMax(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A || mat_nelem(A) == 0) { s.error = "ReduceMax: missing input"; return false; }
    put_out(n, s, 0, reduce_all(A, comb_max, A->data[0]));
    return true;
}
static bool op_ReduceMin(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A || mat_nelem(A) == 0) { s.error = "ReduceMin: missing input"; return false; }
    put_out(n, s, 0, reduce_all(A, comb_min, A->data[0]));
    return true;
}
static bool op_ReduceProd(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "ReduceProd: missing input"; return false; }
    put_out(n, s, 0, reduce_all(A, comb_prod, 1.0));
    return true;
}
static bool op_ReduceL2(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "ReduceL2: missing input"; return false; }
    int64_t n2 = mat_nelem(A);
    double acc = 0;
    for (int64_t i = 0; i < n2; ++i) acc += A->data[i] * A->data[i];
    matlab_mat *Y = mat_alloc(1, 1);
    Y->data[0] = std::sqrt(acc);
    put_out(n, s, 0, Y);
    return true;
}

/* BatchNormalization (inference): Y = γ · (X - mean) / sqrt(var + ε) + β.
 * For 2-D input we treat columns as the feature axis (matches our shipped
 * batchnorm_eval over a (sample, feature) layout). */
static bool op_BatchNormalization(const OnnxNode &n, ExecState &s) {
    matlab_mat *X = get_in(n, s, 0);
    matlab_mat *gamma = get_in(n, s, 1);
    matlab_mat *beta  = get_in(n, s, 2);
    matlab_mat *mean  = get_in(n, s, 3);
    matlab_mat *var   = get_in(n, s, 4);
    if (!X || !gamma || !beta || !mean || !var) { s.error = "BatchNormalization: missing input"; return false; }
    double eps = attr_f(n, "epsilon", 1e-5);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    for (int64_t i = 0; i < X->rows; ++i) {
        for (int64_t j = 0; j < X->cols; ++j) {
            double m = mean->data[j];
            double v = var->data[j];
            double g = gamma->data[j];
            double b = beta->data[j];
            double xh = (X->data[i * X->cols + j] - m) / std::sqrt(v + eps);
            Y->data[i * X->cols + j] = g * xh + b;
        }
    }
    put_out(n, s, 0, Y);
    return true;
}

/* LayerNormalization: per-row normalisation with scale + bias. */
static bool op_LayerNormalization(const OnnxNode &n, ExecState &s) {
    matlab_mat *X = get_in(n, s, 0);
    matlab_mat *scale = get_in(n, s, 1);
    matlab_mat *bias  = (n.input.size() > 2) ? get_in(n, s, 2) : nullptr;
    if (!X || !scale) { s.error = "LayerNormalization: missing input"; return false; }
    double eps = attr_f(n, "epsilon", 1e-5);
    matlab_mat *Y = mat_alloc(X->rows, X->cols);
    int64_t R = X->rows, C = X->cols;
    for (int64_t i = 0; i < R; ++i) {
        double m = 0; for (int64_t j = 0; j < C; ++j) m += X->data[i * C + j]; m /= static_cast<double>(C);
        double v = 0; for (int64_t j = 0; j < C; ++j) { double d = X->data[i * C + j] - m; v += d * d; } v /= static_cast<double>(C);
        double s2 = std::sqrt(v + eps);
        for (int64_t j = 0; j < C; ++j) {
            double xh = (X->data[i * C + j] - m) / s2;
            double y = scale->data[j] * xh;
            if (bias) y += bias->data[j];
            Y->data[i * C + j] = y;
        }
    }
    put_out(n, s, 0, Y);
    return true;
}

/* Cast: int / float promotion within our double-only storage is a no-op. */
static bool op_Cast(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Cast: missing input"; return false; }
    put_out(n, s, 0, mat_clone(A));
    return true;
}

/* Constant: output an initializer-like tensor from attribute "value". */
static bool op_Constant(const OnnxNode &n, ExecState &s) {
    auto *v = find_attr(n, "value");
    if (!v || v->type != OA_TENSOR) { s.error = "Constant: missing 'value' tensor attribute"; return false; }
    int64_t r = 1, c = 1;
    if (v->t.dims.size() == 1) { r = v->t.dims[0]; c = 1; }
    else if (v->t.dims.size() == 2) { r = v->t.dims[0]; c = v->t.dims[1]; }
    matlab_mat *Y = mat_alloc(r, c);
    size_t nn = static_cast<size_t>(r * c);
    for (size_t i = 0; i < nn && i < v->t.data.size(); ++i) Y->data[i] = v->t.data[i];
    put_out(n, s, 0, Y);
    return true;
}

/* Sigmoid composed: Swish = x * sigmoid(x). */
static bool op_Swish(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Swish: missing input"; return false; }
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) {
        double x = A->data[i];
        Y->data[i] = x / (1.0 + std::exp(-x));
    }
    put_out(n, s, 0, Y);
    return true;
}

/* Comparison ops emit 0/1 doubles. */
static bool op_Equal(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Equal: missing input"; return false; }
    matlab_mat *Y = bin_broadcast(A, B, BIN_SUB);
    int64_t nn = mat_nelem(Y);
    for (int64_t i = 0; i < nn; ++i) Y->data[i] = (Y->data[i] == 0.0) ? 1.0 : 0.0;
    put_out(n, s, 0, Y);
    return true;
}
static bool op_Greater(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Greater: missing input"; return false; }
    matlab_mat *Y = bin_broadcast(A, B, BIN_SUB);
    int64_t nn = mat_nelem(Y);
    for (int64_t i = 0; i < nn; ++i) Y->data[i] = (Y->data[i] > 0.0) ? 1.0 : 0.0;
    put_out(n, s, 0, Y);
    return true;
}
static bool op_Less(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Less: missing input"; return false; }
    matlab_mat *Y = bin_broadcast(A, B, BIN_SUB);
    int64_t nn = mat_nelem(Y);
    for (int64_t i = 0; i < nn; ++i) Y->data[i] = (Y->data[i] < 0.0) ? 1.0 : 0.0;
    put_out(n, s, 0, Y);
    return true;
}

/* Pow / Sqrt as op classes. */
static bool op_Pow(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0), *B = get_in(n, s, 1);
    if (!A || !B) { s.error = "Pow: missing input"; return false; }
    matlab_mat *Y = bin_broadcast(A, B, BIN_ADD); /* shape-broadcast helper */
    int64_t nn = mat_nelem(Y);
    int64_t Ar = A->rows, Ac = A->cols, Br = B->rows, Bc = B->cols;
    int64_t Or = Y->rows, Oc = Y->cols;
    for (int64_t i = 0; i < Or; ++i) {
        for (int64_t j = 0; j < Oc; ++j) {
            int64_t ai = (Ar == 1) ? 0 : i;
            int64_t aj = (Ac == 1) ? 0 : j;
            int64_t bi = (Br == 1) ? 0 : i;
            int64_t bj = (Bc == 1) ? 0 : j;
            Y->data[i * Oc + j] = std::pow(A->data[ai * Ac + aj], B->data[bi * Bc + bj]);
        }
    }
    (void)nn;
    put_out(n, s, 0, Y);
    return true;
}

/* Clip: Y = clamp(X, min, max).  ONNX-11+ takes min/max as inputs (1/2);
 * older as attributes.  We support both. */
static bool op_Clip(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Clip: missing input"; return false; }
    double lo = -1e308, hi = 1e308;
    matlab_mat *Lm = (n.input.size() > 1) ? get_in(n, s, 1) : nullptr;
    matlab_mat *Hm = (n.input.size() > 2) ? get_in(n, s, 2) : nullptr;
    if (Lm && mat_nelem(Lm) > 0) lo = Lm->data[0];
    if (Hm && mat_nelem(Hm) > 0) hi = Hm->data[0];
    auto *am = find_attr(n, "min"); auto *aM = find_attr(n, "max");
    if (am) lo = am->f;
    if (aM) hi = aM->f;
    matlab_mat *Y = mat_alloc(A->rows, A->cols);
    int64_t nn = mat_nelem(A);
    for (int64_t i = 0; i < nn; ++i) {
        double x = A->data[i];
        if (x < lo) x = lo; else if (x > hi) x = hi;
        Y->data[i] = x;
    }
    put_out(n, s, 0, Y);
    return true;
}

/* MaxPool / AveragePool (2-D).  Input is treated as (H, W) — single-channel,
 * single-sample case for the matrix lane.  4-D NCHW input is rejected with
 * an explanatory error (rank-N tensor carve-down). */
static matlab_mat *pool2d(matlab_mat *A, int64_t kH, int64_t kW,
                          int64_t sH, int64_t sW, int64_t pH, int64_t pW, bool is_max) {
    int64_t H = A->rows, W = A->cols;
    int64_t Ho = (H + 2 * pH - kH) / sH + 1;
    int64_t Wo = (W + 2 * pW - kW) / sW + 1;
    matlab_mat *Y = mat_alloc(Ho, Wo);
    for (int64_t oi = 0; oi < Ho; ++oi) {
        for (int64_t oj = 0; oj < Wo; ++oj) {
            double acc = is_max ? -1e308 : 0.0;
            int64_t cnt = 0;
            for (int64_t ki = 0; ki < kH; ++ki) {
                for (int64_t kj = 0; kj < kW; ++kj) {
                    int64_t ii = oi * sH + ki - pH;
                    int64_t jj = oj * sW + kj - pW;
                    if (ii < 0 || ii >= H || jj < 0 || jj >= W) continue;
                    double v = A->data[ii * W + jj];
                    if (is_max) { if (v > acc) acc = v; }
                    else { acc += v; cnt++; }
                }
            }
            if (!is_max) acc = (cnt > 0) ? acc / static_cast<double>(cnt) : 0.0;
            Y->data[oi * Wo + oj] = acc;
        }
    }
    return Y;
}
static bool op_MaxPool(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "MaxPool: missing input"; return false; }
    auto ks = attr_ints(n, "kernel_shape");
    auto ss = attr_ints(n, "strides");
    auto pp = attr_ints(n, "pads");
    int64_t kH = ks.size() >= 1 ? ks[0] : 2, kW = ks.size() >= 2 ? ks[1] : kH;
    int64_t sH = ss.size() >= 1 ? ss[0] : 1, sW = ss.size() >= 2 ? ss[1] : sH;
    int64_t pH = pp.size() >= 1 ? pp[0] : 0, pW = pp.size() >= 2 ? pp[1] : 0;
    put_out(n, s, 0, pool2d(A, kH, kW, sH, sW, pH, pW, /*is_max=*/true));
    return true;
}
static bool op_AveragePool(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "AveragePool: missing input"; return false; }
    auto ks = attr_ints(n, "kernel_shape");
    auto ss = attr_ints(n, "strides");
    auto pp = attr_ints(n, "pads");
    int64_t kH = ks.size() >= 1 ? ks[0] : 2, kW = ks.size() >= 2 ? ks[1] : kH;
    int64_t sH = ss.size() >= 1 ? ss[0] : 1, sW = ss.size() >= 2 ? ss[1] : sH;
    int64_t pH = pp.size() >= 1 ? pp[0] : 0, pW = pp.size() >= 2 ? pp[1] : 0;
    put_out(n, s, 0, pool2d(A, kH, kW, sH, sW, pH, pW, /*is_max=*/false));
    return true;
}
static bool op_GlobalAveragePool(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "GlobalAveragePool: missing input"; return false; }
    int64_t nn = mat_nelem(A);
    double acc = 0;
    for (int64_t i = 0; i < nn; ++i) acc += A->data[i];
    matlab_mat *Y = mat_alloc(1, 1);
    Y->data[0] = (nn > 0) ? acc / static_cast<double>(nn) : 0.0;
    put_out(n, s, 0, Y);
    return true;
}
static bool op_GlobalMaxPool(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A || mat_nelem(A) == 0) { s.error = "GlobalMaxPool: missing input"; return false; }
    int64_t nn = mat_nelem(A);
    double acc = A->data[0];
    for (int64_t i = 1; i < nn; ++i) if (A->data[i] > acc) acc = A->data[i];
    matlab_mat *Y = mat_alloc(1, 1);
    Y->data[0] = acc;
    put_out(n, s, 0, Y);
    return true;
}

/* Conv (single-channel 2-D, ONNX-conformant valid-by-default semantics).
 * Treats X as (H, W) and W as (kH, kW).  Supports pads/strides/dilations
 * attributes — without them, output is (H - kH + 1) x (W - kW + 1).
 * 4-D NCHW conv is the proper ONNX surface — that needs rank-N tensors
 * and is carved with the rest of the SSCB lane. */
static bool op_Conv(const OnnxNode &n, ExecState &s) {
    matlab_mat *X = get_in(n, s, 0), *Wk = get_in(n, s, 1);
    matlab_mat *B = (n.input.size() > 2) ? get_in(n, s, 2) : nullptr;
    if (!X || !Wk) { s.error = "Conv: missing input"; return false; }
    auto pads = attr_ints(n, "pads");
    auto strides = attr_ints(n, "strides");
    auto dils = attr_ints(n, "dilations");
    int64_t pT = pads.size() >= 1 ? pads[0] : 0;
    int64_t pL = pads.size() >= 2 ? pads[1] : 0;
    int64_t pB = pads.size() >= 3 ? pads[2] : pT;
    int64_t pR = pads.size() >= 4 ? pads[3] : pL;
    int64_t sH = strides.size() >= 1 ? strides[0] : 1;
    int64_t sW = strides.size() >= 2 ? strides[1] : 1;
    int64_t dH = dils.size() >= 1 ? dils[0] : 1;
    int64_t dW = dils.size() >= 2 ? dils[1] : 1;
    int64_t H = X->rows, W = X->cols;
    int64_t kH = Wk->rows, kW = Wk->cols;
    int64_t effH = (kH - 1) * dH + 1;
    int64_t effW = (kW - 1) * dW + 1;
    int64_t Ho = (H + pT + pB - effH) / sH + 1;
    int64_t Wo = (W + pL + pR - effW) / sW + 1;
    if (Ho <= 0 || Wo <= 0) { s.error = "Conv: invalid output shape"; return false; }
    matlab_mat *Y = mat_alloc(Ho, Wo);
    double bias = (B && mat_nelem(B) >= 1) ? B->data[0] : 0.0;
    for (int64_t oi = 0; oi < Ho; ++oi) {
        for (int64_t oj = 0; oj < Wo; ++oj) {
            double acc = 0.0;
            for (int64_t ki = 0; ki < kH; ++ki) {
                for (int64_t kj = 0; kj < kW; ++kj) {
                    int64_t ii = oi * sH + ki * dH - pT;
                    int64_t jj = oj * sW + kj * dW - pL;
                    if (ii < 0 || ii >= H || jj < 0 || jj >= W) continue;
                    acc += X->data[ii * W + jj] * Wk->data[ki * kW + kj];
                }
            }
            Y->data[oi * Wo + oj] = acc + bias;
        }
    }
    put_out(n, s, 0, Y);
    return true;
}

/* Gather: row-/column-select.  Input 0 = data (2-D), input 1 = indices.
 * axis attribute (default 0). */
static bool op_Gather(const OnnxNode &n, ExecState &s) {
    matlab_mat *D = get_in(n, s, 0), *I = get_in(n, s, 1);
    if (!D || !I) { s.error = "Gather: missing input"; return false; }
    int64_t axis = attr_i(n, "axis", 0);
    int64_t k = mat_nelem(I);
    matlab_mat *Y;
    if (axis == 0) {
        Y = mat_alloc(k, D->cols);
        for (int64_t i = 0; i < k; ++i) {
            int64_t r = static_cast<int64_t>(I->data[i]);
            if (r < 0 || r >= D->rows) continue;
            for (int64_t j = 0; j < D->cols; ++j) Y->data[i * D->cols + j] = D->data[r * D->cols + j];
        }
    } else {
        Y = mat_alloc(D->rows, k);
        for (int64_t j = 0; j < k; ++j) {
            int64_t c = static_cast<int64_t>(I->data[j]);
            if (c < 0 || c >= D->cols) continue;
            for (int64_t i = 0; i < D->rows; ++i) Y->data[i * k + j] = D->data[i * D->cols + c];
        }
    }
    put_out(n, s, 0, Y);
    return true;
}

/* Shape: emit dims as int64 row vector (we keep them as doubles). */
static bool op_Shape(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Shape: missing input"; return false; }
    matlab_mat *Y = mat_alloc(2, 1);
    Y->data[0] = static_cast<double>(A->rows);
    Y->data[1] = static_cast<double>(A->cols);
    put_out(n, s, 0, Y);
    return true;
}

/* Size: total numel. */
static bool op_Size(const OnnxNode &n, ExecState &s) {
    matlab_mat *A = get_in(n, s, 0);
    if (!A) { s.error = "Size: missing input"; return false; }
    matlab_mat *Y = mat_alloc(1, 1);
    Y->data[0] = static_cast<double>(mat_nelem(A));
    put_out(n, s, 0, Y);
    return true;
}

/* Dispatcher table. */
typedef bool (*OpFn)(const OnnxNode &, ExecState &);
static const std::map<std::string, OpFn> &op_table() {
    static const std::map<std::string, OpFn> t = {
        {"Identity",     op_Identity},
        {"MatMul",       op_MatMul},
        {"Gemm",         op_Gemm},
        {"Add",          op_Add},
        {"Sub",          op_Sub},
        {"Mul",          op_Mul},
        {"Div",          op_Div},
        {"Relu",         op_Relu},
        {"Sigmoid",      op_Sigmoid},
        {"Tanh",         op_Tanh},
        {"Exp",          op_Exp},
        {"Log",          op_Log},
        {"Sqrt",         op_Sqrt},
        {"Abs",          op_Abs},
        {"Neg",          op_Neg},
        {"Floor",        op_Floor},
        {"Ceil",         op_Ceil},
        {"Round",        op_Round},
        {"Softplus",     op_Softplus},
        {"Softsign",     op_Softsign},
        {"Reciprocal",   op_Reciprocal},
        {"LeakyRelu",    op_LeakyRelu},
        {"Elu",          op_Elu},
        {"Selu",         op_Selu},
        {"Gelu",         op_Gelu},
        {"HardSigmoid",  op_HardSigmoid},
        {"HardSwish",    op_HardSwish},
        {"Softmax",      op_Softmax},
        {"LogSoftmax",   op_LogSoftmax},
        {"Reshape",      op_Reshape},
        {"Flatten",      op_Flatten},
        {"Transpose",    op_Transpose},
        {"Concat",       op_Concat},
        {"ReduceSum",    op_ReduceSum},
        {"ReduceMean",   op_ReduceMean},
        {"ReduceMax",    op_ReduceMax},
        {"ReduceMin",    op_ReduceMin},
        {"ReduceProd",   op_ReduceProd},
        {"ReduceL2",     op_ReduceL2},
        {"BatchNormalization", op_BatchNormalization},
        {"LayerNormalization", op_LayerNormalization},
        {"Cast",         op_Cast},
        {"Constant",     op_Constant},
        {"Swish",        op_Swish},
        {"Equal",        op_Equal},
        {"Greater",      op_Greater},
        {"Less",         op_Less},
        {"Pow",          op_Pow},
        {"Clip",         op_Clip},
        {"MaxPool",      op_MaxPool},
        {"AveragePool",  op_AveragePool},
        {"GlobalAveragePool", op_GlobalAveragePool},
        {"GlobalMaxPool", op_GlobalMaxPool},
        {"Conv",         op_Conv},
        {"Gather",       op_Gather},
        {"Shape",        op_Shape},
        {"Size",         op_Size},
    };
    return t;
}

/* =====================================================================
 * Global state — single loaded model + single in-progress builder.
 * =================================================================== */

static OnnxModel g_loaded;
static OnnxModel g_build;
static OnnxNode g_build_node;   /* current node being assembled */
static bool g_build_node_open = false;

/* Convert OnnxTensor.data + dims into a matlab_mat (2-D).  For rank-1 we
 * store as a Nx1 column.  Rank-2 = exact.  Other ranks fall back to
 * flatten (preserved as 1xN for tracking). */
static matlab_mat *tensor_to_mat(const OnnxTensor &t) {
    int64_t r = 1, c = 1;
    if (t.dims.size() == 1) { r = t.dims[0]; c = 1; }
    else if (t.dims.size() == 2) { r = t.dims[0]; c = t.dims[1]; }
    else if (!t.dims.empty()) {
        int64_t flat = 1;
        for (auto d : t.dims) flat *= d;
        r = 1; c = flat;
    }
    matlab_mat *m = mat_alloc(r, c);
    size_t nn = static_cast<size_t>(r * c);
    for (size_t i = 0; i < nn && i < t.data.size(); ++i) m->data[i] = t.data[i];
    return m;
}

}  // anonymous namespace

/* =====================================================================
 * extern "C" entry points (called from the lowered LLVM IR).
 * =================================================================== */

extern "C" {

/* ---- Reader / runner ---------------------------------------------- */

matlab_mat *matlab_onnx_read(void *path_s) {
    std::string path = read_str_arg(path_s);
    matlab_mat *handle = mat_alloc(1, 1);
    handle->data[0] = 0.0;
    if (path.empty()) return handle;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return handle;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
    PbReader r;
    r.cur = bytes.data();
    r.end = bytes.data() + bytes.size();
    g_loaded = OnnxModel();
    decode_model(r, g_loaded);
    if (!r.ok) return handle;
    handle->data[0] = 1.0;
    return handle;
}

double matlab_onnx_num_nodes(matlab_mat *) {
    return static_cast<double>(g_loaded.graph.node.size());
}
double matlab_onnx_num_inits(matlab_mat *) {
    return static_cast<double>(g_loaded.graph.initializer.size());
}
double matlab_onnx_opset(matlab_mat *) {
    return static_cast<double>(g_loaded.opset_version);
}

/* onnxRun(handle, X) — feeds X into the first graph input and runs the
 * graph in node order; returns the first output. */
matlab_mat *matlab_onnx_run(matlab_mat *, matlab_mat *X) {
    ExecState st;
    /* Pre-populate initializers. */
    for (auto &t : g_loaded.graph.initializer) {
        st.tensors[t.name] = tensor_to_mat(t);
    }
    /* Map X into the first input. */
    if (!g_loaded.graph.input.empty()) {
        st.tensors[g_loaded.graph.input[0].name] = mat_clone(X);
    }
    /* Execute nodes in order. */
    for (auto &n : g_loaded.graph.node) {
        auto &t = op_table();
        auto it = t.find(n.op_type);
        if (it == t.end()) {
            st.error = "unsupported op: " + n.op_type;
            break;
        }
        if (!it->second(n, st)) break;
    }
    if (!st.ok()) {
        matlab_mat *err = mat_alloc(0, 0);
        return err;
    }
    /* Return the first output. */
    if (!g_loaded.graph.output.empty()) {
        auto it = st.tensors.find(g_loaded.graph.output[0].name);
        if (it != st.tensors.end()) return mat_clone(it->second);
    }
    return mat_alloc(0, 0);
}

/* ---- Programmatic builder ----------------------------------------- */

matlab_mat *matlab_onnx_new_model(void) {
    g_build = OnnxModel();
    g_build_node = OnnxNode();
    g_build_node_open = false;
    return mat_alloc(0, 0);
}

matlab_mat *matlab_onnx_add_init(void *name_s, matlab_mat *T) {
    OnnxTensor t;
    t.name = read_str_arg(name_s);
    t.data_type = ODT_FLOAT;
    if (T) {
        t.dims.push_back(T->rows);
        t.dims.push_back(T->cols);
        int64_t nn = T->rows * T->cols;
        t.data.assign(T->data, T->data + nn);
    }
    g_build.graph.initializer.push_back(std::move(t));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_onnx_set_input(void *name_s, matlab_mat *dims) {
    OnnxValueInfo v;
    v.name = read_str_arg(name_s);
    if (dims) {
        int64_t nn = mat_nelem(dims);
        for (int64_t i = 0; i < nn; ++i) v.dims.push_back(static_cast<int64_t>(dims->data[i]));
    }
    g_build.graph.input.push_back(std::move(v));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_onnx_set_output(void *name_s) {
    OnnxValueInfo v;
    v.name = read_str_arg(name_s);
    g_build.graph.output.push_back(std::move(v));
    return mat_alloc(0, 0);
}

matlab_mat *matlab_onnx_begin_node(void *op_s) {
    g_build_node = OnnxNode();
    g_build_node.op_type = read_str_arg(op_s);
    g_build_node_open = true;
    return mat_alloc(0, 0);
}
matlab_mat *matlab_onnx_node_input(void *name_s) {
    if (g_build_node_open) g_build_node.input.push_back(read_str_arg(name_s));
    return mat_alloc(0, 0);
}
matlab_mat *matlab_onnx_node_output(void *name_s) {
    if (g_build_node_open) g_build_node.output.push_back(read_str_arg(name_s));
    return mat_alloc(0, 0);
}
matlab_mat *matlab_onnx_node_attr_int(void *name_s, double v) {
    if (g_build_node_open) {
        OnnxAttribute a;
        a.name = read_str_arg(name_s);
        a.type = OA_INT;
        a.i = static_cast<int64_t>(v);
        g_build_node.attribute.push_back(std::move(a));
    }
    return mat_alloc(0, 0);
}
matlab_mat *matlab_onnx_node_attr_float(void *name_s, double v) {
    if (g_build_node_open) {
        OnnxAttribute a;
        a.name = read_str_arg(name_s);
        a.type = OA_FLOAT;
        a.f = v;
        g_build_node.attribute.push_back(std::move(a));
    }
    return mat_alloc(0, 0);
}
matlab_mat *matlab_onnx_node_attr_ints(void *name_s, matlab_mat *ints_m) {
    if (g_build_node_open) {
        OnnxAttribute a;
        a.name = read_str_arg(name_s);
        a.type = OA_INTS;
        if (ints_m) {
            int64_t nn = mat_nelem(ints_m);
            for (int64_t i = 0; i < nn; ++i) a.ints.push_back(static_cast<int64_t>(ints_m->data[i]));
        }
        g_build_node.attribute.push_back(std::move(a));
    }
    return mat_alloc(0, 0);
}
matlab_mat *matlab_onnx_end_node(void) {
    if (g_build_node_open) {
        g_build.graph.node.push_back(std::move(g_build_node));
        g_build_node_open = false;
    }
    return mat_alloc(0, 0);
}

matlab_mat *matlab_onnx_save(void *path_s) {
    std::string path = read_str_arg(path_s);
    matlab_mat *out = mat_alloc(1, 1);
    out->data[0] = 0.0;
    if (path.empty()) return out;
    PbWriter w;
    encode_model(w, g_build);
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return out;
    f.write(reinterpret_cast<const char *>(w.buf.data()), static_cast<std::streamsize>(w.buf.size()));
    f.close();
    out->data[0] = 1.0;
    return out;
}

}  // extern "C"
