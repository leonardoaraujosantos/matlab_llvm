/* ============================================================================
 * runtime_images.cpp — Image Processing Toolbox runtime (Tiers 1-2)
 * ----------------------------------------------------------------------------
 * Tier-1: image file I/O (PGM/PPM/BMP) + a synthetic generator, type
 * conversions, image arithmetic, histogram/intensity statistics.
 * Tier-2: spatial filtering (fspecial / imgaussfilt / imboxfilt / medfilt2 /
 * ordfilt2 / stdfilt / rangefilt) and enhancement (histeq / adapthisteq /
 * imsharpen / imhistmatch / imnoise).
 *
 * Representation (documented simplification): images are plain `double`
 * matrices — M×N grayscale (`matlab_mat`) or M×N×3 truecolor
 * (`matlab_mat3`, slice-major), values in [0,255] for uint8-class images
 * or [0,1] for double-class.  This reuses the entire shipped double kernel
 * (conv2 / imfilter / padarray); the uint8 *class* tag is approximated by
 * value range + saturation in the arithmetic ops.  No external dependency.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

/* shipped helpers reused. */
extern "C" matlab_mat *matlab_imfilter(matlab_mat *A, matlab_mat *h);
extern "C" matlab_mat *matlab_conv2(matlab_mat *A, matlab_mat *h);
extern "C" matlab_mat *matlab_rand(double m, double n);
extern "C" matlab_mat *matlab_randn(double m, double n);
extern "C" matlab_mat *matlab_stats_kmeans(matlab_mat *X, matlab_mat *kk);  /* imsegkmeans */
extern "C" matlab_mat_c *matlab_fft2_c(void *Aptr);                          /* deconvwnr */
extern "C" matlab_mat_c *matlab_ifft2_c(void *Aptr);

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
struct img_string_s { char *data; int64_t len; };

/* ===== file-scope helpers ================================================ */

static std::string img_sstr(const void *s) {
    if (!s) return std::string();
    const img_string_s *p = reinterpret_cast<const img_string_s *>(s);
    if (!p->data || p->len <= 0) return std::string();
    return std::string(p->data, p->data + p->len);
}
static double img_sc(const matlab_mat *m, double dflt) {
    return (m && m->data && m->rows * m->cols > 0) ? m->data[0] : dflt;
}
static double img_clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
/* Does the data look like a uint8-range image (max > 1)? */
static bool img_is_u8range(const double *d, int64_t n) {
    for (int64_t i = 0; i < n; ++i) if (d[i] > 1.0) return true;
    return false;
}
static std::string img_ext_lower(const std::string &path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return std::string();
    std::string e = path.substr(dot + 1);
    for (char &c : e) c = static_cast<char>(tolower(c));
    return e;
}

/* ----- PNM (PGM/PPM) parsing ----------------------------------------------*/
static int pnm_next(FILE *f) {           /* next token int, skipping ws/comments */
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while ((c = fgetc(f)) != '\n' && c != EOF) {} continue; }
        if (c == EOF) return -1;
        if (!isspace(c)) break;
    }
    int v = 0;
    while (c != EOF && !isspace(c)) { v = v * 10 + (c - '0'); c = fgetc(f); }
    return v;
}

/* ===== PNG decoder (hand-coded; zlib inflate + chunk parse + unfilter) ===
 * No libpng/zlib dependency.  inflate follows Mark Adler's puff.c reference
 * structure (canonical Huffman, LSB-first bit reader).  Supports the common
 * non-interlaced colour types (grayscale, RGB, palette, gray+alpha, RGBA)
 * at bit depths 1/2/4/8/16.  Returns a matlab_mat (grayscale) or matlab_mat3
 * (truecolor), values in [0,255]; alpha is dropped, 16-bit is scaled down. */

struct PngBits {
    const unsigned char *in; size_t inlen; size_t incnt;
    int bitbuf; int bitcnt;
};
static int png_bits(PngBits &s, int need) {
    long val = s.bitbuf;
    while (s.bitcnt < need) {
        if (s.incnt >= s.inlen) return -1;
        val |= static_cast<long>(s.in[s.incnt++]) << s.bitcnt;
        s.bitcnt += 8;
    }
    s.bitbuf = static_cast<int>(val >> need);
    s.bitcnt -= need;
    return static_cast<int>(val & ((1L << need) - 1));
}
struct PngHuff { std::vector<short> count; std::vector<short> symbol; };
static void png_construct(PngHuff &h, const std::vector<short> &length, int n) {
    h.count.assign(16, 0);
    h.symbol.assign(static_cast<size_t>(n), 0);
    for (int sym = 0; sym < n; ++sym) h.count[static_cast<size_t>(length[static_cast<size_t>(sym)])]++;
    std::vector<short> offs(16, 0);
    for (int len = 1; len < 15; ++len) offs[static_cast<size_t>(len + 1)] = static_cast<short>(offs[static_cast<size_t>(len)] + h.count[static_cast<size_t>(len)]);
    for (int sym = 0; sym < n; ++sym)
        if (length[static_cast<size_t>(sym)] != 0)
            h.symbol[static_cast<size_t>(offs[static_cast<size_t>(length[static_cast<size_t>(sym)])]++)] = static_cast<short>(sym);
}
static int png_decode_sym(PngBits &s, const PngHuff &h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= 15; ++len) {
        int b = png_bits(s, 1); if (b < 0) return -1;
        code |= b;
        int cnt = h.count[static_cast<size_t>(len)];
        if (code - cnt < first) return h.symbol[static_cast<size_t>(index + (code - first))];
        index += cnt; first += cnt; first <<= 1; code <<= 1;
    }
    return -1;
}
static bool png_codes(PngBits &s, const PngHuff &lc, const PngHuff &dc, std::vector<unsigned char> &out) {
    static const short lbase[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static const short lext[29]  = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static const short dbase[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    static const short dext[30]  = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    for (;;) {
        int sym = png_decode_sym(s, lc); if (sym < 0) return false;
        if (sym == 256) break;
        if (sym < 256) { out.push_back(static_cast<unsigned char>(sym)); continue; }
        sym -= 257; if (sym >= 29) return false;
        int eb = png_bits(s, lext[sym]); if (eb < 0) return false;
        int length = lbase[sym] + eb;
        int dsym = png_decode_sym(s, dc); if (dsym < 0 || dsym >= 30) return false;
        int de = png_bits(s, dext[dsym]); if (de < 0) return false;
        int dist = dbase[dsym] + de;
        if (static_cast<size_t>(dist) > out.size()) return false;
        size_t from = out.size() - static_cast<size_t>(dist);
        for (int k = 0; k < length; ++k) out.push_back(out[from + static_cast<size_t>(k)]);
    }
    return true;
}
static bool png_inflate(const unsigned char *src, size_t srclen, std::vector<unsigned char> &out) {
    PngBits s{src, srclen, 0, 0, 0};
    int last;
    do {
        last = png_bits(s, 1); if (last < 0) return false;
        int type = png_bits(s, 2); if (type < 0) return false;
        if (type == 0) {                                   /* stored */
            s.bitbuf = 0; s.bitcnt = 0;
            if (s.incnt + 4 > s.inlen) return false;
            int len = s.in[s.incnt] | (s.in[s.incnt + 1] << 8); s.incnt += 4;
            if (s.incnt + static_cast<size_t>(len) > s.inlen) return false;
            for (int k = 0; k < len; ++k) out.push_back(s.in[s.incnt++]);
        } else if (type == 1 || type == 2) {
            PngHuff lc, dc;
            if (type == 1) {                               /* fixed Huffman */
                std::vector<short> ll(288), dl(30);
                int i = 0;
                for (; i < 144; ++i) ll[static_cast<size_t>(i)] = 8;
                for (; i < 256; ++i) ll[static_cast<size_t>(i)] = 9;
                for (; i < 280; ++i) ll[static_cast<size_t>(i)] = 7;
                for (; i < 288; ++i) ll[static_cast<size_t>(i)] = 8;
                for (int d = 0; d < 30; ++d) dl[static_cast<size_t>(d)] = 5;
                png_construct(lc, ll, 288); png_construct(dc, dl, 30);
            } else {                                        /* dynamic Huffman */
                int hlit = png_bits(s, 5) + 257, hdist = png_bits(s, 5) + 1, hclen = png_bits(s, 4) + 4;
                static const int ord[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
                std::vector<short> cl(19, 0);
                for (int i = 0; i < hclen; ++i) { int v = png_bits(s, 3); if (v < 0) return false; cl[static_cast<size_t>(ord[i])] = static_cast<short>(v); }
                PngHuff clh; png_construct(clh, cl, 19);
                std::vector<short> lengths(static_cast<size_t>(hlit + hdist), 0);
                int idx = 0;
                while (idx < hlit + hdist) {
                    int sym = png_decode_sym(s, clh); if (sym < 0) return false;
                    if (sym < 16) lengths[static_cast<size_t>(idx++)] = static_cast<short>(sym);
                    else if (sym == 16) { if (idx == 0) return false; int r = png_bits(s, 2) + 3; short pv = lengths[static_cast<size_t>(idx - 1)]; while (r-- && idx < hlit + hdist) lengths[static_cast<size_t>(idx++)] = pv; }
                    else if (sym == 17) { int r = png_bits(s, 3) + 3; while (r-- && idx < hlit + hdist) lengths[static_cast<size_t>(idx++)] = 0; }
                    else { int r = png_bits(s, 7) + 11; while (r-- && idx < hlit + hdist) lengths[static_cast<size_t>(idx++)] = 0; }
                }
                std::vector<short> ll(lengths.begin(), lengths.begin() + hlit);
                std::vector<short> dl(lengths.begin() + hlit, lengths.end());
                png_construct(lc, ll, hlit); png_construct(dc, dl, hdist);
            }
            if (!png_codes(s, lc, dc, out)) return false;
        } else return false;
    } while (!last);
    return true;
}
static unsigned char png_paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    if (pa <= pb && pa <= pc) return static_cast<unsigned char>(a);
    if (pb <= pc) return static_cast<unsigned char>(b);
    return static_cast<unsigned char>(c);
}
static uint32_t png_be32(const unsigned char *p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
/* decode a whole PNG file buffer -> matlab_mat (gray) or matlab_mat3 (RGB). */
static matlab_mat *png_decode(const unsigned char *f, size_t n) {
    static const unsigned char sig[8] = {137,80,78,71,13,10,26,10};
    if (n < 8 || memcmp(f, sig, 8) != 0) return nullptr;
    size_t p = 8;
    int W = 0, H = 0, depth = 0, ctype = 0, interlace = 0;
    std::vector<unsigned char> idat, plte;
    while (p + 8 <= n) {
        uint32_t clen = png_be32(f + p);
        const unsigned char *ctyp = f + p + 4;
        const unsigned char *cdat = f + p + 8;
        if (p + 12 + clen > n) break;
        if (memcmp(ctyp, "IHDR", 4) == 0) {
            W = static_cast<int>(png_be32(cdat)); H = static_cast<int>(png_be32(cdat + 4));
            depth = cdat[8]; ctype = cdat[9]; interlace = cdat[12];
        } else if (memcmp(ctyp, "PLTE", 4) == 0) {
            plte.assign(cdat, cdat + clen);
        } else if (memcmp(ctyp, "IDAT", 4) == 0) {
            idat.insert(idat.end(), cdat, cdat + clen);
        } else if (memcmp(ctyp, "IEND", 4) == 0) break;
        p += 12 + clen;                                    /* len+type+data+crc */
    }
    if (W <= 0 || H <= 0 || interlace != 0 || idat.size() < 3) return nullptr;
    int channels = (ctype == 0) ? 1 : (ctype == 2) ? 3 : (ctype == 3) ? 1 : (ctype == 4) ? 2 : (ctype == 6) ? 4 : 0;
    if (channels == 0) return nullptr;
    /* zlib stream: skip 2-byte header, inflate the rest (ignore adler32). */
    std::vector<unsigned char> raw;
    if (!png_inflate(idat.data() + 2, idat.size() - 2, raw)) return nullptr;
    int bpp = (channels * depth + 7) / 8; if (bpp < 1) bpp = 1;
    int rowbytes = (W * channels * depth + 7) / 8;
    if (static_cast<size_t>((rowbytes + 1) * H) > raw.size()) return nullptr;
    /* unfilter in place into a contiguous H×rowbytes buffer. */
    std::vector<unsigned char> img(static_cast<size_t>(rowbytes) * H);
    for (int y = 0; y < H; ++y) {
        const unsigned char *src = raw.data() + static_cast<size_t>(y) * (rowbytes + 1);
        int ft = src[0]; const unsigned char *line = src + 1;
        unsigned char *cur = img.data() + static_cast<size_t>(y) * rowbytes;
        unsigned char *prev = (y > 0) ? img.data() + static_cast<size_t>(y - 1) * rowbytes : nullptr;
        for (int x = 0; x < rowbytes; ++x) {
            int a = (x >= bpp) ? cur[x - bpp] : 0;
            int b = prev ? prev[x] : 0;
            int c = (prev && x >= bpp) ? prev[x - bpp] : 0;
            int v = line[x];
            if (ft == 1) v += a; else if (ft == 2) v += b; else if (ft == 3) v += (a + b) / 2; else if (ft == 4) v += png_paeth(a, b, c);
            cur[x] = static_cast<unsigned char>(v & 0xff);
        }
    }
    /* sample a channel value at pixel (y,x), scaled to [0,255]. */
    auto sample = [&](int y, int x, int ch) -> double {
        const unsigned char *row = img.data() + static_cast<size_t>(y) * rowbytes;
        if (depth == 8) return row[x * channels + ch];
        if (depth == 16) return row[(x * channels + ch) * 2];        /* high byte */
        int bitpos = (x * channels + ch) * depth;                    /* 1/2/4-bit */
        int byte = row[bitpos / 8];
        int shift = 8 - depth - (bitpos % 8);
        int val = (byte >> shift) & ((1 << depth) - 1);
        return (ctype == 3) ? val : val * 255.0 / ((1 << depth) - 1);
    };
    bool gray = (ctype == 0 || ctype == 4);
    if (gray) {
        matlab_mat *R = mat_alloc(H, W);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) R->data[y * W + x] = sample(y, x, 0);
        return R;
    }
    /* RGBA (ctype 6) keeps its alpha as a 4th channel (depth-4); RGB and
     * palette stay depth-3.  Downstream ops are depth-aware (Tier A3). */
    int outch = (ctype == 6) ? 4 : 3;
    matlab_mat3 *R = mat3_alloc(H, W, outch);
    int64_t pl = static_cast<int64_t>(H) * W;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        if (ctype == 3) {                                  /* palette index -> RGB */
            int ix = static_cast<int>(sample(y, x, 0));
            for (int cc = 0; cc < 3; ++cc) R->data[cc * pl + y * W + x] = (static_cast<size_t>(ix * 3 + cc) < plte.size()) ? plte[static_cast<size_t>(ix * 3 + cc)] : 0;
        } else {                                            /* RGB / RGBA: all channels */
            for (int cc = 0; cc < outch; ++cc) R->data[cc * pl + y * W + x] = sample(y, x, cc);
        }
    }
    return reinterpret_cast<matlab_mat *>(R);
}

/* ----- minimal PNG encoder (store-mode deflate; lossless) -------------- */
static uint32_t png_crc(const unsigned char *d, size_t n, uint32_t crc) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) { crc ^= d[i]; for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u)); }
    return ~crc;
}
static void png_put32(std::vector<unsigned char> &v, uint32_t x) {
    v.push_back(static_cast<unsigned char>(x >> 24)); v.push_back(static_cast<unsigned char>(x >> 16));
    v.push_back(static_cast<unsigned char>(x >> 8)); v.push_back(static_cast<unsigned char>(x));
}
static void png_chunk(std::vector<unsigned char> &out, const char *type, const std::vector<unsigned char> &data) {
    png_put32(out, static_cast<uint32_t>(data.size()));
    std::vector<unsigned char> td(type, type + 4); td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    png_put32(out, png_crc(td.data(), td.size(), 0));
}
/* encode a matlab_mat (gray) or matlab_mat3 (RGB) to a PNG byte buffer. */
static std::vector<unsigned char> png_encode(void *img) {
    std::vector<unsigned char> out;
    bool is3d = mat_is_3d(img);
    int64_t H, W, pl = 0; const double *d; int ch;
    if (is3d) { matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(img); H = m->rows; W = m->cols; d = m->data; pl = H * W; ch = static_cast<int>(m->depth); }
    else      { matlab_mat *m = reinterpret_cast<matlab_mat *>(img);  H = m->rows; W = m->cols; d = m->data; ch = 1; }
    /* colour type: 0 gray, 4 gray+alpha, 2 RGB, 6 RGBA — by channel count. */
    int colortype = (ch == 4) ? 6 : (ch == 3) ? 2 : (ch == 2) ? 4 : 0;
    static const unsigned char sig[8] = {137,80,78,71,13,10,26,10};
    out.insert(out.end(), sig, sig + 8);
    std::vector<unsigned char> ihdr;
    png_put32(ihdr, static_cast<uint32_t>(W)); png_put32(ihdr, static_cast<uint32_t>(H));
    ihdr.push_back(8); ihdr.push_back(static_cast<unsigned char>(colortype)); ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    png_chunk(out, "IHDR", ihdr);
    /* raw filtered scanlines: filter byte 0 + ch bytes per pixel. */
    std::vector<unsigned char> raw;
    for (int64_t y = 0; y < H; ++y) {
        raw.push_back(0);
        for (int64_t x = 0; x < W; ++x) for (int c = 0; c < ch; ++c) {
            double v = is3d ? d[c * pl + y * W + x] : d[y * W + x];
            raw.push_back(static_cast<unsigned char>(img_clamp(v + 0.5, 0, 255)));
        }
    }
    /* zlib stream: header + stored deflate blocks + adler32. */
    std::vector<unsigned char> z; z.push_back(0x78); z.push_back(0x01);
    size_t off = 0;
    while (off < raw.size()) {
        size_t blk = std::min<size_t>(65535, raw.size() - off);
        z.push_back((off + blk >= raw.size()) ? 1 : 0);     /* BFINAL, BTYPE=00 */
        z.push_back(static_cast<unsigned char>(blk)); z.push_back(static_cast<unsigned char>(blk >> 8));
        unsigned short nlen = static_cast<unsigned short>(~blk);
        z.push_back(static_cast<unsigned char>(nlen)); z.push_back(static_cast<unsigned char>(nlen >> 8));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + blk); off += blk;
    }
    uint32_t a = 1, b = 0; for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    png_put32(z, (b << 16) | a);
    png_chunk(out, "IDAT", z);
    png_chunk(out, "IEND", std::vector<unsigned char>());
    return out;
}

/* ===== baseline JPEG decoder (hand-coded; JFIF / SOF0 sequential) =======
 * No libjpeg dependency.  Supports baseline Huffman JPEG: DQT/SOF0/DHT/SOS,
 * 4:4:4 / 4:2:2 / 4:2:0 chroma subsampling, naive 8×8 float IDCT,
 * YCbCr→RGB.  Progressive / arithmetic coding are not supported. */
static const int JPG_ZZ[64] = {
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63};
struct JpgHuff {
    unsigned char bits[17]; unsigned char vals[256];
    int mincode[17], maxcode[18], valptr[17]; bool ok;
};
static void jpg_build(JpgHuff &h) {
    int code = 0, k = 0; std::vector<int> sizes;
    for (int l = 1; l <= 16; ++l) for (int i = 0; i < h.bits[l]; ++i) sizes.push_back(l);
    sizes.push_back(0);
    int p = 0;
    for (int l = 1; l <= 16; ++l) {
        if (h.bits[l] == 0) { h.maxcode[l] = -1; }
        else { h.valptr[l] = p; h.mincode[l] = code; code += h.bits[l]; h.maxcode[l] = code - 1; p += h.bits[l]; }
        code <<= 1;
    }
    h.maxcode[17] = 0x7fffffff; (void)k;
    h.ok = true;
}
struct JpgBits { const unsigned char *d; size_t len, pos; uint32_t buf; int cnt; int marker; };
static int jpg_bit(JpgBits &s) {
    if (s.cnt == 0) {
        if (s.pos >= s.len) { s.marker = -1; return 0; }
        unsigned char b = s.d[s.pos++];
        if (b == 0xFF) {
            unsigned char b2 = (s.pos < s.len) ? s.d[s.pos] : 0;
            if (b2 == 0) s.pos++;                 /* stuffed FF00 */
            else { s.marker = b2; return 0; }     /* hit a marker */
        }
        s.buf = b; s.cnt = 8;
    }
    s.cnt--;
    return (s.buf >> s.cnt) & 1;
}
static int jpg_bits(JpgBits &s, int n) { int v = 0; for (int i = 0; i < n; ++i) v = (v << 1) | jpg_bit(s); return v; }
static int jpg_recv_ext(JpgBits &s, int n) {      /* receive + extend (signed) */
    if (n == 0) return 0;
    int v = jpg_bits(s, n);
    if (v < (1 << (n - 1))) v += (-1 << n) + 1;
    return v;
}
static int jpg_decode(JpgBits &s, const JpgHuff &h) {
    int code = 0;
    for (int l = 1; l <= 16; ++l) {
        code = (code << 1) | jpg_bit(s);
        if (h.maxcode[l] >= 0 && code <= h.maxcode[l]) return h.vals[h.valptr[l] + (code - h.mincode[l])];
    }
    return 0;
}
static void jpg_idct(double *blk) {               /* in-place 8×8 float IDCT */
    static double C[8][8]; static bool init = false;
    if (!init) { for (int x = 0; x < 8; ++x) for (int u = 0; u < 8; ++u) C[x][u] = cos((2*x+1)*u*M_PI/16.0) * (u == 0 ? sqrt(0.5) : 1.0); init = true; }
    double tmp[64];
    for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) {       /* rows */
        double sum = 0; for (int u = 0; u < 8; ++u) sum += C[x][u] * blk[y*8+u];
        tmp[y*8+x] = sum * 0.5;
    }
    for (int x = 0; x < 8; ++x) for (int y = 0; y < 8; ++y) {       /* cols */
        double sum = 0; for (int v = 0; v < 8; ++v) sum += C[y][v] * tmp[v*8+x];
        blk[y*8+x] = sum * 0.5;
    }
}
static matlab_mat *jpg_decode_file(const unsigned char *f, size_t n) {
    if (n < 2 || f[0] != 0xFF || f[1] != 0xD8) return nullptr;
    size_t p = 2;
    int W = 0, H = 0, ncomp = 0, maxH = 1, maxV = 1;
    unsigned short qt[4][64] = {{0}};
    JpgHuff hdc[4]{}, hac[4]{};
    struct Comp { int id, hs, vs, qsel, dcsel, acsel, pred; } comp[4]{};
    while (p + 4 <= n) {
        if (f[p] != 0xFF) { p++; continue; }
        int m = f[p + 1]; p += 2;
        if (m == 0xD9) break;                          /* EOI */
        if (m == 0x01 || (m >= 0xD0 && m <= 0xD7)) continue;
        int seg = (f[p] << 8) | f[p + 1]; const unsigned char *d = f + p + 2; int dl = seg - 2;
        if (m == 0xDB) {                                /* DQT */
            int q = 0;
            while (q < dl) { int pq = d[q] >> 4, tq = d[q] & 15; q++;
                for (int i = 0; i < 64; ++i) { qt[tq][i] = pq ? ((d[q] << 8) | d[q+1]) : d[q]; q += pq ? 2 : 1; } }
        } else if (m == 0xC0 || m == 0xC1) {            /* SOF0/1 baseline */
            H = (d[1] << 8) | d[2]; W = (d[3] << 8) | d[4]; ncomp = d[5];
            for (int c = 0; c < ncomp; ++c) { comp[c].id = d[6+c*3]; comp[c].hs = d[7+c*3] >> 4; comp[c].vs = d[7+c*3] & 15; comp[c].qsel = d[8+c*3];
                if (comp[c].hs > maxH) maxH = comp[c].hs; if (comp[c].vs > maxV) maxV = comp[c].vs; }
        } else if (m == 0xC2) { return nullptr;         /* progressive: unsupported */
        } else if (m == 0xC4) {                         /* DHT */
            int q = 0;
            while (q < dl) { int tc = d[q] >> 4, th = d[q] & 15; q++;
                JpgHuff &h = tc ? hac[th] : hdc[th]; int tot = 0;
                for (int i = 1; i <= 16; ++i) { h.bits[i] = d[q+i-1]; tot += h.bits[i]; } q += 16;
                for (int i = 0; i < tot; ++i) h.vals[i] = d[q+i]; q += tot;
                jpg_build(h); }
        } else if (m == 0xDA) {                         /* SOS */
            int ns = d[0];
            for (int c = 0; c < ns; ++c) { int cid = d[1+c*2], tt = d[2+c*2];
                for (int k = 0; k < ncomp; ++k) if (comp[k].id == cid) { comp[k].dcsel = tt >> 4; comp[k].acsel = tt & 15; } }
            p += 2 + dl;                                 /* entropy data starts here */
            /* ---- decode the entropy-coded scan ---- */
            if (W <= 0 || H <= 0) return nullptr;
            int mcuW = 8 * maxH, mcuH = 8 * maxV;
            int mcx = (W + mcuW - 1) / mcuW, mcy = (H + mcuH - 1) / mcuH;
            std::vector<std::vector<double>> plane(static_cast<size_t>(ncomp));
            std::vector<int> pw(static_cast<size_t>(ncomp)), ph(static_cast<size_t>(ncomp));
            for (int c = 0; c < ncomp; ++c) { pw[c] = mcx * comp[c].hs * 8; ph[c] = mcy * comp[c].vs * 8; plane[c].assign(static_cast<size_t>(pw[c]) * ph[c], 0); comp[c].pred = 0; }
            JpgBits s{f, n, p, 0, 0, 0};
            for (int my = 0; my < mcy; ++my) for (int mx = 0; mx < mcx; ++mx)
                for (int c = 0; c < ncomp; ++c)
                    for (int by = 0; by < comp[c].vs; ++by) for (int bx = 0; bx < comp[c].hs; ++bx) {
                        double blk[64] = {0};
                        int t = jpg_decode(s, hdc[comp[c].dcsel]); int diff = jpg_recv_ext(s, t);
                        comp[c].pred += diff; blk[0] = comp[c].pred * qt[comp[c].qsel][0];
                        for (int k = 1; k < 64; ) {
                            int rs = jpg_decode(s, hac[comp[c].acsel]); int r = rs >> 4, sz = rs & 15;
                            if (sz == 0) { if (r == 15) { k += 16; continue; } break; }
                            k += r; if (k >= 64) break;
                            blk[JPG_ZZ[k]] = jpg_recv_ext(s, sz) * qt[comp[c].qsel][k]; k++;
                        }
                        jpg_idct(blk);
                        int ox = (mx * comp[c].hs + bx) * 8, oy = (my * comp[c].vs + by) * 8;
                        for (int yy = 0; yy < 8; ++yy) for (int xx = 0; xx < 8; ++xx) {
                            int px = ox + xx, py = oy + yy; if (px < pw[c] && py < ph[c]) plane[c][static_cast<size_t>(py) * pw[c] + px] = blk[yy*8+xx] + 128.0;
                        }
                    }
            /* ---- upsample + colour convert into the output ---- */
            bool gray = (ncomp == 1);
            auto fetch = [&](int c, int x, int y) {
                int sx = x * comp[c].hs / maxH, sy = y * comp[c].vs / maxV;
                if (sx >= pw[c]) sx = pw[c]-1; if (sy >= ph[c]) sy = ph[c]-1;
                return plane[c][static_cast<size_t>(sy) * pw[c] + sx];
            };
            if (gray) {
                matlab_mat *R = mat_alloc(H, W);
                for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) R->data[y*W+x] = img_clamp(fetch(0, x, y), 0, 255);
                return R;
            }
            matlab_mat3 *R = mat3_alloc(H, W, 3); int64_t pl = static_cast<int64_t>(H) * W;
            for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                double Y = fetch(0, x, y), Cb = fetch(1, x, y) - 128.0, Cr = fetch(2, x, y) - 128.0;
                R->data[0*pl + y*W+x] = img_clamp(Y + 1.402*Cr, 0, 255);
                R->data[1*pl + y*W+x] = img_clamp(Y - 0.344136*Cb - 0.714136*Cr, 0, 255);
                R->data[2*pl + y*W+x] = img_clamp(Y + 1.772*Cb, 0, 255);
            }
            return reinterpret_cast<matlab_mat *>(R);
        }
        p += dl;
    }
    return nullptr;
}

/* image arithmetic binop (saturating); file scope — templates
 * can't live inside extern "C". */
template <class F>
static matlab_mat *img_binop(matlab_mat *A, matlab_mat *B, F f) {
    if (!A || !A->data) return mat_alloc(0, 0);
    /* Depth-aware: a 3-D image (M×N×P) is processed over its full buffer
     * and a fresh mat3 returned, so imadd/imsubtract/… work on truecolor
     * (and any depth).  B may be the same-shape 3-D image or a scalar. */
    if (mat_is_3d(A)) {
        matlab_mat3 *A3 = reinterpret_cast<matlab_mat3 *>(A);
        int64_t tot = A3->rows * A3->cols * A3->depth;
        bool b3 = (B && mat_is_3d(B));
        bool bscalar = (B && !b3 && B->rows * B->cols == 1);
        matlab_mat3 *B3 = b3 ? reinterpret_cast<matlab_mat3 *>(B) : nullptr;
        matlab_mat3 *R3 = mat3_alloc(A3->rows, A3->cols, A3->depth);
        for (int64_t i = 0; i < tot; ++i) {
            double b = B ? (b3 ? B3->data[i] : (bscalar ? B->data[0] : 0.0)) : 0.0;
            R3->data[i] = img_clamp(f(A3->data[i], b), 0, 255);
        }
        return reinterpret_cast<matlab_mat *>(R3);
    }
    int64_t n = A->rows * A->cols;
    bool bscalar = (B && B->rows * B->cols == 1);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) {
        double b = B ? (bscalar ? B->data[0] : B->data[i]) : 0.0;
        R->data[i] = img_clamp(f(A->data[i], b), 0, 255);
    }
    return R;
}

/* ===== Tier-3 geometric helpers (file scope) ============================ */

/* obj accessors for the affine2d / projective2d transform classdefs. */
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" int         matlab_obj_is_known(const void *p);

/* method code from a string arg (0=nearest, 1=bilinear, 2=bicubic). */
static int img_method(const void *s, int dflt) {
    std::string t = img_sstr(s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    if (t.find("nearest") != std::string::npos) return 0;
    if (t.find("bilinear") != std::string::npos || t.find("linear") != std::string::npos) return 1;
    if (t.find("bicubic") != std::string::npos || t.find("cubic") != std::string::npos) return 2;
    return dflt;
}
static double img_cubic(double t) {                 /* cubic-convolution kernel, a=-0.5 */
    double a = -0.5, x = fabs(t);
    if (x <= 1.0) return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    if (x < 2.0)  return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    return 0.0;
}
/* sample plane d (H×W row-major) at fractional (fy,fx); 0 outside. */
static double img_sample_plane(const double *d, int64_t H, int64_t W,
                              double fy, double fx, int method) {
    if (method == 0) {
        int64_t iy = static_cast<int64_t>(floor(fy + 0.5)), ix = static_cast<int64_t>(floor(fx + 0.5));
        return (iy >= 0 && iy < H && ix >= 0 && ix < W) ? d[iy * W + ix] : 0.0;
    }
    int64_t y0 = static_cast<int64_t>(floor(fy)), x0 = static_cast<int64_t>(floor(fx));
    auto at = [&](int64_t yy, int64_t xx) { return (yy >= 0 && yy < H && xx >= 0 && xx < W) ? d[yy * W + xx] : 0.0; };
    if (method == 2) {
        double sum = 0.0;
        for (int m = -1; m <= 2; ++m) for (int nn = -1; nn <= 2; ++nn)
            sum += at(y0 + m, x0 + nn) * img_cubic(fy - (y0 + m)) * img_cubic(fx - (x0 + nn));
        return sum;
    }
    double dy = fy - y0, dx = fx - x0;
    return at(y0, x0) * (1 - dy) * (1 - dx) + at(y0, x0 + 1) * (1 - dy) * dx +
           at(y0 + 1, x0) * dy * (1 - dx) + at(y0 + 1, x0 + 1) * dy * dx;
}

/* 3×3 inverse (row-major in/out); returns identity on singular. */
static void img_inv3x3(const double *M, double *out) {
    double det = M[0]*(M[4]*M[8]-M[5]*M[7]) - M[1]*(M[3]*M[8]-M[5]*M[6]) + M[2]*(M[3]*M[7]-M[4]*M[6]);
    if (fabs(det) < 1e-300) { for (int i = 0; i < 9; ++i) out[i] = (i % 4 == 0) ? 1.0 : 0.0; return; }
    double id = 1.0 / det;
    out[0] =  (M[4]*M[8]-M[5]*M[7]) * id; out[1] = -(M[1]*M[8]-M[2]*M[7]) * id; out[2] =  (M[1]*M[5]-M[2]*M[4]) * id;
    out[3] = -(M[3]*M[8]-M[5]*M[6]) * id; out[4] =  (M[0]*M[8]-M[2]*M[6]) * id; out[5] = -(M[0]*M[5]-M[2]*M[3]) * id;
    out[6] =  (M[3]*M[7]-M[4]*M[6]) * id; out[7] = -(M[0]*M[7]-M[1]*M[6]) * id; out[8] =  (M[0]*M[4]-M[1]*M[3]) * id;
}

/* solve A x = b (A n×n row-major) by Gaussian elimination w/ partial pivot. */
static std::vector<double> img_gauss(std::vector<double> A, std::vector<double> b, int n) {
    for (int col = 0; col < n; ++col) {
        int piv = col; double best = fabs(A[static_cast<size_t>(col * n + col)]);
        for (int r = col + 1; r < n; ++r) { double v = fabs(A[static_cast<size_t>(r * n + col)]); if (v > best) { best = v; piv = r; } }
        if (best < 1e-300) continue;
        if (piv != col) { for (int j = 0; j < n; ++j) std::swap(A[static_cast<size_t>(col*n+j)], A[static_cast<size_t>(piv*n+j)]); std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(piv)]); }
        double d = A[static_cast<size_t>(col*n+col)];
        for (int j = 0; j < n; ++j) A[static_cast<size_t>(col*n+j)] /= d; b[static_cast<size_t>(col)] /= d;
        for (int r = 0; r < n; ++r) { if (r == col) continue; double f = A[static_cast<size_t>(r*n+col)];
            for (int j = 0; j < n; ++j) A[static_cast<size_t>(r*n+j)] -= f * A[static_cast<size_t>(col*n+j)]; b[static_cast<size_t>(r)] -= f * b[static_cast<size_t>(col)]; }
    }
    return b;
}

/* Apply an inverse-coordinate map to produce an OH×OW output (grayscale or
 * per-channel RGB), sampling the input with `method`.  invmap(oy,ox) ->
 * (iy,ix) in input pixel coordinates. */
template <class Map>
static matlab_mat *img_geo_apply(void *A, int64_t OH, int64_t OW, int method, Map invmap) {
    if (mat_is_3d(A)) {
        matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
        int64_t H = m->rows, W = m->cols, ip = H * W, op = OH * OW;
        matlab_mat3 *R = mat3_alloc(OH, OW, m->depth);
        for (int64_t oy = 0; oy < OH; ++oy) for (int64_t ox = 0; ox < OW; ++ox) {
            double iy, ix; invmap(oy, ox, iy, ix);
            for (int64_t c = 0; c < m->depth; ++c)
                R->data[c * op + oy * OW + ox] = img_sample_plane(m->data + c * ip, H, W, iy, ix, method);
        }
        return reinterpret_cast<matlab_mat *>(R);
    }
    matlab_mat *m = reinterpret_cast<matlab_mat *>(A);
    int64_t H = m->rows, W = m->cols;
    matlab_mat *R = mat_alloc(OH, OW);
    for (int64_t oy = 0; oy < OH; ++oy) for (int64_t ox = 0; ox < OW; ++ox) {
        double iy, ix; invmap(oy, ox, iy, ix);
        R->data[oy * OW + ox] = img_sample_plane(m->data, H, W, iy, ix, method);
    }
    return R;
}
static void img_dims(void *A, int64_t &H, int64_t &W) {
    if (mat_is_3d(A)) { matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A); H = m->rows; W = m->cols; }
    else { matlab_mat *m = reinterpret_cast<matlab_mat *>(A); H = m->rows; W = m->cols; }
}

/* ===== Tier-6 helpers (file scope) ====================================== */

/* per-pixel colour transform over an M×N×3 slice-major image. */
template <class F>
static matlab_mat *img_color_apply(matlab_mat *A, F fn) {
    if (!A || !mat_is_3d(A)) return mat_alloc(0, 0);
    matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
    int64_t H = m->rows, W = m->cols, pl = H * W;
    matlab_mat3 *R = mat3_alloc(H, W, 3);
    for (int64_t i = 0; i < pl; ++i)
        fn(m->data[i], m->data[pl + i], m->data[2 * pl + i],
           R->data[i], R->data[pl + i], R->data[2 * pl + i]);
    return reinterpret_cast<matlab_mat *>(R);
}
static double img_srgb2lin(double c) { c /= 255.0; return (c <= 0.04045) ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
static double img_lin2srgb(double c) { double v = (c <= 0.0031308) ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055; return img_clamp(v * 255.0, 0, 255); }
static double img_labf(double t)  { return (t > 0.008856) ? cbrt(t) : (7.787 * t + 16.0 / 116.0); }
static double img_labfi(double t) { double t3 = t * t * t; return (t3 > 0.008856) ? t3 : (t - 16.0 / 116.0) / 7.787; }

extern "C" {

/* ===== Tier-1 — file I/O ================================================= */

matlab_mat *matlab_image_imread(void *path_s) {
    std::string path = img_sstr(path_s);
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return mat_alloc(0, 0);
    /* PNG: read the whole file and hand it to the hand-coded decoder. */
    unsigned char head[8] = {0};
    size_t got8 = fread(head, 1, 8, f);
    static const unsigned char pngsig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    bool isPng = (got8 == 8 && memcmp(head, pngsig, 8) == 0);
    bool isJpg = (got8 >= 2 && head[0] == 0xFF && head[1] == 0xD8);
    if (isPng || isJpg) {
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        if (sz <= 0) { fclose(f); return mat_alloc(0, 0); }
        std::vector<unsigned char> buf(static_cast<size_t>(sz));
        size_t rd = fread(buf.data(), 1, static_cast<size_t>(sz), f);
        fclose(f);
        matlab_mat *r = isPng ? png_decode(buf.data(), rd) : jpg_decode_file(buf.data(), rd);
        return r ? r : mat_alloc(0, 0);
    }
    fseek(f, 0, SEEK_SET);
    char magic[2] = {0, 0};
    if (fread(magic, 1, 2, f) != 2) { fclose(f); return mat_alloc(0, 0); }
    matlab_mat *out = nullptr;

    if (magic[0] == 'P' && (magic[1] == '5' || magic[1] == '2')) {   /* PGM */
        int w = pnm_next(f), h = pnm_next(f), mx = pnm_next(f);
        if (w <= 0 || h <= 0) { fclose(f); return mat_alloc(0, 0); }
        out = mat_alloc(h, w);
        if (magic[1] == '5') {
            std::vector<unsigned char> buf(static_cast<size_t>(w * h));
            fread(buf.data(), 1, buf.size(), f);
            for (int64_t i = 0; i < static_cast<int64_t>(w) * h; ++i) out->data[i] = buf[static_cast<size_t>(i)];
        } else {
            for (int64_t i = 0; i < static_cast<int64_t>(w) * h; ++i) out->data[i] = pnm_next(f);
        }
        (void)mx;
    } else if (magic[0] == 'P' && (magic[1] == '6' || magic[1] == '3')) { /* PPM RGB */
        int w = pnm_next(f), h = pnm_next(f), mx = pnm_next(f);
        if (w <= 0 || h <= 0) { fclose(f); return mat_alloc(0, 0); }
        matlab_mat3 *rgb = mat3_alloc(h, w, 3);
        int64_t plane = static_cast<int64_t>(w) * h;
        if (magic[1] == '6') {
            std::vector<unsigned char> buf(static_cast<size_t>(w * h * 3));
            fread(buf.data(), 1, buf.size(), f);
            for (int64_t i = 0; i < plane; ++i)
                for (int c = 0; c < 3; ++c)
                    rgb->data[c * plane + i] = buf[static_cast<size_t>(i * 3 + c)];
        } else {
            for (int64_t i = 0; i < plane; ++i)
                for (int c = 0; c < 3; ++c)
                    rgb->data[c * plane + i] = pnm_next(f);
        }
        (void)mx; fclose(f);
        return reinterpret_cast<matlab_mat *>(rgb);
    } else if (magic[0] == 'B' && magic[1] == 'M') {                 /* BMP */
        unsigned char hdr[52];
        if (fread(hdr, 1, 52, f) != 52) { fclose(f); return mat_alloc(0, 0); }
        uint32_t off = hdr[8] | (hdr[9] << 8) | (hdr[10] << 16) | (hdr[11] << 24);
        int32_t w = hdr[16] | (hdr[17] << 8) | (hdr[18] << 16) | (hdr[19] << 24);
        int32_t h = hdr[20] | (hdr[21] << 8) | (hdr[22] << 16) | (hdr[23] << 24);
        uint16_t bpp = hdr[26] | (hdr[27] << 8);
        bool topdown = h < 0; if (topdown) h = -h;
        if (w <= 0 || h <= 0) { fclose(f); return mat_alloc(0, 0); }
        fseek(f, off, SEEK_SET);
        int channels = (bpp == 24) ? 3 : 1;
        int rowbytes = ((w * (bpp / 8) + 3) / 4) * 4;
        std::vector<unsigned char> row(static_cast<size_t>(rowbytes));
        if (channels == 3) {
            matlab_mat3 *rgb = mat3_alloc(h, w, 3);
            int64_t plane = static_cast<int64_t>(w) * h;
            for (int r = 0; r < h; ++r) {
                fread(row.data(), 1, row.size(), f);
                int dr = topdown ? r : (h - 1 - r);
                for (int x = 0; x < w; ++x) {
                    rgb->data[2 * plane + dr * w + x] = row[static_cast<size_t>(x * 3 + 0)]; /* B */
                    rgb->data[1 * plane + dr * w + x] = row[static_cast<size_t>(x * 3 + 1)]; /* G */
                    rgb->data[0 * plane + dr * w + x] = row[static_cast<size_t>(x * 3 + 2)]; /* R */
                }
            }
            fclose(f);
            return reinterpret_cast<matlab_mat *>(rgb);
        }
        out = mat_alloc(h, w);
        for (int r = 0; r < h; ++r) {
            fread(row.data(), 1, row.size(), f);
            int dr = topdown ? r : (h - 1 - r);
            for (int x = 0; x < w; ++x) out->data[dr * w + x] = row[static_cast<size_t>(x)];
        }
    } else {
        fclose(f);
        return mat_alloc(0, 0);   /* PNG/JPEG/TIFF: documented format follow-on */
    }
    fclose(f);
    return out ? out : mat_alloc(0, 0);
}

double matlab_image_imwrite(void *img, void *path_s) {
    std::string path = img_sstr(path_s);
    std::string ext = img_ext_lower(path);
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return 0.0;
    if (ext == "png") {                              /* lossless PNG (store mode) */
        std::vector<unsigned char> buf = png_encode(img);
        fwrite(buf.data(), 1, buf.size(), f); fclose(f); return 1.0;
    }
    bool rgb = mat_is_3d(img);
    int64_t H, W; const double *d; int64_t plane = 0;
    if (rgb) { matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(img); H = m->rows; W = m->cols; d = m->data; plane = H * W; }
    else     { matlab_mat *m = reinterpret_cast<matlab_mat *>(img);  H = m->rows; W = m->cols; d = m->data; }
    auto px = [&](int64_t i, int c) { return static_cast<unsigned char>(img_clamp(rgb ? d[c * plane + i] : d[i], 0, 255) + 0.5); };

    if (ext == "ppm" || (ext != "pgm" && ext != "bmp" && rgb)) {     /* PPM (P6) */
        fprintf(f, "P6\n%lld %lld\n255\n", static_cast<long long>(W), static_cast<long long>(H));
        for (int64_t i = 0; i < H * W; ++i)
            for (int c = 0; c < 3; ++c) { unsigned char b = px(i, rgb ? c : 0); fwrite(&b, 1, 1, f); }
    } else if (ext == "pgm" || (!rgb && ext != "bmp")) {             /* PGM (P5) */
        fprintf(f, "P5\n%lld %lld\n255\n", static_cast<long long>(W), static_cast<long long>(H));
        for (int64_t i = 0; i < H * W; ++i) { unsigned char b = px(i, 0); fwrite(&b, 1, 1, f); }
    } else {                                                          /* BMP 24-bit */
        int rowbytes = ((W * 3 + 3) / 4) * 4;
        uint32_t dataSize = static_cast<uint32_t>(rowbytes * H);
        uint32_t fileSize = 54 + dataSize;
        unsigned char hdr[54] = {0};
        hdr[0] = 'B'; hdr[1] = 'M';
        hdr[2] = fileSize & 0xff; hdr[3] = (fileSize >> 8) & 0xff; hdr[4] = (fileSize >> 16) & 0xff; hdr[5] = (fileSize >> 24) & 0xff;
        hdr[10] = 54; hdr[14] = 40;
        hdr[18] = W & 0xff; hdr[19] = (W >> 8) & 0xff; hdr[20] = (W >> 16) & 0xff; hdr[21] = (W >> 24) & 0xff;
        hdr[22] = H & 0xff; hdr[23] = (H >> 8) & 0xff; hdr[24] = (H >> 16) & 0xff; hdr[25] = (H >> 24) & 0xff;
        hdr[26] = 1; hdr[28] = 24;
        fwrite(hdr, 1, 54, f);
        std::vector<unsigned char> row(static_cast<size_t>(rowbytes), 0);
        for (int64_t r = H - 1; r >= 0; --r) {
            for (int64_t x = 0; x < W; ++x) {
                int64_t i = r * W + x;
                row[static_cast<size_t>(x * 3 + 0)] = px(i, rgb ? 2 : 0);  /* B */
                row[static_cast<size_t>(x * 3 + 1)] = px(i, rgb ? 1 : 0);  /* G */
                row[static_cast<size_t>(x * 3 + 2)] = px(i, rgb ? 0 : 0);  /* R */
            }
            fwrite(row.data(), 1, row.size(), f);
        }
    }
    fclose(f);
    return 1.0;
}

/* checkerboard(n[, p, q]) — p×q tiles of 2n×2n; values in [0,1] (double). */
matlab_mat *matlab_image_checkerboard(matlab_mat *nm, matlab_mat *pm, matlab_mat *qm) {
    int n = static_cast<int>(img_sc(nm, 10));
    int p = static_cast<int>(img_sc(pm, 4));
    int q = static_cast<int>(img_sc(qm, p));
    if (n < 1) n = 10; if (p < 1) p = 4; if (q < 1) q = p;
    int H = 2 * n * p, W = 2 * n * q;
    matlab_mat *R = mat_alloc(H, W);
    for (int i = 0; i < H; ++i) for (int j = 0; j < W; ++j) {
        int bi = (i / n) & 1, bj = (j / n) & 1;
        R->data[i * W + j] = (bi == bj) ? 0.0 : 1.0;
    }
    return R;
}

/* ===== Tier-1 — type conversions ======================================== */

matlab_mat *matlab_image_im2double(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    if (mat_is_3d(A)) {
        matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
        int64_t n = m->rows * m->cols * m->depth;
        bool u8 = img_is_u8range(m->data, n);
        matlab_mat3 *R = mat3_alloc(m->rows, m->cols, m->depth);
        for (int64_t i = 0; i < n; ++i) R->data[i] = u8 ? m->data[i] / 255.0 : m->data[i];
        return reinterpret_cast<matlab_mat *>(R);
    }
    int64_t n = A->rows * A->cols;
    bool u8 = img_is_u8range(A->data, n);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = u8 ? A->data[i] / 255.0 : A->data[i];
    return R;
}
matlab_mat *matlab_image_im2single(matlab_mat *A) { return matlab_image_im2double(A); }

matlab_mat *matlab_image_im2uint8(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);   /* values in [0,1] -> scale by 255 */
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i)
        R->data[i] = img_clamp(floor((norm ? 255.0 * A->data[i] : A->data[i]) + 0.5), 0, 255);
    return R;
}

/* rgb2gray / im2gray: 0.2989R + 0.5870G + 0.1140B; passthrough if 2-D. */
matlab_mat *matlab_image_rgb2gray(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    if (!mat_is_3d(A)) {
        matlab_mat *R = mat_alloc(A->rows, A->cols);
        memcpy(R->data, A->data, sizeof(double) * static_cast<size_t>(A->rows * A->cols));
        return R;
    }
    matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A);
    int64_t plane = m->rows * m->cols;
    matlab_mat *R = mat_alloc(m->rows, m->cols);
    for (int64_t i = 0; i < plane; ++i)
        R->data[i] = 0.2989 * m->data[i] + 0.5870 * m->data[plane + i] + 0.1140 * m->data[2 * plane + i];
    return R;
}

matlab_mat *matlab_image_mat2gray(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    double lo = A->data[0], hi = A->data[0];
    for (int64_t i = 1; i < n; ++i) { if (A->data[i] < lo) lo = A->data[i]; if (A->data[i] > hi) hi = A->data[i]; }
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    double rng = hi - lo;
    for (int64_t i = 0; i < n; ++i) R->data[i] = (rng > 0) ? (A->data[i] - lo) / rng : 0.0;
    return R;
}

/* ===== Tier-1 — image arithmetic (saturating to [0,255]) ================ */

matlab_mat *matlab_image_imadd(matlab_mat *A, matlab_mat *B)      { return img_binop(A, B, [](double a, double b){ return a + b; }); }
matlab_mat *matlab_image_imsubtract(matlab_mat *A, matlab_mat *B) { return img_binop(A, B, [](double a, double b){ return a - b; }); }
matlab_mat *matlab_image_immultiply(matlab_mat *A, matlab_mat *B) { return img_binop(A, B, [](double a, double b){ return a * b; }); }
matlab_mat *matlab_image_imdivide(matlab_mat *A, matlab_mat *B)   { return img_binop(A, B, [](double a, double b){ return b != 0 ? a / b : 0.0; }); }
matlab_mat *matlab_image_imabsdiff(matlab_mat *A, matlab_mat *B)  { return img_binop(A, B, [](double a, double b){ return fabs(a - b); }); }
matlab_mat *matlab_image_imcomplement(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    if (mat_is_3d(A)) {
        matlab_mat3 *A3 = reinterpret_cast<matlab_mat3 *>(A);
        int64_t tot = A3->rows * A3->cols * A3->depth;
        double top = img_is_u8range(A3->data, tot) ? 255.0 : 1.0;
        matlab_mat3 *R3 = mat3_alloc(A3->rows, A3->cols, A3->depth);
        for (int64_t i = 0; i < tot; ++i) R3->data[i] = top - A3->data[i];
        return reinterpret_cast<matlab_mat *>(R3);
    }
    int64_t n = A->rows * A->cols;
    double top = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = top - A->data[i];
    return R;
}
/* imlincomb(k1, A1, k2, A2) = k1*A1 + k2*A2 (saturated). */
matlab_mat *matlab_image_imlincomb(matlab_mat *k1, matlab_mat *A1, matlab_mat *k2, matlab_mat *A2) {
    if (!A1 || !A1->data) return mat_alloc(0, 0);
    double a = img_sc(k1, 1.0), b = img_sc(k2, 0.0);
    if (mat_is_3d(A1)) {
        matlab_mat3 *X = reinterpret_cast<matlab_mat3 *>(A1);
        int64_t tot = X->rows * X->cols * X->depth;
        bool y3 = (A2 && mat_is_3d(A2));
        matlab_mat3 *Y = y3 ? reinterpret_cast<matlab_mat3 *>(A2) : nullptr;
        matlab_mat3 *R3 = mat3_alloc(X->rows, X->cols, X->depth);
        for (int64_t i = 0; i < tot; ++i)
            R3->data[i] = img_clamp(a * X->data[i] + (Y ? b * Y->data[i] : 0.0), 0, 255);
        return reinterpret_cast<matlab_mat *>(R3);
    }
    int64_t n = A1->rows * A1->cols;
    matlab_mat *R = mat_alloc(A1->rows, A1->cols);
    for (int64_t i = 0; i < n; ++i)
        R->data[i] = img_clamp(a * A1->data[i] + (A2 ? b * A2->data[i] : 0.0), 0, 255);
    return R;
}

/* ===== Tier-1 — histogram + intensity stats ============================= */

matlab_mat *matlab_image_imhist(matlab_mat *A) {
    matlab_mat *R = mat_alloc(256, 1);
    if (!A || !A->data) return R;
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    for (int64_t i = 0; i < n; ++i) {
        double v = norm ? 255.0 * A->data[i] : A->data[i];
        int b = static_cast<int>(img_clamp(floor(v + 0.5), 0, 255));
        R->data[b] += 1.0;
    }
    return R;
}
matlab_mat *matlab_image_stretchlim(matlab_mat *A) {
    matlab_mat *R = mat_alloc(2, 1);
    if (!A || !A->data) { R->data[0] = 0; R->data[1] = 1; return R; }
    int64_t n = A->rows * A->cols;
    std::vector<double> v(A->data, A->data + n);
    std::sort(v.begin(), v.end());
    double scale = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    int64_t lo = static_cast<int64_t>(0.01 * n), hi = static_cast<int64_t>(0.99 * n);
    if (hi >= n) hi = n - 1;
    R->data[0] = v[static_cast<size_t>(lo)] / scale;
    R->data[1] = v[static_cast<size_t>(hi)] / scale;
    return R;
}
/* imadjust(A) auto-stretch; imadjust(A,[lo hi],[bot top][,gamma]). */
static matlab_mat *img_adjust(matlab_mat *A, double lin, double hin, double lout, double hout, double gamma) {
    int64_t n = A->rows * A->cols;
    double scale = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    double span = (hin - lin);
    for (int64_t i = 0; i < n; ++i) {
        double x = A->data[i] / scale;
        double t = (span > 0) ? img_clamp((x - lin) / span, 0, 1) : 0.0;
        double y = pow(t, gamma) * (hout - lout) + lout;
        R->data[i] = y * scale;
    }
    return R;
}
matlab_mat *matlab_image_imadjust1(matlab_mat *A) {
    matlab_mat *sl = matlab_image_stretchlim(A);
    return img_adjust(A, sl->data[0], sl->data[1], 0.0, 1.0, 1.0);
}
matlab_mat *matlab_image_imadjust(matlab_mat *A, matlab_mat *in, matlab_mat *out) {
    double lin = (in && in->rows * in->cols >= 2) ? in->data[0] : 0.0;
    double hin = (in && in->rows * in->cols >= 2) ? in->data[1] : 1.0;
    double lout = (out && out->rows * out->cols >= 2) ? out->data[0] : 0.0;
    double hout = (out && out->rows * out->cols >= 2) ? out->data[1] : 1.0;
    return img_adjust(A, lin, hin, lout, hout, 1.0);
}
matlab_mat *matlab_image_imadjustg(matlab_mat *A, matlab_mat *in, matlab_mat *out, matlab_mat *g) {
    double lin = (in && in->rows * in->cols >= 2) ? in->data[0] : 0.0;
    double hin = (in && in->rows * in->cols >= 2) ? in->data[1] : 1.0;
    double lout = (out && out->rows * out->cols >= 2) ? out->data[0] : 0.0;
    double hout = (out && out->rows * out->cols >= 2) ? out->data[1] : 1.0;
    return img_adjust(A, lin, hin, lout, hout, img_sc(g, 1.0));
}
double matlab_image_mean2(matlab_mat *A) {
    if (!A || !A->data) return 0.0;
    int64_t n = A->rows * A->cols; double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += A->data[i];
    return n ? s / n : 0.0;
}
double matlab_image_std2(matlab_mat *A) {
    if (!A || !A->data) return 0.0;
    int64_t n = A->rows * A->cols; double mu = matlab_image_mean2(A), s = 0.0;
    for (int64_t i = 0; i < n; ++i) { double d = A->data[i] - mu; s += d * d; }
    return n ? sqrt(s / n) : 0.0;
}

/* ===== Tier-2 — fspecial kernels ======================================== */
/* type codes: 1=gaussian 2=average 3=laplacian 4=log 5=sobel 6=prewitt
 * 7=disk 8=motion 9=unsharp. */
matlab_mat *matlab_image_fspecial(void *type_s, matlab_mat *p1m, matlab_mat *p2m) {
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    if (t == "sobel")   { matlab_mat *K = mat_alloc(3, 3); double s[9] = {1,2,1, 0,0,0, -1,-2,-1}; for (int i=0;i<9;++i) K->data[i]=s[i]; return K; }
    if (t == "prewitt") { matlab_mat *K = mat_alloc(3, 3); double s[9] = {1,1,1, 0,0,0, -1,-1,-1}; for (int i=0;i<9;++i) K->data[i]=s[i]; return K; }
    if (t == "laplacian") {
        double a = img_sc(p1m, 0.2);
        matlab_mat *K = mat_alloc(3, 3);
        double c = 4.0 / (a + 1.0), e = a / (a + 1.0), mid = -4.0 / (a + 1.0);
        double s[9] = {e, c/4*0+ (1-a)/(a+1), e,  (1-a)/(a+1), -4.0/(a+1.0), (1-a)/(a+1),  e, (1-a)/(a+1), e};
        (void)c; (void)mid; for (int i=0;i<9;++i) K->data[i]=s[i]; return K;
    }
    if (t == "average") {
        int sz = static_cast<int>(img_sc(p1m, 3));
        matlab_mat *K = mat_alloc(sz, sz);
        for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] = 1.0 / (sz * sz);
        return K;
    }
    if (t == "disk") {
        int r = static_cast<int>(img_sc(p1m, 5));
        int sz = 2 * r + 1; matlab_mat *K = mat_alloc(sz, sz); double sum = 0.0;
        for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
            double dy = i - r, dx = j - r; double w = (dx*dx + dy*dy <= static_cast<double>(r) * r) ? 1.0 : 0.0;
            K->data[i*sz+j] = w; sum += w;
        }
        if (sum > 0) for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] /= sum;
        return K;
    }
    if (t == "log") {
        int sz = static_cast<int>(img_sc(p1m, 5)); double sg = img_sc(p2m, 0.5);
        int r = sz / 2; matlab_mat *K = mat_alloc(sz, sz); double s2 = sg*sg, sum = 0.0;
        for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
            double x = j - r, y = i - r, h = (x*x + y*y - 2*s2) / (s2*s2) * exp(-(x*x+y*y)/(2*s2));
            K->data[i*sz+j] = h; sum += h;
        }
        double mean = sum / (sz*sz); for (int64_t i=0;i<static_cast<int64_t>(sz) * sz;++i) K->data[i] -= mean;
        return K;
    }
    if (t == "motion") {
        int len = static_cast<int>(img_sc(p1m, 9));
        matlab_mat *K = mat_alloc(1, len);
        for (int j = 0; j < len; ++j) K->data[j] = 1.0 / len;
        return K;
    }
    /* gaussian (default) */
    int sz = static_cast<int>(img_sc(p1m, 3)); double sg = img_sc(p2m, 0.5);
    int r = sz / 2; matlab_mat *K = mat_alloc(sz, sz); double sum = 0.0;
    for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
        double x = j - r, y = i - r, h = exp(-(x*x + y*y) / (2*sg*sg));
        K->data[i*sz+j] = h; sum += h;
    }
    if (sum > 0) for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] /= sum;
    return K;
}

/* imgaussfilt(A, sigma): separable Gaussian via fspecial+imfilter. */
matlab_mat *matlab_image_imgaussfilt(matlab_mat *A, matlab_mat *sigm) {
    double sg = img_sc(sigm, 0.5);
    int r = static_cast<int>(ceil(2.0 * sg)); if (r < 1) r = 1;
    int sz = 2 * r + 1;
    matlab_mat *K = mat_alloc(sz, sz); double sum = 0.0;
    for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) {
        double x = j - r, y = i - r, h = exp(-(x*x + y*y) / (2*sg*sg));
        K->data[i*sz+j] = h; sum += h;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] /= sum;
    return matlab_imfilter(A, K);
}
matlab_mat *matlab_image_imboxfilt(matlab_mat *A, matlab_mat *szm) {
    int sz = static_cast<int>(img_sc(szm, 3));
    matlab_mat *K = mat_alloc(sz, sz);
    for (int64_t i = 0; i < static_cast<int64_t>(sz) * sz; ++i) K->data[i] = 1.0 / (sz*sz);
    return matlab_imfilter(A, K);
}

/* ----- rank/order filters ------------------------------------------------ */
static matlab_mat *img_rankfilt(matlab_mat *A, int mh, int mw, int order /* 1-based; -1 = mean of window stat */, int mode) {
    /* mode 0 = order statistic; mode 1 = std; mode 2 = range. */
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    int rr = mh / 2, rc = mw / 2;
    matlab_mat *R = mat_alloc(H, W);
    std::vector<double> win;
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        win.clear();
        for (int di = -rr; di <= rr; ++di) for (int dj = -rc; dj <= rc; ++dj) {
            int64_t y = i + di, x = j + dj;
            double v = (y >= 0 && y < H && x >= 0 && x < W) ? A->data[y * W + x] : 0.0;
            win.push_back(v);
        }
        if (mode == 0) {
            std::sort(win.begin(), win.end());
            int k = order; if (k < 1) k = static_cast<int>(win.size()) / 2 + 1; if (k > static_cast<int>(win.size())) k = static_cast<int>(win.size());
            R->data[i * W + j] = win[static_cast<size_t>(k - 1)];
        } else {
            double mu = 0.0; for (double v : win) mu += v; mu /= win.size();
            if (mode == 1) { double s = 0.0; for (double v : win) s += (v-mu)*(v-mu); R->data[i*W+j] = sqrt(s/win.size()); }
            else { double lo = win[0], hi = win[0]; for (double v : win) { if (v<lo) lo=v; if (v>hi) hi=v; } R->data[i*W+j] = hi - lo; }
        }
    }
    return R;
}
matlab_mat *matlab_image_medfilt2(matlab_mat *A, matlab_mat *szm) {
    int m = 3, n = 3;
    if (szm && szm->rows * szm->cols >= 2) { m = static_cast<int>(szm->data[0]); n = static_cast<int>(szm->data[1]); }
    else if (szm && szm->rows * szm->cols == 1) { m = n = static_cast<int>(szm->data[0]); }
    return img_rankfilt(A, m, n, (m*n)/2 + 1, 0);
}
matlab_mat *matlab_image_ordfilt2(matlab_mat *A, matlab_mat *ordm, matlab_mat *domm) {
    int order = static_cast<int>(img_sc(ordm, 1));
    int sz = 3; if (domm && domm->rows * domm->cols > 1) sz = static_cast<int>(domm->rows);
    return img_rankfilt(A, sz, sz, order, 0);
}
matlab_mat *matlab_image_stdfilt(matlab_mat *A)   { return img_rankfilt(A, 3, 3, 0, 1); }
matlab_mat *matlab_image_rangefilt(matlab_mat *A) { return img_rankfilt(A, 3, 3, 0, 2); }

/* ===== Tier-2 — enhancement ============================================= */

/* histeq(A): global histogram equalisation to [0,255] (uint8-class). */
matlab_mat *matlab_image_histeq(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    double hist[256] = {0};
    for (int64_t i = 0; i < n; ++i) {
        double v = norm ? 255.0 * A->data[i] : A->data[i];
        hist[static_cast<int>(img_clamp(floor(v + 0.5), 0, 255))] += 1.0;
    }
    double cdf[256]; double acc = 0.0;
    for (int k = 0; k < 256; ++k) { acc += hist[k]; cdf[k] = acc; }
    double total = cdf[255], cdfmin = 0.0;
    for (int k = 0; k < 256; ++k) if (cdf[k] > 0) { cdfmin = cdf[k]; break; }
    double map[256];
    for (int k = 0; k < 256; ++k)
        map[k] = (total > cdfmin) ? (cdf[k] - cdfmin) / (total - cdfmin) * 255.0 : k;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) {
        double v = norm ? 255.0 * A->data[i] : A->data[i];
        int b = static_cast<int>(img_clamp(floor(v + 0.5), 0, 255));
        R->data[i] = norm ? map[b] / 255.0 : map[b];
    }
    return R;
}

/* adapthisteq(A): tiled adaptive equalisation (8×8 tiles, clip 0.01),
 * bilinear interpolation between tile mappings (a compact CLAHE). */
matlab_mat *matlab_image_adapthisteq(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    bool norm = !img_is_u8range(A->data, H * W);
    int nt = 8;
    int th = static_cast<int>((H + nt - 1) / nt), tw = static_cast<int>((W + nt - 1) / nt);
    std::vector<std::vector<double>> maps(static_cast<size_t>(nt * nt), std::vector<double>(256, 0.0));
    auto pix = [&](int64_t i, int64_t j) { double v = A->data[i*W+j]; return img_clamp(floor((norm?255.0*v:v)+0.5),0,255); };
    for (int ty = 0; ty < nt; ++ty) for (int tx = 0; tx < nt; ++tx) {
        double hist[256] = {0}; int64_t cnt = 0;
        for (int64_t i = ty*th; i < std::min<int64_t>((ty+1)*th, H); ++i)
            for (int64_t j = tx*tw; j < std::min<int64_t>((tx+1)*tw, W); ++j) { hist[static_cast<int>(pix(i,j))]++; cnt++; }
        double clip = 0.01 * cnt, excess = 0.0;
        for (int k = 0; k < 256; ++k) if (hist[k] > clip) { excess += hist[k] - clip; hist[k] = clip; }
        double add = excess / 256.0; for (int k = 0; k < 256; ++k) hist[k] += add;
        double acc = 0.0; auto &mp = maps[static_cast<size_t>(ty*nt+tx)];
        for (int k = 0; k < 256; ++k) { acc += hist[k]; mp[static_cast<size_t>(k)] = (cnt>0) ? acc/cnt*255.0 : k; }
    }
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        int b = static_cast<int>(pix(i, j));
        double gy = (i + 0.5) / th - 0.5, gx = (j + 0.5) / tw - 0.5;
        int ty0 = static_cast<int>(floor(gy)), tx0 = static_cast<int>(floor(gx));
        double fy = gy - ty0, fx = gx - tx0;
        auto M = [&](int ty, int tx) { ty = std::max(0, std::min(nt-1, ty)); tx = std::max(0, std::min(nt-1, tx)); return maps[static_cast<size_t>(ty*nt+tx)][static_cast<size_t>(b)]; };
        double v = (1-fy)*(1-fx)*M(ty0,tx0) + (1-fy)*fx*M(ty0,tx0+1) + fy*(1-fx)*M(ty0+1,tx0) + fy*fx*M(ty0+1,tx0+1);
        R->data[i*W+j] = norm ? v/255.0 : v;
    }
    return R;
}

/* imsharpen(A): unsharp masking, amount 0.8. */
matlab_mat *matlab_image_imsharpen(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    double sgv = 1.0; matlab_mat sg; sg.data = &sgv; sg.rows = 1; sg.cols = 1;
    matlab_mat *blur = matlab_image_imgaussfilt(A, &sg);
    int64_t n = A->rows * A->cols; double amount = 0.8;
    double top = img_is_u8range(A->data, n) ? 255.0 : 1.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = img_clamp(A->data[i] + amount * (A->data[i] - blur->data[i]), 0, top);
    return R;
}

/* imhistmatch(A, ref): match A's histogram to ref's via CDF mapping. */
matlab_mat *matlab_image_imhistmatch(matlab_mat *A, matlab_mat *ref) {
    if (!A || !A->data || !ref || !ref->data) return mat_alloc(0, 0);
    auto cdf256 = [](matlab_mat *X, double *cdf) {
        int64_t n = X->rows * X->cols; bool norm = !img_is_u8range(X->data, n);
        double h[256] = {0};
        for (int64_t i = 0; i < n; ++i) h[static_cast<int>(img_clamp(floor((norm?255*X->data[i]:X->data[i])+0.5),0,255))]++;
        double acc = 0; for (int k = 0; k < 256; ++k) { acc += h[k]; cdf[k] = acc / n; }
    };
    double ca[256], cr[256]; cdf256(A, ca); cdf256(ref, cr);
    int map[256];
    for (int k = 0; k < 256; ++k) { int j = 0; while (j < 255 && cr[j] < ca[k]) j++; map[k] = j; }
    int64_t n = A->rows * A->cols; bool norm = !img_is_u8range(A->data, n);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) {
        int b = static_cast<int>(img_clamp(floor((norm?255*A->data[i]:A->data[i])+0.5),0,255));
        R->data[i] = norm ? map[b] / 255.0 : map[b];
    }
    return R;
}

/* imnoise(A, type, p): gaussian / salt & pepper / speckle. */
matlab_mat *matlab_image_imnoise(matlab_mat *A, void *type_s, matlab_mat *pm) {
    if (!A || !A->data) return mat_alloc(0, 0);
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    double scale = norm ? 1.0 : 255.0;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    if (t.find("salt") != std::string::npos) {
        double d = img_sc(pm, 0.05);
        matlab_mat *u = matlab_rand(A->rows, A->cols);
        for (int64_t i = 0; i < n; ++i) {
            if (u->data[i] < d / 2) R->data[i] = 0.0;
            else if (u->data[i] > 1 - d / 2) R->data[i] = scale;
            else R->data[i] = A->data[i];
        }
    } else if (t.find("speckle") != std::string::npos) {
        double v = img_sc(pm, 0.04);
        matlab_mat *g = matlab_randn(A->rows, A->cols);
        for (int64_t i = 0; i < n; ++i) R->data[i] = img_clamp(A->data[i] + A->data[i] * sqrt(v) * g->data[i], 0, scale);
    } else {                                  /* gaussian */
        double var = img_sc(pm, 0.01);
        matlab_mat *g = matlab_randn(A->rows, A->cols);
        for (int64_t i = 0; i < n; ++i) R->data[i] = img_clamp(A->data[i] + sqrt(var) * scale * g->data[i], 0, scale);
    }
    return R;
}

/* ===== Tier-3 — geometric transformations =============================== */

/* imresize(A, scale|[rows cols][, method]).  scale scalar = uniform; a
 * 1×2 vector gives the explicit output size.  Default method bicubic. */
matlab_mat *matlab_image_imresize(matlab_mat *A, matlab_mat *scalem, void *method_s) {
    if (!A) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    int64_t OH, OW;
    if (scalem && scalem->rows * scalem->cols >= 2) { OH = static_cast<int64_t>(scalem->data[0]); OW = static_cast<int64_t>(scalem->data[1]); }
    else { double s = img_sc(scalem, 1.0); OH = static_cast<int64_t>(floor(H * s + 0.5)); OW = static_cast<int64_t>(floor(W * s + 0.5)); }
    if (OH < 1) OH = 1; if (OW < 1) OW = 1;
    int method = img_method(method_s, 2);
    double sh = static_cast<double>(OH) / H, sw = static_cast<double>(OW) / W;
    return img_geo_apply(A, OH, OW, method, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        iy = (oy + 0.5) / sh - 0.5; ix = (ox + 0.5) / sw - 0.5;
    });
}
matlab_mat *matlab_image_imresize2(matlab_mat *A, matlab_mat *scalem) { return matlab_image_imresize(A, scalem, nullptr); }

/* imrotate(A, angle[, method][, bbox]).  bbox 'crop' keeps the input size;
 * default 'loose' fits the whole rotated image.  Default method nearest. */
matlab_mat *matlab_image_imrotate(matlab_mat *A, matlab_mat *anglem, void *method_s, void *bbox_s) {
    if (!A) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    double ang = img_sc(anglem, 0.0) * M_PI / 180.0;
    double ca = cos(ang), sa = sin(ang);
    int method = img_method(method_s, 0);
    std::string bb = img_sstr(bbox_s);
    bool crop = (bb.find("crop") != std::string::npos);
    int64_t OH, OW;
    if (crop) { OH = H; OW = W; }
    else {
        double cw = fabs(W * ca) + fabs(H * sa), ch = fabs(W * sa) + fabs(H * ca);
        OW = static_cast<int64_t>(ceil(cw)); OH = static_cast<int64_t>(ceil(ch));
    }
    double cy = (H - 1) / 2.0, cx = (W - 1) / 2.0;
    double ocy = (OH - 1) / 2.0, ocx = (OW - 1) / 2.0;
    return img_geo_apply(A, OH, OW, method, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        double dx = ox - ocx, dy = oy - ocy;
        ix = cx + ( ca * dx + sa * dy);
        iy = cy + (-sa * dx + ca * dy);
    });
}
matlab_mat *matlab_image_imrotate2(matlab_mat *A, matlab_mat *ang) { return matlab_image_imrotate(A, ang, nullptr, nullptr); }
matlab_mat *matlab_image_imrotate3(matlab_mat *A, matlab_mat *ang, void *m) { return matlab_image_imrotate(A, ang, m, nullptr); }

/* imcrop(A, [xmin ymin width height]) — 1-based rect, height+1 × width+1. */
matlab_mat *matlab_image_imcrop(matlab_mat *A, matlab_mat *rect) {
    if (!A || !rect || rect->rows * rect->cols < 4) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    int64_t x0 = static_cast<int64_t>(floor(rect->data[0] + 0.5)) - 1;
    int64_t y0 = static_cast<int64_t>(floor(rect->data[1] + 0.5)) - 1;
    int64_t ow = static_cast<int64_t>(floor(rect->data[2] + 0.5)) + 1;
    int64_t oh = static_cast<int64_t>(floor(rect->data[3] + 0.5)) + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x0 + ow > W) ow = W - x0; if (y0 + oh > H) oh = H - y0;
    if (ow < 1 || oh < 1) return mat_alloc(0, 0);
    return img_geo_apply(A, oh, ow, 0, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        iy = static_cast<double>(y0 + oy); ix = static_cast<double>(x0 + ox);
    });
}

/* imtranslate(A, [tx ty]) — same-size shift, bilinear, zero fill. */
matlab_mat *matlab_image_imtranslate(matlab_mat *A, matlab_mat *t) {
    if (!A) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    double tx = (t && t->rows * t->cols >= 1) ? t->data[0] : 0.0;
    double ty = (t && t->rows * t->cols >= 2) ? t->data[1] : 0.0;
    return img_geo_apply(A, H, W, 1, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        ix = ox - tx; iy = oy - ty;
    });
}

/* imwarp(A, tform) — affine (Kind 1) / projective (Kind 2).  Output is the
 * bounding box of the forward-transformed input; bilinear resampling. */
matlab_mat *matlab_image_imwarp(matlab_mat *A, matlab_obj *tform) {
    if (!A || !tform || !matlab_obj_is_known(tform)) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W);
    matlab_mat *Tm = matlab_obj_get_mat(tform, "T", 1);
    if (!Tm || Tm->rows * Tm->cols < 9) return mat_alloc(0, 0);
    double T[9]; for (int i = 0; i < 9; ++i) T[i] = Tm->data[i];
    int kind = static_cast<int>(matlab_obj_get_f64(tform, "Kind", 4));
    /* forward: [x y 1] * T (row-major).  x=col, y=row. */
    auto fwd = [&](double x, double y, double &xp, double &yp) {
        double u = T[0]*x + T[3]*y + T[6];
        double v = T[1]*x + T[4]*y + T[7];
        double w = T[2]*x + T[5]*y + T[8];
        if (kind == 2 && fabs(w) > 1e-300) { xp = u / w; yp = v / w; } else { xp = u; yp = v; }
    };
    double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
    double cx[4] = {0, static_cast<double>(W - 1), 0, static_cast<double>(W - 1)};
    double cy[4] = {0, 0, static_cast<double>(H - 1), static_cast<double>(H - 1)};
    for (int k = 0; k < 4; ++k) { double xp, yp; fwd(cx[k], cy[k], xp, yp);
        minx = std::min(minx, xp); maxx = std::max(maxx, xp); miny = std::min(miny, yp); maxy = std::max(maxy, yp); }
    int64_t OW = static_cast<int64_t>(floor(maxx - minx + 0.5)) + 1;
    int64_t OH = static_cast<int64_t>(floor(maxy - miny + 0.5)) + 1;
    if (OW < 1) OW = 1; if (OH < 1) OH = 1;
    double Tinv[9]; img_inv3x3(T, Tinv);
    return img_geo_apply(A, OH, OW, 1, [&](int64_t oy, int64_t ox, double &iy, double &ix) {
        double wx = ox + minx, wy = oy + miny;
        double u = wx*Tinv[0] + wy*Tinv[3] + Tinv[6];
        double v = wx*Tinv[1] + wy*Tinv[4] + Tinv[7];
        double w = wx*Tinv[2] + wy*Tinv[5] + Tinv[8];
        if (kind == 2 && fabs(w) > 1e-300) { ix = u / w; iy = v / w; } else { ix = u; iy = v; }
    });
}

/* fitgeotform2d(moving, fixed, type) — least-squares affine / similarity;
 * populates an already-allocated affine2d (alloc-then-populate). */
matlab_mat *matlab_image_fitgeo_init(matlab_obj *obj, matlab_mat *moving, matlab_mat *fixed, void *type_s) {
    if (!obj || !moving || !fixed) return mat_alloc(0, 0);
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    int64_t N = moving->rows;
    double T[9] = {1,0,0, 0,1,0, 0,0,1};
    if (t.find("similarity") != std::string::npos || t.find("rigid") != std::string::npos) {
        /* unknowns [a b tx ty]: fx = a*mx - b*my + tx ; fy = b*mx + a*my + ty. */
        std::vector<double> AtA(16, 0.0), Atb(4, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            double mx = moving->data[i*2+0], my = moving->data[i*2+1];
            double fx = fixed->data[i*2+0],  fy = fixed->data[i*2+1];
            double r1[4] = {mx, -my, 1, 0};   /* = fx */
            double r2[4] = {my,  mx, 0, 1};   /* = fy */
            for (int a = 0; a < 4; ++a) { for (int b = 0; b < 4; ++b) AtA[static_cast<size_t>(a*4+b)] += r1[a]*r1[b] + r2[a]*r2[b];
                Atb[static_cast<size_t>(a)] += r1[a]*fx + r2[a]*fy; }
        }
        std::vector<double> p = img_gauss(AtA, Atb, 4);
        T[0] = p[0]; T[1] = p[1]; T[3] = -p[1]; T[4] = p[0]; T[6] = p[2]; T[7] = p[3];
    } else {
        /* affine: [a c e] from (mx,my,1)->fx ; [b d f]->fy. */
        std::vector<double> AtA(9, 0.0), Atfx(3, 0.0), Atfy(3, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            double mx = moving->data[i*2+0], my = moving->data[i*2+1];
            double fx = fixed->data[i*2+0],  fy = fixed->data[i*2+1];
            double r[3] = {mx, my, 1};
            for (int a = 0; a < 3; ++a) { for (int b = 0; b < 3; ++b) AtA[static_cast<size_t>(a*3+b)] += r[a]*r[b];
                Atfx[static_cast<size_t>(a)] += r[a]*fx; Atfy[static_cast<size_t>(a)] += r[a]*fy; }
        }
        std::vector<double> px = img_gauss(AtA, Atfx, 3), py = img_gauss(AtA, Atfy, 3);
        T[0] = px[0]; T[3] = px[1]; T[6] = px[2];     /* a c e */
        T[1] = py[0]; T[4] = py[1]; T[7] = py[2];     /* b d f */
    }
    matlab_mat *Tm = mat_alloc(3, 3);
    for (int i = 0; i < 9; ++i) Tm->data[i] = T[i];
    matlab_obj_set_mat(obj, "T", 1, Tm);
    matlab_obj_set_f64(obj, "Kind", 4, 1.0);
    return mat_alloc(0, 0);
}

/* ===== Tier-4 — binarization + morphology =============================== */

/* Otsu threshold (between-class variance) over a 256-bin histogram → [0,1]. */
static double img_otsu(const double *hist, double total) {
    double sum = 0.0; for (int k = 0; k < 256; ++k) sum += k * hist[k];
    double varr[256]; for (int k = 0; k < 256; ++k) varr[k] = -1.0;
    double sumB = 0.0, wB = 0.0, maxVar = 0.0;
    for (int k = 0; k < 256; ++k) {
        wB += hist[k]; if (wB == 0.0) continue;
        double wF = total - wB; if (wF == 0.0) break;
        sumB += k * hist[k];
        double mB = sumB / wB, mF = (sum - sumB) / wF, var = wB * wF * (mB - mF) * (mB - mF);
        varr[k] = var; if (var > maxVar) maxVar = var;
    }
    /* average the threshold over the max-variance plateau (relative tol). */
    double thrSum = 0.0; int thrCnt = 0;
    for (int k = 0; k < 256; ++k) if (varr[k] >= maxVar * (1.0 - 1e-9) && varr[k] >= 0.0) { thrSum += k; thrCnt++; }
    return thrCnt ? (thrSum / thrCnt) / 255.0 : 0.0;
}
matlab_mat *matlab_image_graythresh_m(matlab_mat *A, double *outlevel) {
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    double hist[256] = {0};
    for (int64_t i = 0; i < n; ++i) hist[static_cast<int>(img_clamp(floor((norm ? 255.0 * A->data[i] : A->data[i]) + 0.5), 0, 255))]++;
    *outlevel = img_otsu(hist, static_cast<double>(n));
    return nullptr;
}
double matlab_image_graythresh(matlab_mat *A) { double lv; matlab_image_graythresh_m(A, &lv); return lv; }
double matlab_image_otsuthresh(matlab_mat *counts) {
    double hist[256] = {0}; double total = 0.0;
    int64_t n = counts ? counts->rows * counts->cols : 0;
    for (int64_t i = 0; i < n && i < 256; ++i) { hist[i] = counts->data[i]; total += counts->data[i]; }
    return img_otsu(hist, total);
}
/* imbinarize(A[, level]) — level in [0,1]; default Otsu.  Returns 0/1. */
matlab_mat *matlab_image_imbinarize2(matlab_mat *A, matlab_mat *lvm) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t n = A->rows * A->cols;
    bool norm = !img_is_u8range(A->data, n);
    double lv = lvm ? img_sc(lvm, 0.5) : matlab_image_graythresh(A);
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) { double x = norm ? A->data[i] : A->data[i] / 255.0; R->data[i] = (x >= lv) ? 1.0 : 0.0; }
    return R;
}
matlab_mat *matlab_image_imbinarize(matlab_mat *A) { return matlab_image_imbinarize2(A, nullptr); }

/* strel(type[, p1[, p2]]) — returns the neighborhood mask (0/1 matrix). */
matlab_mat *matlab_image_strel(void *type_s, matlab_mat *p1m, matlab_mat *p2m) {
    std::string t = img_sstr(type_s);
    for (char &c : t) c = static_cast<char>(tolower(c));
    if (t == "square") { int n = static_cast<int>(img_sc(p1m, 3)); matlab_mat *K = mat_alloc(n, n); for (int64_t i = 0; i < static_cast<int64_t>(n) * n; ++i) K->data[i] = 1.0; return K; }
    if (t == "rectangle") {
        int m = (p1m && p1m->rows * p1m->cols >= 2) ? static_cast<int>(p1m->data[0]) : 3;
        int n = (p1m && p1m->rows * p1m->cols >= 2) ? static_cast<int>(p1m->data[1]) : 3;
        matlab_mat *K = mat_alloc(m, n); for (int64_t i = 0; i < static_cast<int64_t>(m) * n; ++i) K->data[i] = 1.0; return K;
    }
    if (t == "line") {
        int len = static_cast<int>(img_sc(p1m, 3)); double deg = img_sc(p2m, 0.0) * M_PI / 180.0;
        int half = len / 2; int sz = 2 * half + 1; matlab_mat *K = mat_alloc(sz, sz);
        for (int d = -half; d <= half; ++d) { int x = static_cast<int>(floor(half + d * cos(deg) + 0.5)), y = static_cast<int>(floor(half - d * sin(deg) + 0.5));
            if (x >= 0 && x < sz && y >= 0 && y < sz) K->data[y * sz + x] = 1.0; }
        return K;
    }
    /* disk (default) */
    int r = static_cast<int>(img_sc(p1m, 3)); int sz = 2 * r + 1; matlab_mat *K = mat_alloc(sz, sz);
    for (int i = 0; i < sz; ++i) for (int j = 0; j < sz; ++j) { double dy = i - r, dx = j - r; if (dx * dx + dy * dy <= static_cast<double>(r) * r + r) K->data[i * sz + j] = 1.0; }
    return K;
}

/* grayscale/binary erosion (min) / dilation (max) over an SE mask. */
static matlab_mat *img_morph(matlab_mat *A, matlab_mat *SE, bool dilate) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    int64_t sm = SE ? SE->rows : 3, sn = SE ? SE->cols : 3;
    int64_t ar = (sm - 1) / 2, ac = (sn - 1) / 2;
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        double best = dilate ? -1e300 : 1e300; bool any = false;
        for (int64_t di = 0; di < sm; ++di) for (int64_t dj = 0; dj < sn; ++dj) {
            if (SE && SE->data[di * sn + dj] == 0.0) continue;
            int64_t iy = i + di - ar, ix = j + dj - ac;
            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
            double v = A->data[iy * W + ix]; best = dilate ? std::max(best, v) : std::min(best, v); any = true;
        }
        R->data[i * W + j] = any ? best : A->data[i * W + j];
    }
    return R;
}
matlab_mat *matlab_image_imerode(matlab_mat *A, matlab_mat *SE)  { return img_morph(A, SE, false); }
matlab_mat *matlab_image_imdilate(matlab_mat *A, matlab_mat *SE) { return img_morph(A, SE, true); }
matlab_mat *matlab_image_imopen(matlab_mat *A, matlab_mat *SE)   { matlab_mat *e = img_morph(A, SE, false); return img_morph(e, SE, true); }
matlab_mat *matlab_image_imclose(matlab_mat *A, matlab_mat *SE)  { matlab_mat *d = img_morph(A, SE, true); return img_morph(d, SE, false); }
matlab_mat *matlab_image_imtophat(matlab_mat *A, matlab_mat *SE) {
    matlab_mat *o = matlab_image_imopen(A, SE); int64_t n = A->rows * A->cols;
    matlab_mat *R = mat_alloc(A->rows, A->cols); for (int64_t i = 0; i < n; ++i) R->data[i] = A->data[i] - o->data[i]; return R;
}
matlab_mat *matlab_image_imbothat(matlab_mat *A, matlab_mat *SE) {
    matlab_mat *c = matlab_image_imclose(A, SE); int64_t n = A->rows * A->cols;
    matlab_mat *R = mat_alloc(A->rows, A->cols); for (int64_t i = 0; i < n; ++i) R->data[i] = c->data[i] - A->data[i]; return R;
}

/* imfill(BW,'holes') — fill background regions not connected to a border. */
matlab_mat *matlab_image_imfill(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols, N = H * W;
    std::vector<char> reach(static_cast<size_t>(N), 0);
    std::vector<int64_t> stack;
    auto push = [&](int64_t i, int64_t j) { if (i>=0&&i<H&&j>=0&&j<W) { int64_t p=i*W+j; if (!reach[static_cast<size_t>(p)] && A->data[p]==0.0) { reach[static_cast<size_t>(p)]=1; stack.push_back(p); } } };
    for (int64_t j = 0; j < W; ++j) { push(0, j); push(H - 1, j); }
    for (int64_t i = 0; i < H; ++i) { push(i, 0); push(i, W - 1); }
    while (!stack.empty()) { int64_t p = stack.back(); stack.pop_back(); int64_t i = p / W, j = p % W; push(i-1,j); push(i+1,j); push(i,j-1); push(i,j+1); }
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t p = 0; p < N; ++p) R->data[p] = (A->data[p] != 0.0 || !reach[static_cast<size_t>(p)]) ? 1.0 : 0.0;
    return R;
}

/* edge(A[, method]) — sobel (default) or canny.  Returns 0/1. */
static void img_sobel_grad(matlab_mat *A, std::vector<double> &gx, std::vector<double> &gy, int64_t H, int64_t W) {
    gx.assign(static_cast<size_t>(H * W), 0.0); gy.assign(static_cast<size_t>(H * W), 0.0);
    auto at = [&](int64_t i, int64_t j) { i = std::max<int64_t>(0, std::min(H - 1, i)); j = std::max<int64_t>(0, std::min(W - 1, j)); return A->data[i * W + j]; };
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        gx[static_cast<size_t>(i*W+j)] = (at(i-1,j+1)+2*at(i,j+1)+at(i+1,j+1)) - (at(i-1,j-1)+2*at(i,j-1)+at(i+1,j-1));
        gy[static_cast<size_t>(i*W+j)] = (at(i+1,j-1)+2*at(i+1,j)+at(i+1,j+1)) - (at(i-1,j-1)+2*at(i-1,j)+at(i-1,j+1));
    }
}
matlab_mat *matlab_image_edge(matlab_mat *A, void *method_s) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols, N = H * W;
    std::string m = img_sstr(method_s); for (char &c : m) c = static_cast<char>(tolower(c));
    std::vector<double> gx, gy; img_sobel_grad(A, gx, gy, H, W);
    std::vector<double> mag(static_cast<size_t>(N));
    double mmax = 0.0;
    for (int64_t p = 0; p < N; ++p) { mag[static_cast<size_t>(p)] = sqrt(gx[static_cast<size_t>(p)]*gx[static_cast<size_t>(p)] + gy[static_cast<size_t>(p)]*gy[static_cast<size_t>(p)]); mmax = std::max(mmax, mag[static_cast<size_t>(p)]); }
    matlab_mat *R = mat_alloc(H, W);
    if (m.find("canny") != std::string::npos) {
        double hi = 0.4 * mmax, lo = 0.4 * hi;
        std::vector<char> strong(static_cast<size_t>(N), 0), weak(static_cast<size_t>(N), 0);
        for (int64_t i = 1; i < H - 1; ++i) for (int64_t j = 1; j < W - 1; ++j) {
            int64_t p = i*W+j; double gxx = gx[static_cast<size_t>(p)], gyy = gy[static_cast<size_t>(p)], g = mag[static_cast<size_t>(p)];
            double ang = atan2(gyy, gxx); double a = fmod(ang + M_PI, M_PI) / M_PI * 4.0; int dir = static_cast<int>(floor(a + 0.5)) & 3;
            double n1, n2;
            if (dir == 0) { n1 = mag[static_cast<size_t>(p-1)]; n2 = mag[static_cast<size_t>(p+1)]; }
            else if (dir == 1) { n1 = mag[static_cast<size_t>(p-W+1)]; n2 = mag[static_cast<size_t>(p+W-1)]; }
            else if (dir == 2) { n1 = mag[static_cast<size_t>(p-W)]; n2 = mag[static_cast<size_t>(p+W)]; }
            else { n1 = mag[static_cast<size_t>(p-W-1)]; n2 = mag[static_cast<size_t>(p+W+1)]; }
            if (g >= n1 && g >= n2) { if (g >= hi) strong[static_cast<size_t>(p)] = 1; else if (g >= lo) weak[static_cast<size_t>(p)] = 1; }
        }
        std::vector<int64_t> st; for (int64_t p = 0; p < N; ++p) if (strong[static_cast<size_t>(p)]) { R->data[p] = 1.0; st.push_back(p); }
        while (!st.empty()) { int64_t p = st.back(); st.pop_back(); int64_t i = p/W, j = p%W;
            for (int64_t di = -1; di <= 1; ++di) for (int64_t dj = -1; dj <= 1; ++dj) { int64_t y=i+di,x=j+dj; if (y<0||y>=H||x<0||x>=W) continue; int64_t q=y*W+x;
                if (weak[static_cast<size_t>(q)] && R->data[q]==0.0) { R->data[q]=1.0; st.push_back(q); } } }
        return R;
    }
    double thr = 0.0; for (int64_t p = 0; p < N; ++p) thr += mag[static_cast<size_t>(p)]; thr = 4.0 * thr / N;  /* sobel auto-threshold */
    for (int64_t p = 0; p < N; ++p) R->data[p] = (mag[static_cast<size_t>(p)] >= thr) ? 1.0 : 0.0;
    return R;
}
matlab_mat *matlab_image_edge1(matlab_mat *A) { return matlab_image_edge(A, nullptr); }

/* ===== Tier-5 — segmentation + region analysis ========================== */

/* bwlabel(BW) — 8-connectivity labels via BFS; returns the label matrix. */
matlab_mat *matlab_image_bwlabel(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    matlab_mat *L = mat_alloc(H, W);
    double lab = 0.0;
    std::vector<int64_t> st;
    for (int64_t s = 0; s < H * W; ++s) {
        if (A->data[s] == 0.0 || L->data[s] != 0.0) continue;
        lab += 1.0; L->data[s] = lab; st.clear(); st.push_back(s);
        while (!st.empty()) { int64_t p = st.back(); st.pop_back(); int64_t i = p / W, j = p % W;
            for (int64_t di = -1; di <= 1; ++di) for (int64_t dj = -1; dj <= 1; ++dj) {
                if (!di && !dj) continue; int64_t y = i + di, x = j + dj; if (y<0||y>=H||x<0||x>=W) continue; int64_t q = y*W+x;
                if (A->data[q] != 0.0 && L->data[q] == 0.0) { L->data[q] = lab; st.push_back(q); } } }
    }
    return L;
}

/* regionprops(L, prop) — returns the property as a matrix (N×k). */
matlab_mat *matlab_image_regionprops(matlab_mat *L, void *prop_s) {
    if (!L || !L->data) return mat_alloc(0, 0);
    int64_t H = L->rows, W = L->cols;
    int N = 0; for (int64_t p = 0; p < H * W; ++p) N = std::max(N, static_cast<int>(L->data[p]));
    std::string pr = img_sstr(prop_s); for (char &c : pr) c = static_cast<char>(tolower(c));
    std::vector<double> area(static_cast<size_t>(N), 0.0), sx(static_cast<size_t>(N), 0.0), sy(static_cast<size_t>(N), 0.0);
    std::vector<double> sxx(static_cast<size_t>(N), 0.0), syy(static_cast<size_t>(N), 0.0), sxy(static_cast<size_t>(N), 0.0);
    std::vector<double> xmin(static_cast<size_t>(N), 1e18), xmax(static_cast<size_t>(N), -1e18), ymin(static_cast<size_t>(N), 1e18), ymax(static_cast<size_t>(N), -1e18);
    std::vector<double> perim(static_cast<size_t>(N), 0.0);
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        int lab = static_cast<int>(L->data[i*W+j]); if (lab < 1) continue; size_t k = static_cast<size_t>(lab - 1);
        double x = j + 1.0, y = i + 1.0;       /* 1-based pixel coords */
        area[k]++; sx[k] += x; sy[k] += y; sxx[k] += x*x; syy[k] += y*y; sxy[k] += x*y;
        if (x < xmin[k]) xmin[k] = x; if (x > xmax[k]) xmax[k] = x; if (y < ymin[k]) ymin[k] = y; if (y > ymax[k]) ymax[k] = y;
        bool border = false;
        for (int64_t di = -1; di <= 1 && !border; ++di) for (int64_t dj = -1; dj <= 1; ++dj) { int64_t yy=i+di,xx=j+dj;
            if (yy<0||yy>=H||xx<0||xx>=W || static_cast<int>(L->data[yy*W+xx]) != lab) { border = true; break; } }
        if (border) perim[k] += 1.0;
    }
    auto emit1 = [&](const std::vector<double> &v) { matlab_mat *R = mat_alloc(N, 1); for (int i = 0; i < N; ++i) R->data[i] = v[static_cast<size_t>(i)]; return R; };
    if (pr.find("area") != std::string::npos)       return emit1(area);
    if (pr.find("perimeter") != std::string::npos)  return emit1(perim);
    if (pr.find("centroid") != std::string::npos) {
        matlab_mat *R = mat_alloc(N, 2); for (int i = 0; i < N; ++i) { R->data[i*2] = sx[static_cast<size_t>(i)]/area[static_cast<size_t>(i)]; R->data[i*2+1] = sy[static_cast<size_t>(i)]/area[static_cast<size_t>(i)]; } return R;
    }
    if (pr.find("boundingbox") != std::string::npos) {
        matlab_mat *R = mat_alloc(N, 4); for (int i = 0; i < N; ++i) { R->data[i*4]=xmin[static_cast<size_t>(i)]-0.5; R->data[i*4+1]=ymin[static_cast<size_t>(i)]-0.5; R->data[i*4+2]=xmax[static_cast<size_t>(i)]-xmin[static_cast<size_t>(i)]+1; R->data[i*4+3]=ymax[static_cast<size_t>(i)]-ymin[static_cast<size_t>(i)]+1; } return R;
    }
    if (pr.find("equivdiameter") != std::string::npos) {
        matlab_mat *R = mat_alloc(N, 1); for (int i = 0; i < N; ++i) R->data[i] = 2.0 * sqrt(area[static_cast<size_t>(i)] / M_PI); return R;
    }
    if (pr.find("extent") != std::string::npos) {
        matlab_mat *R = mat_alloc(N, 1); for (int i = 0; i < N; ++i) { double bw=(xmax[static_cast<size_t>(i)]-xmin[static_cast<size_t>(i)]+1)*(ymax[static_cast<size_t>(i)]-ymin[static_cast<size_t>(i)]+1); R->data[i] = bw>0?area[static_cast<size_t>(i)]/bw:0; } return R;
    }
    /* axis lengths / eccentricity / orientation from 2nd central moments. */
    bool wantMaj = pr.find("majoraxis") != std::string::npos, wantMin = pr.find("minoraxis") != std::string::npos;
    bool wantEcc = pr.find("eccentric") != std::string::npos, wantOri = pr.find("orient") != std::string::npos;
    if (wantMaj || wantMin || wantEcc || wantOri) {
        matlab_mat *R = mat_alloc(N, 1);
        for (int i = 0; i < N; ++i) {
            double a = area[static_cast<size_t>(i)]; double mx = sx[static_cast<size_t>(i)]/a, my = sy[static_cast<size_t>(i)]/a;
            double uxx = sxx[static_cast<size_t>(i)]/a - mx*mx + 1.0/12.0, uyy = syy[static_cast<size_t>(i)]/a - my*my + 1.0/12.0, uxy = sxy[static_cast<size_t>(i)]/a - mx*my;
            double common = sqrt((uxx-uyy)*(uxx-uyy) + 4*uxy*uxy);
            double maj = 2.0*sqrt(2.0)*sqrt(uxx+uyy+common), mn = 2.0*sqrt(2.0)*sqrt(std::max(0.0, uxx+uyy-common));
            if (wantMaj) R->data[i] = maj;
            else if (wantMin) R->data[i] = mn;
            else if (wantEcc) R->data[i] = (maj>0) ? 2.0*sqrt(std::max(0.0,(maj/2)*(maj/2)-(mn/2)*(mn/2)))/maj : 0.0;
            else R->data[i] = -atan2(2*uxy, uxx-uyy) / 2.0 * 180.0 / M_PI;
        }
        return R;
    }
    return emit1(area);   /* default */
}

/* bwareaopen(BW, P) — drop connected components smaller than P pixels. */
matlab_mat *matlab_image_bwareaopen(matlab_mat *A, matlab_mat *Pm) {
    matlab_mat *L = matlab_image_bwlabel(A);
    int64_t H = A->rows, W = A->cols; int P = static_cast<int>(img_sc(Pm, 0));
    int N = 0; for (int64_t p = 0; p < H*W; ++p) N = std::max(N, static_cast<int>(L->data[p]));
    std::vector<int> cnt(static_cast<size_t>(N + 1), 0); for (int64_t p = 0; p < H*W; ++p) cnt[static_cast<size_t>(L->data[p])]++;
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t p = 0; p < H*W; ++p) { int lab = static_cast<int>(L->data[p]); R->data[p] = (lab >= 1 && cnt[static_cast<size_t>(lab)] >= P) ? 1.0 : 0.0; }
    return R;
}

/* bweuler(BW) — Euler number = #objects − #holes (8-conn objects). */
double matlab_image_bweuler(matlab_mat *A) {
    matlab_mat *L = matlab_image_bwlabel(A);
    int objs = 0; for (int64_t p = 0; p < A->rows * A->cols; ++p) objs = std::max(objs, static_cast<int>(L->data[p]));
    /* holes via imfill difference labelling */
    matlab_mat *F = matlab_image_imfill(A);
    int64_t holes_px = 0; for (int64_t p = 0; p < A->rows * A->cols; ++p) if (F->data[p] != 0.0 && A->data[p] == 0.0) holes_px++;
    matlab_mat *Hd = mat_alloc(A->rows, A->cols);
    for (int64_t p = 0; p < A->rows * A->cols; ++p) Hd->data[p] = (F->data[p] != 0.0 && A->data[p] == 0.0) ? 1.0 : 0.0;
    matlab_mat *HL = matlab_image_bwlabel(Hd); int holes = 0; for (int64_t p = 0; p < A->rows*A->cols; ++p) holes = std::max(holes, static_cast<int>(HL->data[p]));
    (void)holes_px;
    return static_cast<double>(objs - holes);
}

/* label2rgb(L) — colour each label; background (0) is white.  M×N×3. */
matlab_mat *matlab_image_label2rgb(matlab_mat *L) {
    if (!L || !L->data) return mat_alloc(0, 0);
    int64_t H = L->rows, W = L->cols, plane = H * W;
    matlab_mat3 *R = mat3_alloc(H, W, 3);
    for (int64_t p = 0; p < plane; ++p) {
        int lab = static_cast<int>(L->data[p]);
        if (lab == 0) { R->data[p] = 255; R->data[plane + p] = 255; R->data[2*plane + p] = 255; }
        else { unsigned h = static_cast<unsigned>(lab) * 2654435761u;
            R->data[p] = 60 + (h & 0xff) * 195 / 255; R->data[plane + p] = 60 + ((h >> 8) & 0xff) * 195 / 255; R->data[2*plane + p] = 60 + ((h >> 16) & 0xff) * 195 / 255; }
    }
    return reinterpret_cast<matlab_mat *>(R);
}

/* imsegkmeans(I, k) — k-means over pixel features; returns a label image. */
matlab_mat *matlab_image_imsegkmeans(matlab_mat *A, matlab_mat *km) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H, W; img_dims(A, H, W); int64_t n = H * W;
    int ch = mat_is_3d(A) ? static_cast<int>(reinterpret_cast<matlab_mat3 *>(A)->depth) : 1;
    matlab_mat *X = mat_alloc(n, ch);
    if (ch == 1) { for (int64_t i = 0; i < n; ++i) X->data[i] = A->data[i]; }
    else { matlab_mat3 *m = reinterpret_cast<matlab_mat3 *>(A); int64_t pl = H * W;
        for (int64_t i = 0; i < n; ++i) for (int c = 0; c < ch; ++c) X->data[i * ch + c] = m->data[c * pl + i]; }
    matlab_mat *idx = matlab_stats_kmeans(X, km);
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t i = 0; i < n; ++i) R->data[i] = idx->data[i];
    return R;
}

/* ===== Tier-6 — quality metrics ========================================= */

double matlab_image_immse(matlab_mat *A, matlab_mat *B) {
    if (!A || !B || !A->data || !B->data) return 0.0;
    int64_t n = A->rows * A->cols, m = B->rows * B->cols; if (m < n) n = m;
    double s = 0.0; for (int64_t i = 0; i < n; ++i) { double d = A->data[i] - B->data[i]; s += d * d; }
    return n ? s / n : 0.0;
}
double matlab_image_psnr(matlab_mat *A, matlab_mat *B) {
    double mse = matlab_image_immse(A, B);
    if (mse <= 0.0) return INFINITY;
    double peak = img_is_u8range(A->data, A->rows * A->cols) ? 255.0 : 1.0;
    return 10.0 * log10(peak * peak / mse);
}
/* ssim — windowed (8×8 box) mean structural similarity. */
double matlab_image_ssim(matlab_mat *A, matlab_mat *B) {
    if (!A || !B || !A->data || !B->data) return 0.0;
    int64_t H = A->rows, W = A->cols;
    double L = img_is_u8range(A->data, H * W) ? 255.0 : 1.0;
    double C1 = (0.01 * L) * (0.01 * L), C2 = (0.03 * L) * (0.03 * L);
    int win = 8; double total = 0.0; int64_t cnt = 0;
    for (int64_t i = 0; i + win <= H; i += win) for (int64_t j = 0; j + win <= W; j += win) {
        double ma = 0, mb = 0; int64_t N = win * win;
        for (int di = 0; di < win; ++di) for (int dj = 0; dj < win; ++dj) { ma += A->data[(i+di)*W+j+dj]; mb += B->data[(i+di)*W+j+dj]; }
        ma /= N; mb /= N;
        double va = 0, vb = 0, cov = 0;
        for (int di = 0; di < win; ++di) for (int dj = 0; dj < win; ++dj) { double a = A->data[(i+di)*W+j+dj]-ma, b = B->data[(i+di)*W+j+dj]-mb; va += a*a; vb += b*b; cov += a*b; }
        va /= (N-1); vb /= (N-1); cov /= (N-1);
        total += ((2*ma*mb+C1)*(2*cov+C2)) / ((ma*ma+mb*mb+C1)*(va+vb+C2)); cnt++;
    }
    return cnt ? total / cnt : 1.0;
}

/* ===== Tier-6 — colour-space conversions (M×N×3) ======================== */

matlab_mat *matlab_image_rgb2hsv(matlab_mat *A) {
    return img_color_apply(A, [](double r, double g, double b, double &h, double &s, double &v) {
        r /= 255; g /= 255; b /= 255;
        double mx = std::max(r, std::max(g, b)), mn = std::min(r, std::min(g, b)), d = mx - mn;
        v = mx; s = (mx > 0) ? d / mx : 0;
        if (d == 0) h = 0;
        else if (mx == r) h = fmod((g - b) / d, 6.0) / 6.0;
        else if (mx == g) h = ((b - r) / d + 2.0) / 6.0;
        else h = ((r - g) / d + 4.0) / 6.0;
        if (h < 0) h += 1.0;
    });
}
matlab_mat *matlab_image_hsv2rgb(matlab_mat *A) {
    return img_color_apply(A, [](double h, double s, double v, double &r, double &g, double &b) {
        double hh = h * 6.0; int i = static_cast<int>(floor(hh)) % 6; if (i < 0) i += 6;
        double f = hh - floor(hh), p = v*(1-s), q = v*(1-f*s), t = v*(1-(1-f)*s);
        double rr, gg, bb;
        switch (i) { case 0: rr=v; gg=t; bb=p; break; case 1: rr=q; gg=v; bb=p; break;
                     case 2: rr=p; gg=v; bb=t; break; case 3: rr=p; gg=q; bb=v; break;
                     case 4: rr=t; gg=p; bb=v; break; default: rr=v; gg=p; bb=q; }
        r = rr*255; g = gg*255; b = bb*255;
    });
}
matlab_mat *matlab_image_rgb2ycbcr(matlab_mat *A) {
    return img_color_apply(A, [](double r, double g, double b, double &y, double &cb, double &cr) {
        y  = 16  + ( 65.481*r + 128.553*g +  24.966*b) / 255.0;
        cb = 128 + (-37.797*r -  74.203*g + 112.000*b) / 255.0;
        cr = 128 + (112.000*r -  93.786*g -  18.214*b) / 255.0;
    });
}
matlab_mat *matlab_image_ycbcr2rgb(matlab_mat *A) {
    return img_color_apply(A, [](double y, double cb, double cr, double &r, double &g, double &b) {
        double yy = y - 16, u = cb - 128, v = cr - 128;
        r = img_clamp(1.164*yy + 1.596*v, 0, 255);
        g = img_clamp(1.164*yy - 0.392*u - 0.813*v, 0, 255);
        b = img_clamp(1.164*yy + 2.017*u, 0, 255);
    });
}
matlab_mat *matlab_image_rgb2lab(matlab_mat *A) {
    return img_color_apply(A, [](double r, double g, double b, double &L, double &aa, double &bb) {
        double rl = img_srgb2lin(r), gl = img_srgb2lin(g), bl = img_srgb2lin(b);
        double X = (0.4124*rl + 0.3576*gl + 0.1805*bl) / 0.95047;
        double Y = (0.2126*rl + 0.7152*gl + 0.0722*bl);
        double Z = (0.0193*rl + 0.1192*gl + 0.9505*bl) / 1.08883;
        double fx = img_labf(X), fy = img_labf(Y), fz = img_labf(Z);
        L = 116*fy - 16; aa = 500*(fx - fy); bb = 200*(fy - fz);
    });
}
matlab_mat *matlab_image_lab2rgb(matlab_mat *A) {
    return img_color_apply(A, [](double L, double aa, double bb, double &r, double &g, double &b) {
        double fy = (L + 16) / 116, fx = fy + aa / 500, fz = fy - bb / 200;
        double X = 0.95047 * img_labfi(fx), Y = img_labfi(fy), Z = 1.08883 * img_labfi(fz);
        double rl =  3.2406*X - 1.5372*Y - 0.4986*Z;
        double gl = -0.9689*X + 1.8758*Y + 0.0415*Z;
        double bl =  0.0557*X - 0.2040*Y + 1.0570*Z;
        r = img_lin2srgb(rl); g = img_lin2srgb(gl); b = img_lin2srgb(bl);
    });
}

/* ===== Tier-6 — transforms ============================================== */

/* dct2 / idct2 — separable orthonormal DCT-II / DCT-III (2-D). */
static matlab_mat *img_dct_axis(matlab_mat *A, bool inverse) {
    int64_t H = A->rows, W = A->cols;
    matlab_mat *R = mat_alloc(H, W);
    std::vector<double> col(static_cast<size_t>(H)), out(static_cast<size_t>(H));
    for (int64_t j = 0; j < W; ++j) {
        for (int64_t i = 0; i < H; ++i) col[static_cast<size_t>(i)] = A->data[i * W + j];
        for (int64_t k = 0; k < H; ++k) {
            double acc = 0.0;
            if (!inverse) {
                for (int64_t n = 0; n < H; ++n) acc += col[static_cast<size_t>(n)] * cos(M_PI * (2*n + 1) * k / (2.0 * H));
                acc *= sqrt((k == 0 ? 1.0 : 2.0) / H);
            } else {
                for (int64_t n = 0; n < H; ++n) { double ck = sqrt((n == 0 ? 1.0 : 2.0) / H); acc += ck * col[static_cast<size_t>(n)] * cos(M_PI * (2*k + 1) * n / (2.0 * H)); }
            }
            out[static_cast<size_t>(k)] = acc;
        }
        for (int64_t i = 0; i < H; ++i) R->data[i * W + j] = out[static_cast<size_t>(i)];
    }
    return R;
}
static matlab_mat *img_transpose(matlab_mat *A) {
    matlab_mat *R = mat_alloc(A->cols, A->rows);
    for (int64_t i = 0; i < A->rows; ++i) for (int64_t j = 0; j < A->cols; ++j) R->data[j * A->rows + i] = A->data[i * A->cols + j];
    return R;
}
matlab_mat *matlab_image_dct2(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    matlab_mat *c = img_dct_axis(A, false);       /* DCT over columns */
    matlab_mat *ct = img_transpose(c);
    matlab_mat *cr = img_dct_axis(ct, false);     /* DCT over rows */
    return img_transpose(cr);
}
matlab_mat *matlab_image_idct2(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    matlab_mat *c = img_dct_axis(A, true);
    matlab_mat *ct = img_transpose(c);
    matlab_mat *cr = img_dct_axis(ct, true);
    return img_transpose(cr);
}

/* radon(A, theta) — line-integral projections; sinogram (nrho × ntheta). */
matlab_mat *matlab_image_radon(matlab_mat *A, matlab_mat *thetam) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    int64_t nt = thetam ? thetam->rows * thetam->cols : 0; if (nt == 0) return mat_alloc(0, 0);
    int64_t diag = static_cast<int64_t>(ceil(sqrt(static_cast<double>(H * H + W * W)))) + 2;
    int64_t nrho = 2 * (diag / 2) + 1;
    double cy = (H - 1) / 2.0, cx = (W - 1) / 2.0, rho0 = (nrho - 1) / 2.0;
    matlab_mat *R = mat_alloc(nrho, nt);
    for (int64_t t = 0; t < nt; ++t) {
        double th = thetam->data[t] * M_PI / 180.0, ct = cos(th), st = sin(th);
        for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
            double rho = (j - cx) * ct + (i - cy) * st;
            int64_t bin = static_cast<int64_t>(floor(rho + rho0 + 0.5));
            if (bin >= 0 && bin < nrho) R->data[bin * nt + t] += A->data[i * W + j];
        }
    }
    return R;
}

/* hough(BW) — line accumulator over theta ∈ [-90,89]°, rho ∈ [-D,D]. */
matlab_mat *matlab_image_hough(matlab_mat *A) {
    if (!A || !A->data) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols;
    int nt = 180; int64_t D = static_cast<int64_t>(ceil(sqrt(static_cast<double>(H*H + W*W))));
    int64_t nrho = 2 * D + 1;
    matlab_mat *R = mat_alloc(nrho, nt);
    std::vector<double> cs(static_cast<size_t>(nt)), sn(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) { double th = (t - 90) * M_PI / 180.0; cs[static_cast<size_t>(t)] = cos(th); sn[static_cast<size_t>(t)] = sin(th); }
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        if (A->data[i * W + j] == 0.0) continue;
        for (int t = 0; t < nt; ++t) { int64_t rho = static_cast<int64_t>(floor(j * cs[static_cast<size_t>(t)] + i * sn[static_cast<size_t>(t)] + 0.5)) + D;
            if (rho >= 0 && rho < nrho) R->data[rho * nt + t] += 1.0; }
    }
    return R;
}
/* houghpeaks(H, numpeaks) — top-N accumulator cells, [rho_idx theta_idx] 1-based. */
matlab_mat *matlab_image_houghpeaks(matlab_mat *Hm, matlab_mat *nm) {
    if (!Hm || !Hm->data) return mat_alloc(0, 0);
    int np = static_cast<int>(img_sc(nm, 1)); if (np < 1) np = 1;
    int64_t R = Hm->rows, C = Hm->cols, N = R * C;
    std::vector<double> v(Hm->data, Hm->data + N);
    matlab_mat *P = mat_alloc(np, 2);
    for (int k = 0; k < np; ++k) {
        int64_t best = 0; double bv = -1.0;
        for (int64_t p = 0; p < N; ++p) if (v[static_cast<size_t>(p)] > bv) { bv = v[static_cast<size_t>(p)]; best = p; }
        P->data[k * 2 + 0] = static_cast<double>(best / C + 1); P->data[k * 2 + 1] = static_cast<double>(best % C + 1);
        /* suppress a small neighbourhood around the peak */
        int64_t pr = best / C, pc = best % C;
        for (int64_t dr = -2; dr <= 2; ++dr) for (int64_t dc = -2; dc <= 2; ++dc) { int64_t r = pr+dr, c = pc+dc; if (r>=0&&r<R&&c>=0&&c<C) v[static_cast<size_t>(r*C+c)] = -1.0; }
    }
    return P;
}

/* ===== Tier-6 — ROI ===================================================== */

/* poly2mask(x, y, M, N) — scanline polygon fill → M×N binary mask. */
matlab_mat *matlab_image_poly2mask(matlab_mat *xm, matlab_mat *ym, matlab_mat *Mm, matlab_mat *Nm) {
    if (!xm || !ym) return mat_alloc(0, 0);
    int64_t M = static_cast<int64_t>(img_sc(Mm, 0)), N = static_cast<int64_t>(img_sc(Nm, 0));
    int64_t nv = xm->rows * xm->cols;
    matlab_mat *R = mat_alloc(M, N);
    for (int64_t row = 0; row < M; ++row) {
        double yc = row + 1.0;                    /* 1-based pixel centre */
        std::vector<double> xs;
        for (int64_t k = 0; k < nv; ++k) {
            int64_t k2 = (k + 1) % nv;
            double y1 = ym->data[k], y2 = ym->data[k2], x1 = xm->data[k], x2 = xm->data[k2];
            if ((y1 <= yc && y2 > yc) || (y2 <= yc && y1 > yc))
                xs.push_back(x1 + (yc - y1) / (y2 - y1) * (x2 - x1));
        }
        std::sort(xs.begin(), xs.end());
        for (size_t a = 0; a + 1 < xs.size(); a += 2)
            for (int64_t col = static_cast<int64_t>(ceil(xs[a] - 1)); col <= static_cast<int64_t>(floor(xs[a+1] - 1)); ++col)
                if (col >= 0 && col < N) R->data[row * N + col] = 1.0;
    }
    return R;
}
/* roifilt2(h, A, mask) — filter A with h, keep filtered values inside mask. */
matlab_mat *matlab_image_roifilt2(matlab_mat *h, matlab_mat *A, matlab_mat *mask) {
    if (!A || !A->data) return mat_alloc(0, 0);
    matlab_mat *F = matlab_imfilter(A, h);
    int64_t n = A->rows * A->cols;
    matlab_mat *R = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < n; ++i) R->data[i] = (mask && i < mask->rows * mask->cols && mask->data[i] != 0.0) ? F->data[i] : A->data[i];
    return R;
}

/* ===== Tier-6 — block processing ======================================= */

/* im2col(A, [m n]) — sliding-window columns: each m×n window → a column. */
matlab_mat *matlab_image_im2col(matlab_mat *A, matlab_mat *bm) {
    if (!A || !A->data || !bm || bm->rows * bm->cols < 2) return mat_alloc(0, 0);
    int64_t H = A->rows, W = A->cols, m = static_cast<int64_t>(bm->data[0]), n = static_cast<int64_t>(bm->data[1]);
    int64_t cols = (H - m + 1) * (W - n + 1); if (cols < 0) cols = 0;
    matlab_mat *R = mat_alloc(m * n, cols);
    int64_t c = 0;
    for (int64_t j = 0; j <= W - n; ++j) for (int64_t i = 0; i <= H - m; ++i) {
        int64_t r = 0;
        for (int64_t dj = 0; dj < n; ++dj) for (int64_t di = 0; di < m; ++di) R->data[(r++) * cols + c] = A->data[(i+di) * W + (j+dj)];
        c++;
    }
    return R;
}
/* col2im(B, [m n], [M N]) — distinct-block reassembly. */
matlab_mat *matlab_image_col2im(matlab_mat *B, matlab_mat *bm, matlab_mat *sz) {
    if (!B || !B->data || !bm || !sz) return mat_alloc(0, 0);
    int64_t m = static_cast<int64_t>(bm->data[0]), n = static_cast<int64_t>(bm->data[1]);
    int64_t M = static_cast<int64_t>(sz->data[0]), N = static_cast<int64_t>(sz->data[1]);
    matlab_mat *R = mat_alloc(M, N);
    int64_t mb = M / m, nb = N / n, blkcols = B->cols, c = 0;
    for (int64_t bj = 0; bj < nb; ++bj) for (int64_t bi = 0; bi < mb; ++bi) {
        int64_t r = 0;
        for (int64_t dj = 0; dj < n; ++dj) for (int64_t di = 0; di < m; ++di) {
            if (c < blkcols && r < B->rows) R->data[(bi*m+di) * N + (bj*n+dj)] = B->data[(r) * blkcols + c];
            r++;
        }
        c++;
    }
    return R;
}

/* ===== Tier-6 — deblurring ============================================= */

/* deconvwnr(I, psf, nsr) — Wiener deconvolution via the 2-D FFT. */
matlab_mat *matlab_image_deconvwnr(matlab_mat *I, matlab_mat *psf, matlab_mat *nsrm) {
    if (!I || !I->data || !psf || !psf->data) return mat_alloc(0, 0);
    int64_t H = I->rows, W = I->cols;
    double nsr = img_sc(nsrm, 0.0);
    /* PSF padded to image size, centred at the origin (circular). */
    matlab_mat *P = mat_alloc(H, W);
    int64_t ph = psf->rows, pw = psf->cols;
    for (int64_t i = 0; i < ph; ++i) for (int64_t j = 0; j < pw; ++j) {
        int64_t r = ((i - ph / 2) % H + H) % H, c = ((j - pw / 2) % W + W) % W;
        P->data[r * W + c] = psf->data[i * pw + j];
    }
    matlab_mat_c *FI = matlab_fft2_c(I), *FP = matlab_fft2_c(P);
    matlab_mat_c *G = mat_c_alloc(H, W);
    for (int64_t k = 0; k < H * W; ++k) {
        double hr = FP->re[k], hi = FP->im[k], h2 = hr * hr + hi * hi;
        double wr = hr / (h2 + nsr), wi = -hi / (h2 + nsr);   /* conj(H)/(|H|²+nsr) */
        G->re[k] = wr * FI->re[k] - wi * FI->im[k];
        G->im[k] = wr * FI->im[k] + wi * FI->re[k];
    }
    matlab_mat_c *out = matlab_ifft2_c(G);
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t k = 0; k < H * W; ++k) R->data[k] = img_clamp(out->re[k], 0, 255);
    return R;
}
/* edgetaper(I, psf) — blur the borders to suppress ringing (weighted blend). */
matlab_mat *matlab_image_edgetaper(matlab_mat *I, matlab_mat *psf) {
    if (!I || !I->data) return mat_alloc(0, 0);
    matlab_mat *blur = matlab_imfilter(I, psf);
    int64_t H = I->rows, W = I->cols, b = 4;
    matlab_mat *R = mat_alloc(H, W);
    for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < W; ++j) {
        double dy = std::min<double>(i, H - 1 - i), dx = std::min<double>(j, W - 1 - j);
        double w = std::min(1.0, std::min(dy, dx) / static_cast<double>(b));
        R->data[i * W + j] = w * I->data[i * W + j] + (1 - w) * blur->data[i * W + j];
    }
    return R;
}

/* ----- multi-arity wrappers (pde_table matches one arity per entry) ------ */
matlab_mat *matlab_image_fspecial1(void *t)            { return matlab_image_fspecial(t, nullptr, nullptr); }
matlab_mat *matlab_image_fspecial2(void *t, matlab_mat *p1) { return matlab_image_fspecial(t, p1, nullptr); }
matlab_mat *matlab_image_checkerboard1(matlab_mat *n)  { return matlab_image_checkerboard(n, nullptr, nullptr); }
matlab_mat *matlab_image_checkerboard2(matlab_mat *n, matlab_mat *p) { return matlab_image_checkerboard(n, p, nullptr); }
matlab_mat *matlab_image_imnoise1(matlab_mat *A)       { return matlab_image_imnoise(A, nullptr, nullptr); }
matlab_mat *matlab_image_imnoise2(matlab_mat *A, void *t) { return matlab_image_imnoise(A, t, nullptr); }
matlab_mat *matlab_image_imgaussfilt1(matlab_mat *A)   { double v = 0.5; matlab_mat s; s.data = &v; s.rows = 1; s.cols = 1; return matlab_image_imgaussfilt(A, &s); }
matlab_mat *matlab_image_medfilt2_1(matlab_mat *A)     { return matlab_image_medfilt2(A, nullptr); }
matlab_mat *matlab_image_imboxfilt1(matlab_mat *A)     { return matlab_image_imboxfilt(A, nullptr); }
matlab_mat *matlab_image_strel1(void *t)               { return matlab_image_strel(t, nullptr, nullptr); }
matlab_mat *matlab_image_strel2(void *t, matlab_mat *p1) { return matlab_image_strel(t, p1, nullptr); }
matlab_mat *matlab_image_imfill2(matlab_mat *A, matlab_mat *opt) { (void)opt; return matlab_image_imfill(A); }
matlab_mat *matlab_image_deconvwnr2(matlab_mat *I, matlab_mat *psf) { return matlab_image_deconvwnr(I, psf, nullptr); }

}  /* extern "C" */
