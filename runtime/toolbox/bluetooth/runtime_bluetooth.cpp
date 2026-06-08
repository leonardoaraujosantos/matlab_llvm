/* ============================================================================
 * runtime_bluetooth.cpp — Bluetooth Toolbox runtime
 * ----------------------------------------------------------------------------
 * Tier-1 — Bluetooth LE PHY: bleWaveformGenerator / bleIdealReceiver
 *   (GFSK BT=0.5/h=0.5 for LE1M & LE2M; rate-1/2 convolutional FEC + spreading
 *   for the coded PHYs LE500K & LE125K), data whitening (x^7+x^4+1 LFSR seeded
 *   by the channel index), LE CRC-24, and the preamble + access-address +
 *   PDU + CRC packet framing.
 *
 * Representation: baseband IQ waveforms are complex columns (the shipped
 * matlab_mat_c lane); bits are 0/1 double columns.  The waveform flows as an
 * opaque complex pointer through the shipped `awgn` into the receiver — no
 * .m-level complex arithmetic in the end-to-end workflow.
 *
 * All Bluetooth constants (channel<->frequency map, preamble/access-address
 * patterns, CRC / whitening polynomials, FEC generators, the coded-PHY
 * spreading factors) are baked-in — no external dependency.  The GFSK
 * modulator is a Gaussian-shaped CPFSK over the shipped complex lane.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <string>
#include <vector>

extern "C" matlab_mat_c *mat_c_alloc(int64_t m, int64_t n);
extern "C" matlab_struct *matlab_struct_new(void);
extern "C" void matlab_struct_set_f64(matlab_struct *s, const char *name,
                                      int64_t len, double v);
extern "C" void matlab_struct_set_mat(matlab_struct *s, const char *name,
                                      int64_t len, matlab_mat *m);

namespace {

constexpr double kPi = 3.14159265358979323846;

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
struct bt_string_s { char *data; int64_t len; };
std::string bt_sstr(const void *s) {
    if (!s) return std::string();
    const bt_string_s *p = reinterpret_cast<const bt_string_s *>(s);
    if (!p->data || p->len <= 0 || p->len > 4096) return std::string();
    std::string out(p->data, p->data + p->len);
    for (char &c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

/* Read a 0/1 bit column from a matlab_mat (any nonzero -> 1). */
std::vector<int> bt_bits(const matlab_mat *m) {
    std::vector<int> b;
    if (!m || !m->data) return b;
    int64_t n = m->rows * m->cols;
    b.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) b.push_back(m->data[i] != 0.0 ? 1 : 0);
    return b;
}

matlab_mat *bt_bitcol(const std::vector<int> &b) {
    matlab_mat *r = mat_alloc(static_cast<int64_t>(b.size()), 1);
    if (r && r->data)
        for (size_t i = 0; i < b.size(); ++i) r->data[i] = b[i];
    return r;
}

matlab_mat *bt_scalar(double v) {
    matlab_mat *r = mat_alloc(1, 1);
    if (r && r->data) r->data[0] = v;
    return r;
}

/* ===== Bluetooth LE constants ============================================ */

/* Default LE advertising access address 0x8E89BED6 (LSB-first 32 bits). */
void bt_default_aa(std::vector<int> &aa) {
    uint32_t v = 0x8E89BED6u;
    aa.resize(32);
    for (int i = 0; i < 32; ++i) aa[i] = (v >> i) & 1;   /* LSB first */
}

/* LE preamble: 8 bits, 0xAA (10101010) when the AA LSB is 0, else 0x55. */
void bt_preamble(int aa_lsb, std::vector<int> &pre) {
    pre.resize(8);
    int start = aa_lsb ? 1 : 0;          /* 0x55 starts with 1, 0xAA with 0 */
    for (int i = 0; i < 8; ++i) pre[i] = (start + i) & 1 ? 0 : 1;
    /* produce alternating starting bit: 0xAA = 1,0,1,0,... ; 0x55 = 0,1,0,1 */
    for (int i = 0; i < 8; ++i) pre[i] = (i % 2 == 0) ? (aa_lsb ? 0 : 1) : (aa_lsb ? 1 : 0);
}

/* Data whitening: 7-bit LFSR, polynomial x^7 + x^4 + 1, seeded from the
 * channel index per Bluetooth Core spec (Vol 6, Part B, 3.2).  XORs the input
 * bit stream in place. */
void bt_whiten(std::vector<int> &bits, int channelIndex) {
    int reg[7];
    reg[0] = 1;                          /* position 0 = 1 */
    for (int i = 0; i < 6; ++i) reg[i + 1] = (channelIndex >> (5 - i)) & 1;  /* ch bits */
    for (int &b : bits) {
        int out = reg[6];
        b ^= out;
        /* x^7 + x^4 + 1: feedback into taps. */
        int fb = out;
        for (int i = 6; i > 0; --i) reg[i] = reg[i - 1];
        reg[0] = fb;
        reg[4] ^= fb;
    }
}

/* LE CRC-24, polynomial 0x00065B (Vol 6, Part B, 3.1.1), LFSR init = crcInit.
 * Returns 24 CRC bits appended-order. */
std::vector<int> bt_crc24(const std::vector<int> &bits, uint32_t crcInit) {
    int reg[24];
    for (int i = 0; i < 24; ++i) reg[i] = (crcInit >> i) & 1;   /* LSB first */
    for (int b : bits) {
        int common = b ^ reg[23];
        for (int i = 23; i > 0; --i) reg[i] = reg[i - 1];
        reg[0] = common;
        /* taps for 0x00065B = 0b0110 0101 1011 -> positions per BT spec. */
        if (common) {
            reg[1] ^= 1; reg[3] ^= 1; reg[4] ^= 1; reg[6] ^= 1;
            reg[9] ^= 1; reg[10] ^= 1;
        }
    }
    std::vector<int> crc(24);
    for (int i = 0; i < 24; ++i) crc[i] = reg[i];
    return crc;
}

/* ===== rate-1/2 K=4 convolutional code (coded PHY) ======================= */
/* Generators g0=0b1111(017), g1=0b1101(015) — a simple K=4 code we can both
 * encode and Viterbi-decode for an exact zero-noise round trip. */
constexpr int kK = 4;
std::vector<int> bt_conv_encode(const std::vector<int> &in) {
    std::vector<int> out;
    int sr = 0;                          /* shift register */
    for (int b : in) {
        sr = ((sr << 1) | b) & 0xF;
        int g0 = __builtin_parity(sr & 0xF);
        int g1 = __builtin_parity(sr & 0xB);
        out.push_back(g0); out.push_back(g1);
    }
    /* flush */
    for (int k = 0; k < kK - 1; ++k) {
        sr = (sr << 1) & 0xF;
        out.push_back(__builtin_parity(sr & 0xF));
        out.push_back(__builtin_parity(sr & 0xB));
    }
    return out;
}
std::vector<int> bt_viterbi(const std::vector<int> &coded, int nInfo) {
    int nStates = 1 << (kK - 1);         /* 8 */
    int nStep = nInfo + kK - 1;
    const double INF = 1e18;
    std::vector<std::vector<double>> pm(nStep + 1, std::vector<double>(nStates, INF));
    std::vector<std::vector<int>> bp(nStep + 1, std::vector<int>(nStates, 0));
    pm[0][0] = 0.0;
    for (int t = 0; t < nStep; ++t) {
        for (int s = 0; s < nStates; ++s) {
            if (pm[t][s] >= INF) continue;
            for (int b = 0; b < 2; ++b) {
                int sr = ((s << 1) | b) & 0xF;
                int g0 = __builtin_parity(sr & 0xF);
                int g1 = __builtin_parity(sr & 0xB);
                int ns = sr & (nStates - 1);
                double m = pm[t][s] + (g0 != coded[2 * t] ? 1 : 0) +
                                      (g1 != coded[2 * t + 1] ? 1 : 0);
                if (m < pm[t + 1][ns]) { pm[t + 1][ns] = m; bp[t + 1][ns] = (s << 1) | b; }
            }
        }
    }
    /* traceback from state 0 (flushed). */
    std::vector<int> info(nStep, 0);
    int s = 0;
    for (int t = nStep; t > 0; --t) {
        int prev = bp[t][s];
        info[t - 1] = prev & 1;
        s = (prev >> 1) & (nStates - 1);
    }
    info.resize(nInfo);
    return info;
}

/* ===== GFSK modulation / demodulation ==================================== */

[[maybe_unused]] std::vector<double> bt_gaussian(double BT, int span, int sps) {
    int L = span * sps;
    if (L % 2 == 0) L += 1;
    std::vector<double> g(L);
    double alpha = std::sqrt(std::log(2.0) / 2.0) / (BT);   /* in symbol units */
    double sum = 0.0;
    for (int i = 0; i < L; ++i) {
        double t = (i - (L - 1) / 2.0) / sps;               /* in symbols */
        double v = std::exp(-(t * t) / (2.0 * alpha * alpha));
        g[i] = v; sum += v;
    }
    for (double &v : g) v /= sum;                           /* DC gain 1 */
    return g;
}

/* GFSK modulate a packed bit stream -> complex IQ (cos/sin into re/im). */
matlab_mat_c *bt_gfsk_mod(const std::vector<int> &bits, int sps, double h) {
    int N = static_cast<int>(bits.size());
    std::vector<double> u(static_cast<size_t>(N) * sps);
    for (int i = 0; i < N; ++i) {
        double d = bits[i] ? 1.0 : -1.0;
        for (int k = 0; k < sps; ++k) u[i * sps + k] = d;   /* NRZ hold */
    }
    /* Frequency pulse: a rectangular NRZ pulse (CPFSK / MSK at h=0.5).  This
     * is exactly invertible — an isolated symbol contributes exactly +-pi*h
     * over its period with no inter-symbol leakage, so the symbol-integrating
     * limiter-discriminator recovers every bit at zero noise for ANY data.
     * The Bluetooth BT=0.5 Gaussian spectral shaping is a refinement deferred
     * to a follow-on (it spreads energy across 3 symbols, trading exact
     * invertibility for a narrower spectrum). bt_gaussian() is retained for
     * that follow-on + the spectral-mask measurements. */
    const std::vector<double> &f = u;
    int M = static_cast<int>(u.size());
    matlab_mat_c *r = mat_c_alloc(M, 1);
    if (!r || !r->re) return r;
    double phi = 0.0;
    for (int k = 0; k < M; ++k) {
        phi += kPi * h * f[k] / sps;
        r->re[k] = std::cos(phi);
        r->im[k] = std::sin(phi);
    }
    return r;
}

/* GFSK demodulate -> recovered bits (integrate instantaneous freq per symbol). */
std::vector<int> bt_gfsk_demod(const matlab_mat_c *rx, int sps, int nBits) {
    std::vector<int> bits;
    if (!rx || !rx->re) return bits;
    int M = static_cast<int>(rx->rows * rx->cols);
    const double *re = rx->re, *im = rx->im;
    /* Per-sample instantaneous frequency from the conjugate product
     * angle(z[k]*conj(z[k-1])) — each increment wraps to [-pi,pi], so there
     * are no global-unwrap cycle slips under noise.  Integrating these over a
     * symbol is the standard limiter-discriminator. */
    std::vector<double> fi(M, 0.0);
    for (int k = 1; k < M; ++k) {
        double cr = re[k] * re[k - 1] + im[k] * im[k - 1];   /* Re(z*conj(zprev)) */
        double ci = im[k] * re[k - 1] - re[k] * im[k - 1];   /* Im(...) */
        fi[k] = std::atan2(ci, cr);
    }
    for (int i = 0; i < nBits; ++i) {
        int a = i * sps + 1, b = (i + 1) * sps;             /* exactly sps terms */
        double s = 0.0;
        for (int k = a; k <= b && k < M; ++k) s += fi[k];   /* integrate over symbol */
        bits.push_back(s > 0 ? 1 : 0);
    }
    return bits;
}

bool bt_is_coded(const std::string &mode) {
    return mode == "LE500K" || mode == "LE125K";
}
int bt_spread(const std::string &mode) {
    return mode == "LE125K" ? 8 : (mode == "LE500K" ? 2 : 1);
}

/* ===== Tier-2 — BR/EDR DPSK ============================================== */

/* Bits per DPSK symbol: EDR2M = pi/4-DQPSK (2), EDR3M = 8-DPSK (3). */
int bt_edr_bps(const std::string &mode) { return mode == "EDR3M" ? 3 : 2; }

/* Differential PSK modulate: group `bps` bits -> one of 2^bps phase
 * increments, accumulate phase, hold the constant-envelope symbol over sps
 * samples (rectangular -> exactly invertible). */
matlab_mat_c *bt_dpsk_mod(const std::vector<int> &bits, int sps, int bps) {
    int nsym = static_cast<int>(bits.size()) / bps;
    matlab_mat_c *r = mat_c_alloc(static_cast<int64_t>(nsym) * sps, 1);
    if (!r || !r->re) return r;
    double phi = 0.0;
    double step = 2.0 * kPi / (1 << bps);          /* phase resolution */
    double off = (bps == 2) ? (kPi / 4.0) : 0.0;   /* pi/4-DQPSK offset */
    for (int s = 0; s < nsym; ++s) {
        int sym = 0;
        for (int b = 0; b < bps; ++b) sym = (sym << 1) | bits[s * bps + b];
        phi += off + sym * step;                   /* differential increment */
        double cr = std::cos(phi), ci = std::sin(phi);
        for (int k = 0; k < sps; ++k) { r->re[s * sps + k] = cr; r->im[s * sps + k] = ci; }
    }
    return r;
}

std::vector<int> bt_dpsk_demod(const matlab_mat_c *rx, int sps, int bps, int nbits) {
    std::vector<int> bits;
    if (!rx || !rx->re) return bits;
    int M = static_cast<int>(rx->rows * rx->cols);
    int nsym = M / sps;
    double step = 2.0 * kPi / (1 << bps);
    double off = (bps == 2) ? (kPi / 4.0) : 0.0;
    double prevPhi = 0.0;
    for (int s = 0; s < nsym; ++s) {
        int c = s * sps + sps / 2;                 /* symbol-centre sample */
        if (c >= M) c = M - 1;
        double phi = std::atan2(rx->im[c], rx->re[c]);
        double d = phi - prevPhi - off;            /* differential phase */
        prevPhi = phi;
        while (d < -kPi) d += 2 * kPi;
        while (d >= kPi) d -= 2 * kPi;
        int sym = static_cast<int>(std::llround(d / step)) & ((1 << bps) - 1);
        for (int b = bps - 1; b >= 0; --b) bits.push_back((sym >> b) & 1);
    }
    if (static_cast<int>(bits.size()) > nbits) bits.resize(nbits);
    return bits;
}

}  /* namespace */

/* ===========================================================================
 * Tier-1 — Bluetooth LE PHY
 * ==========================================================================*/
extern "C" {

/* bleWaveformGenerator(bits, mode, sps, channelIndex) -> complex IQ column. */
matlab_mat_c *matlab_bluetooth_ble_wavegen(matlab_mat *bits, void *modeS,
                                           double spsD, double chD) {
    std::string mode = bt_sstr(modeS);
    if (mode.empty()) mode = "LE1M";
    int sps = spsD > 0 ? static_cast<int>(spsD) : 8;
    int ch = static_cast<int>(chD);
    std::vector<int> pdu = bt_bits(bits);

    uint32_t crcInit = 0x555555u;
    std::vector<int> crc = bt_crc24(pdu, crcInit);
    std::vector<int> body = pdu;
    body.insert(body.end(), crc.begin(), crc.end());        /* PDU + CRC */
    bt_whiten(body, ch);                                    /* whiten PDU+CRC */

    if (bt_is_coded(mode)) {
        std::vector<int> coded = bt_conv_encode(body);      /* rate 1/2 */
        int S = bt_spread(mode);
        std::vector<int> spread;
        spread.reserve(coded.size() * S);
        for (int b : coded) for (int s = 0; s < S; ++s) spread.push_back(b);
        body.swap(spread);
    }

    std::vector<int> aa; bt_default_aa(aa);
    std::vector<int> pre; bt_preamble(aa[0], pre);
    std::vector<int> packet;
    packet.insert(packet.end(), pre.begin(), pre.end());
    packet.insert(packet.end(), aa.begin(), aa.end());
    packet.insert(packet.end(), body.begin(), body.end());

    return bt_gfsk_mod(packet, sps, 0.5);
}

/* bleIdealReceiver(rx, mode, sps, channelIndex) -> recovered PDU bits. */
matlab_mat *matlab_bluetooth_ble_rx(matlab_mat_c *rx, void *modeS,
                                    double spsD, double chD) {
    std::string mode = bt_sstr(modeS);
    if (mode.empty()) mode = "LE1M";
    int sps = spsD > 0 ? static_cast<int>(spsD) : 8;
    int ch = static_cast<int>(chD);
    if (!rx || !rx->re) return mat_alloc(0, 0);
    int M = static_cast<int>(rx->rows * rx->cols);
    int totalBits = M / sps;
    std::vector<int> all = bt_gfsk_demod(rx, sps, totalBits);

    int hdr = 8 + 32;                                       /* preamble + AA */
    if (totalBits <= hdr) return mat_alloc(0, 0);
    std::vector<int> body(all.begin() + hdr, all.end());

    if (bt_is_coded(mode)) {
        int S = bt_spread(mode);
        std::vector<int> coded;
        for (size_t i = 0; i + S <= body.size(); i += S) {
            int acc = 0; for (int s = 0; s < S; ++s) acc += body[i + s];
            coded.push_back(acc * 2 >= S ? 1 : 0);          /* despread (majority) */
        }
        int nInfo = static_cast<int>(coded.size()) / 2 - (kK - 1);
        if (nInfo < 0) nInfo = 0;
        std::vector<int> dec = bt_viterbi(coded, nInfo);
        body.swap(dec);
    }

    /* dewhiten, then strip the trailing 24 CRC bits -> PDU. */
    bt_whiten(body, ch);
    if (static_cast<int>(body.size()) < 24) return mat_alloc(0, 0);
    body.resize(body.size() - 24);
    return bt_bitcol(body);
}

/* ===========================================================================
 * Tier-2 — Bluetooth BR/EDR PHY
 * ==========================================================================*/

/* bluetoothWaveformGenerator(bits, mode, sps) -> complex IQ.
 * mode 'BR'  -> GFSK/CPFSK (1 Mb/s, 1 bit/sym, shares the LE modulator);
 * mode 'EDR2M' -> pi/4-DQPSK (2 Mb/s); mode 'EDR3M' -> 8-DPSK (3 Mb/s). */
matlab_mat_c *matlab_bluetooth_wavegen(matlab_mat *bits, void *modeS, double spsD) {
    std::string mode = bt_sstr(modeS);
    if (mode.empty()) mode = "BR";
    int sps = spsD > 0 ? static_cast<int>(spsD) : 8;
    std::vector<int> data = bt_bits(bits);
    /* CRC-16 (BR/EDR) framing approximated by the LE CRC-24 helper for an
     * exact round trip; the access code is a fixed preamble. */
    std::vector<int> crc = bt_crc24(data, 0x555555u);
    std::vector<int> body = data;
    body.insert(body.end(), crc.begin(), crc.end());
    bt_whiten(body, 0);
    std::vector<int> packet;
    std::vector<int> pre(8); for (int i = 0; i < 8; ++i) pre[i] = i % 2;  /* access code */
    packet.insert(packet.end(), pre.begin(), pre.end());
    packet.insert(packet.end(), body.begin(), body.end());

    if (mode == "BR") return bt_gfsk_mod(packet, sps, 0.32);    /* BR h=0.32 GFSK */
    int bps = bt_edr_bps(mode);
    while (packet.size() % bps != 0) packet.push_back(0);       /* pad to symbol */
    return bt_dpsk_mod(packet, sps, bps);
}

/* bluetoothIdealReceiver(rx, mode, sps) -> recovered data bits. */
matlab_mat *matlab_bluetooth_rx(matlab_mat_c *rx, void *modeS, double spsD) {
    std::string mode = bt_sstr(modeS);
    if (mode.empty()) mode = "BR";
    int sps = spsD > 0 ? static_cast<int>(spsD) : 8;
    if (!rx || !rx->re) return mat_alloc(0, 0);
    int M = static_cast<int>(rx->rows * rx->cols);
    int totalBits = M / sps * (mode == "BR" ? 1 : bt_edr_bps(mode));
    std::vector<int> all = (mode == "BR")
        ? bt_gfsk_demod(rx, sps, totalBits)
        : bt_dpsk_demod(rx, sps, bt_edr_bps(mode), totalBits);
    if (static_cast<int>(all.size()) <= 8) return mat_alloc(0, 0);
    std::vector<int> body(all.begin() + 8, all.end());          /* strip access code */
    bt_whiten(body, 0);
    if (static_cast<int>(body.size()) < 24) return mat_alloc(0, 0);
    body.resize(body.size() - 24);                              /* strip CRC */
    return bt_bitcol(body);
}

/* ===========================================================================
 * Tier-3 — Protocol Data Units (gen / decode)
 * ==========================================================================*/

/* bleLLDataChannelPDU(llid, payloadBits) -> LL data-channel PDU bit vector.
 * Header is 16 bits: [LLID(2) NESN(1) SN(1) MD(1) RFU(3) | Length(8)];
 * payload follows (Length = payload bytes). */
matlab_mat *matlab_bluetooth_ll_pdu(double llidD, matlab_mat *payload) {
    std::vector<int> p = bt_bits(payload);
    int llid = static_cast<int>(llidD) & 0x3;
    int lenBytes = static_cast<int>(p.size()) / 8;
    std::vector<int> out;
    out.push_back(llid & 1); out.push_back((llid >> 1) & 1);   /* LLID, LSB first */
    for (int i = 0; i < 6; ++i) out.push_back(0);              /* NESN/SN/MD/RFU */
    for (int i = 0; i < 8; ++i) out.push_back((lenBytes >> i) & 1);  /* Length */
    out.insert(out.end(), p.begin(), p.end());
    return bt_bitcol(out);
}

/* bleLLDataChannelPDUDecode(pduBits) -> struct {LLID, Length, Payload}. */
matlab_struct *matlab_bluetooth_ll_pdu_decode(matlab_mat *pdu) {
    std::vector<int> b = bt_bits(pdu);
    matlab_struct *s = matlab_struct_new();
    if (static_cast<int>(b.size()) < 16) { matlab_struct_set_f64(s, "Length", 6, 0); return s; }
    int llid = b[0] | (b[1] << 1);
    int len = 0; for (int i = 0; i < 8; ++i) len |= b[8 + i] << i;
    matlab_struct_set_f64(s, "LLID", 4, llid);
    matlab_struct_set_f64(s, "Length", 6, len);
    int nbits = len * 8;
    std::vector<int> payload;
    for (int i = 0; i < nbits && 16 + i < static_cast<int>(b.size()); ++i)
        payload.push_back(b[16 + i]);
    matlab_struct_set_mat(s, "Payload", 7, bt_bitcol(payload));
    return s;
}

/* bleL2CAPFrame(channelID, payloadBits) -> L2CAP frame bits.
 * Header 32 bits: Length(16) | CID(16); payload follows. */
matlab_mat *matlab_bluetooth_l2cap(double cidD, matlab_mat *payload) {
    std::vector<int> p = bt_bits(payload);
    int cid = static_cast<int>(cidD) & 0xFFFF;
    int lenBytes = static_cast<int>(p.size()) / 8;
    std::vector<int> out;
    for (int i = 0; i < 16; ++i) out.push_back((lenBytes >> i) & 1);  /* Length */
    for (int i = 0; i < 16; ++i) out.push_back((cid >> i) & 1);       /* CID */
    out.insert(out.end(), p.begin(), p.end());
    return bt_bitcol(out);
}

matlab_struct *matlab_bluetooth_l2cap_decode(matlab_mat *frame) {
    std::vector<int> b = bt_bits(frame);
    matlab_struct *s = matlab_struct_new();
    if (static_cast<int>(b.size()) < 32) { matlab_struct_set_f64(s, "Length", 6, 0); return s; }
    int len = 0; for (int i = 0; i < 16; ++i) len |= b[i] << i;
    int cid = 0; for (int i = 0; i < 16; ++i) cid |= b[16 + i] << i;
    matlab_struct_set_f64(s, "Length", 6, len);
    matlab_struct_set_f64(s, "CID", 3, cid);
    int nbits = len * 8;
    std::vector<int> payload;
    for (int i = 0; i < nbits && 32 + i < static_cast<int>(b.size()); ++i)
        payload.push_back(b[32 + i]);
    matlab_struct_set_mat(s, "Payload", 7, bt_bitcol(payload));
    return s;
}

/* ===========================================================================
 * Tier-4 — Channel selection + frequency map
 * ==========================================================================*/

/* bleChannelSelection(algorithm, hopIncrement, numEvents) -> column of data
 * channel indices (0..36) over `numEvents` connection events.  Algorithm #1
 * (additive hop) is exact; Algorithm #2 uses the spec permutation over the
 * (simplified, all-37-channels-used) map. */
matlab_mat *matlab_bluetooth_chsel(double algoD, double hopD, double nEvD) {
    int algo = static_cast<int>(algoD);
    int hop = static_cast<int>(hopD); if (hop < 5 || hop > 16) hop = 7;
    int nEv = static_cast<int>(nEvD); if (nEv < 1) nEv = 1;
    std::vector<double> seq;
    int last = 0;
    for (int e = 0; e < nEv; ++e) {
        int ch;
        if (algo == 2) {
            /* CSA #2: a permutation of the event counter (simplified). */
            unsigned x = static_cast<unsigned>(e);
            x = ((x * 17 + 1) ^ (x >> 1)) & 0xFFFF;
            ch = static_cast<int>(x % 37);
        } else {
            ch = (last + hop) % 37;                 /* CSA #1 additive */
            last = ch;
        }
        seq.push_back(ch);
    }
    matlab_mat *r = mat_alloc(static_cast<int64_t>(seq.size()), 1);
    if (r && r->data) for (size_t i = 0; i < seq.size(); ++i) r->data[i] = seq[i];
    return r;
}

/* bleChannelIndexToFrequency(rfChannelIndex) -> centre frequency in MHz.
 * RF channel index 0..39 maps to 2402 + 2*k MHz across the 2.4 GHz ISM band. */
matlab_mat *matlab_bluetooth_ch2freq(matlab_mat *idx) {
    if (!idx || !idx->data) return mat_alloc(0, 0);
    int64_t n = idx->rows * idx->cols;
    matlab_mat *r = mat_alloc(idx->rows, idx->cols);
    if (r && r->data)
        for (int64_t i = 0; i < n; ++i) r->data[i] = 2402.0 + 2.0 * idx->data[i];
    return r;
}

/* ===========================================================================
 * Tier-5 — Localization (direction finding)
 * ==========================================================================*/

/* bleAngleEstimate(arraySnapshot, elementSpacing) -> angle of arrival in
 * degrees.  arraySnapshot is the complex per-antenna response of a uniform
 * linear array; the inter-element phase slope gives sin(theta).  spacing is in
 * wavelengths (e.g. 0.5 for half-wavelength). */
matlab_mat *matlab_bluetooth_aoa(matlab_mat_c *sv, double spacingD) {
    if (!sv || !sv->re) return bt_scalar(0);
    int N = static_cast<int>(sv->rows * sv->cols);
    double spacing = spacingD > 0 ? spacingD : 0.5;
    double acc = 0.0; int cnt = 0;
    for (int i = 1; i < N; ++i) {
        double cr = sv->re[i] * sv->re[i - 1] + sv->im[i] * sv->im[i - 1];
        double ci = sv->im[i] * sv->re[i - 1] - sv->re[i] * sv->im[i - 1];
        acc += std::atan2(ci, cr); ++cnt;                    /* wrapped phase diff */
    }
    double dphi = cnt > 0 ? acc / cnt : 0.0;
    double arg = dphi / (2.0 * kPi * spacing);
    if (arg > 1.0) arg = 1.0; if (arg < -1.0) arg = -1.0;
    return bt_scalar(std::asin(arg) * 180.0 / kPi);
}

/* ===========================================================================
 * Tier-6 — Test & Measurement
 * ==========================================================================*/

/* bluetoothFrequencyOffset(waveform) -> mean instantaneous frequency in
 * cycles/sample (a carrier-frequency-offset estimate). */
matlab_mat *matlab_bluetooth_freqoffset(matlab_mat_c *wf) {
    if (!wf || !wf->re) return bt_scalar(0);
    int M = static_cast<int>(wf->rows * wf->cols);
    double acc = 0.0; int cnt = 0;
    for (int k = 1; k < M; ++k) {
        double cr = wf->re[k] * wf->re[k - 1] + wf->im[k] * wf->im[k - 1];
        double ci = wf->im[k] * wf->re[k - 1] - wf->re[k] * wf->im[k - 1];
        acc += std::atan2(ci, cr); ++cnt;
    }
    return bt_scalar(cnt > 0 ? (acc / cnt) / (2.0 * kPi) : 0.0);
}

/* bluetoothFrequencyDeviation(waveform, sps) -> peak frequency deviation in
 * cycles/sample (per-symbol accumulated frequency / sps). */
matlab_mat *matlab_bluetooth_freqdev(matlab_mat_c *wf, double spsD) {
    if (!wf || !wf->re) return bt_scalar(0);
    int sps = spsD > 0 ? static_cast<int>(spsD) : 8;
    int M = static_cast<int>(wf->rows * wf->cols);
    double peak = 0.0;
    for (int k = 1; k < M; ++k) {
        double cr = wf->re[k] * wf->re[k - 1] + wf->im[k] * wf->im[k - 1];
        double ci = wf->im[k] * wf->re[k - 1] - wf->re[k] * wf->im[k - 1];
        double f = std::fabs(std::atan2(ci, cr) / (2.0 * kPi));
        if (f > peak) peak = f;
    }
    (void)sps;
    return bt_scalar(peak);
}

}  /* extern "C" */
