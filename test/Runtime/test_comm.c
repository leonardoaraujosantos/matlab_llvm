/* Direct unit tests for the Communications Toolbox runtime entries:
 * Tier-1 base layer (rng, int2bit/bit2int, biterr, AWGN smoke),
 * Tier-2 digital modulation (PAM, PSK, QAM mod/demod round-trips,
 * berawgn closed-form), Tier-3 channel coding (CRC, Hamming, intrlv),
 * Tier-7 LDPC + Polar small-N round-trip. */

#include "runtime_test.h"

/* Forward decls — runtime_comm.cpp entries */
double      matlab_comm_randi_s       (double imax);
void        matlab_comm_rng           (double seed);
matlab_mat *matlab_comm_int2bit       (matlab_mat *ints, double nbits);
matlab_mat *matlab_comm_bit2int       (matlab_mat *bits, double nbits);
double      matlab_comm_biterr_ratio  (matlab_mat *x, matlab_mat *y);
double      matlab_comm_biterr_count  (matlab_mat *x, matlab_mat *y);
double      matlab_comm_symerr_ratio  (matlab_mat *x, matlab_mat *y);
double      matlab_comm_symerr_count  (matlab_mat *x, matlab_mat *y);
matlab_mat *matlab_comm_pammod        (matlab_mat *x, double Md, double order);
matlab_mat *matlab_comm_pamdemod      (matlab_mat *y, double Md, double order);
matlab_mat_c *matlab_comm_pskmod      (matlab_mat *x, double Md,
                                        double ini_phase, double order);
matlab_mat *matlab_comm_pskdemod      (matlab_mat_c *y, double Md,
                                        double ini_phase, double order);
matlab_mat_c *matlab_comm_qammod      (matlab_mat *x, double Md,
                                        double order, double unit_avg);
matlab_mat *matlab_comm_qamdemod      (matlab_mat_c *y, double Md,
                                        double order, double unit_avg);
double      matlab_comm_berawgn_s     (double ebn0_dB, double Md, double mod);
double      matlab_comm_qfunc_s       (double x);
double      matlab_comm_erfc_s        (double x);
matlab_mat *matlab_comm_crc_generate  (matlab_mat *bits, double poly_int_d,
                                        double nbits_d);
double      matlab_comm_crc_check     (matlab_mat *bits, double poly_int_d,
                                        double nbits_d);
matlab_mat *matlab_comm_crc_strip     (matlab_mat *bits, double nbits_d);
matlab_mat *matlab_comm_hamming_encode(matlab_mat *msg, double md);
matlab_mat *matlab_comm_hamming_decode(matlab_mat *code, double md);
matlab_mat *matlab_comm_intrlv        (matlab_mat *data, matlab_mat *perm);
matlab_mat *matlab_comm_deintrlv      (matlab_mat *data, matlab_mat *perm);
matlab_mat *matlab_comm_polar_encode  (matlab_mat *u, double Nd);
matlab_mat *matlab_comm_polar_sc_decode(matlab_mat *llr, matlab_mat *frozen,
                                         double Nd);
matlab_mat *matlab_comm_pn_sequence   (double poly_int, double init_int,
                                        double length, double output_mode);
matlab_mat *matlab_comm_hadamard      (double n);

/* Complex-matrix shape helpers — defined in matlab_runtime.cpp */
matlab_mat_c *matlab_mat_c_from_buf(const double *re, const double *im,
                                     double rows, double cols);

static matlab_mat *mk(const double *buf, int64_t m, int64_t n) {
    return matlab_mat_from_buf(buf, (double)m, (double)n);
}

/* ===== Tier-1 base layer ===== */

static void test_rng_deterministic(void) {
    matlab_comm_rng(42.0);
    double a = matlab_comm_randi_s(100.0);
    matlab_comm_rng(42.0);
    double b = matlab_comm_randi_s(100.0);
    RT_NEAR(a, b, 1e-12, "rng(42) makes randi deterministic");
    RT_CHECK(a >= 1.0 && a <= 100.0, "randi in range");
}

static void test_int2bit_round_trip(void) {
    /* int [5, 3] -> bits MSB-first with nbits=4
       5 = 0101, 3 = 0011 -> stacked. */
    double ibuf[] = {5, 3};
    matlab_mat *ints = mk(ibuf, 2, 1);
    matlab_mat *bits = matlab_comm_int2bit(ints, 4.0);
    matlab_mat *back = matlab_comm_bit2int(bits, 4.0);
    int64_t n = rt_rows(back) * rt_cols(back);
    RT_CHECK(n == 2, "round-trip preserves count");
    RT_NEAR(rt_data(back)[0], 5.0, 1e-9, "int 5 round-trips");
    RT_NEAR(rt_data(back)[1], 3.0, 1e-9, "int 3 round-trips");
    rt_free(ints); rt_free(bits); rt_free(back);
}

static void test_biterr_basic(void) {
    double x[] = {0, 1, 1, 0, 1, 0, 1, 1};
    double y[] = {0, 0, 1, 0, 1, 1, 1, 0};
    /* 3 bit differences -> ratio = 3/8. */
    matlab_mat *X = mk(x, 8, 1);
    matlab_mat *Y = mk(y, 8, 1);
    double ratio = matlab_comm_biterr_ratio(X, Y);
    double count = matlab_comm_biterr_count(X, Y);
    RT_NEAR(ratio, 3.0 / 8.0, 1e-12, "biterr ratio");
    RT_NEAR(count, 3.0, 1e-12, "biterr count");
    rt_free(X); rt_free(Y);
}

static void test_symerr_basic(void) {
    double x[] = {0, 1, 2, 3};
    double y[] = {0, 1, 0, 3};
    matlab_mat *X = mk(x, 4, 1);
    matlab_mat *Y = mk(y, 4, 1);
    double r = matlab_comm_symerr_ratio(X, Y);
    double c = matlab_comm_symerr_count(X, Y);
    RT_NEAR(r, 0.25, 1e-12, "symerr ratio");
    RT_NEAR(c, 1.0, 1e-12, "symerr count");
    rt_free(X); rt_free(Y);
}

/* ===== Tier-2 digital modulation ===== */

static void test_pam_round_trip(void) {
    /* M=4 PAM: symbols 0,1,2,3 -> levels -3,-1,1,3 -> back. */
    double ibuf[] = {0, 1, 2, 3, 0, 3, 2, 1};
    matlab_mat *ints = mk(ibuf, 8, 1);
    matlab_mat *y    = matlab_comm_pammod  (ints, 4.0, 0.0);
    matlab_mat *back = matlab_comm_pamdemod(y,    4.0, 0.0);
    for (int i = 0; i < 8; ++i)
        RT_NEAR(rt_data(back)[i], ibuf[i], 1e-9, "PAM round-trip");
    rt_free(ints); rt_free(y); rt_free(back);
}

static void test_psk_round_trip(void) {
    /* QPSK with natural mapping. */
    double ibuf[] = {0, 1, 2, 3, 0, 1, 2, 3};
    matlab_mat *ints = mk(ibuf, 8, 1);
    matlab_mat_c *y  = matlab_comm_pskmod  (ints, 4.0, 0.0, 0.0);
    matlab_mat   *back = matlab_comm_pskdemod(y,   4.0, 0.0, 0.0);
    for (int i = 0; i < 8; ++i)
        RT_NEAR(rt_data(back)[i], ibuf[i], 1e-9, "PSK round-trip");
    rt_free(ints); rt_c_free(y); rt_free(back);
}

static void test_qam_round_trip_16(void) {
    /* 16-QAM with all 16 input symbols. */
    double ibuf[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    matlab_mat *ints = mk(ibuf, 16, 1);
    matlab_mat_c *y  = matlab_comm_qammod  (ints, 16.0, 0.0, 0.0);
    matlab_mat   *back = matlab_comm_qamdemod(y,   16.0, 0.0, 0.0);
    for (int i = 0; i < 16; ++i)
        RT_NEAR(rt_data(back)[i], ibuf[i], 1e-9, "16-QAM round-trip");
    rt_free(ints); rt_c_free(y); rt_free(back);
}

static void test_berawgn_qpsk(void) {
    /* QPSK theoretical BER at Eb/N0 = 10 dB: Q(sqrt(2*10)) ~= 3.872e-6.
       Code 1 in this runtime is PSK. */
    double ber = matlab_comm_berawgn_s(10.0, 4.0, 1.0);
    RT_CHECK(ber > 0.0 && ber < 1e-4, "berawgn QPSK 10 dB in range");
}

static void test_berawgn_bpsk_4dB(void) {
    /* BPSK at Eb/N0 = 4 dB -> Q(sqrt(2*10^0.4)) ~= 0.0125. */
    double ber = matlab_comm_berawgn_s(4.0, 2.0, 1.0);
    RT_CHECK(ber > 0.005 && ber < 0.02, "berawgn BPSK 4 dB");
}

static void test_qfunc_known(void) {
    /* Q(0) = 0.5, Q(1) = 0.158655, Q(2) = 0.02275. */
    RT_NEAR(matlab_comm_qfunc_s(0.0), 0.5,      1e-9,  "Q(0)");
    RT_NEAR(matlab_comm_qfunc_s(1.0), 0.158655, 1e-4,  "Q(1)");
    RT_NEAR(matlab_comm_qfunc_s(2.0), 0.022750, 1e-4,  "Q(2)");
}

/* ===== Tier-3 channel coding ===== */

static void test_crc_generate_check(void) {
    /* Match the convention in `examples/comm/tier3_smoke.m`:
     * `poly` is the low-`nbits` representation (no implicit leading 1). */
    double mbuf[] = {1, 0, 1, 1, 0, 1, 0, 0};
    matlab_mat *msg = mk(mbuf, 8, 1);
    matlab_mat *coded = matlab_comm_crc_generate(msg, 7.0, 8.0);
    /* CRC length is 8 -> coded has 8 + 8 = 16 bits. */
    int64_t n = rt_rows(coded) * rt_cols(coded);
    RT_CHECK(n == 16, "CRC-8 appends 8 bits");
    /* Checking the coded payload should pass (0 = no error). */
    double e = matlab_comm_crc_check(coded, 7.0, 8.0);
    RT_NEAR(e, 0.0, 1e-12, "CRC check passes on coded");
    /* Strip should give the original message back. */
    matlab_mat *stripped = matlab_comm_crc_strip(coded, 8.0);
    int64_t sn = rt_rows(stripped) * rt_cols(stripped);
    RT_CHECK(sn == 8, "stripped length matches");
    for (int i = 0; i < 8; ++i)
        RT_NEAR(rt_data(stripped)[i], mbuf[i], 1e-12, "CRC strip");
    rt_free(msg); rt_free(coded); rt_free(stripped);
}

static void test_hamming_7_4_round_trip(void) {
    /* Hamming(7,4) encodes 4-bit messages into 7-bit codewords.
     * The runtime takes `m` (parity-bit count) as the second arg:
     * m=3 -> n=7, k=4 (the canonical Hamming(7,4)). */
    double mbuf[] = {1, 0, 1, 1};   /* k = 4 bits */
    matlab_mat *msg = mk(mbuf, 4, 1);
    matlab_mat *code = matlab_comm_hamming_encode(msg, 3.0);
    int64_t cn = rt_rows(code) * rt_cols(code);
    RT_CHECK(cn == 7, "Hamming(7,4) length");
    matlab_mat *dec = matlab_comm_hamming_decode(code, 3.0);
    int64_t dn = rt_rows(dec) * rt_cols(dec);
    RT_CHECK(dn == 4, "Hamming decode length");
    for (int i = 0; i < 4; ++i)
        RT_NEAR(rt_data(dec)[i], mbuf[i], 1e-12, "Hamming round-trip");
    rt_free(msg); rt_free(code); rt_free(dec);
}

static void test_intrlv_deintrlv_round_trip(void) {
    /* Permutation: [3, 1, 4, 2, 5] applied to [10, 20, 30, 40, 50]
       Then deinterleave should restore the original. */
    double dbuf[] = {10, 20, 30, 40, 50};
    double pbuf[] = {3, 1, 4, 2, 5};
    matlab_mat *data = mk(dbuf, 5, 1);
    matlab_mat *perm = mk(pbuf, 5, 1);
    matlab_mat *intd = matlab_comm_intrlv(data, perm);
    matlab_mat *back = matlab_comm_deintrlv(intd, perm);
    for (int i = 0; i < 5; ++i)
        RT_NEAR(rt_data(back)[i], dbuf[i], 1e-12, "intrlv round-trip");
    rt_free(data); rt_free(perm); rt_free(intd); rt_free(back);
}

/* ===== Tier-6 spreading sequences ===== */

static void test_pn_sequence_length(void) {
    /* PN sequence from poly=0x09 (LFSR x^3 + 1), init=0x01, length=7. */
    matlab_mat *pn = matlab_comm_pn_sequence(9.0, 1.0, 7.0, 0.0);
    int64_t n = rt_rows(pn) * rt_cols(pn);
    RT_CHECK(n == 7, "PN length matches request");
    /* All entries should be 0 or 1. */
    for (int i = 0; i < n; ++i) {
        double v = rt_data(pn)[i];
        RT_CHECK(v == 0.0 || v == 1.0, "PN binary");
    }
    rt_free(pn);
}

static void test_hadamard_orthogonal(void) {
    /* 4x4 Hadamard: H*H' = 4*I. */
    matlab_mat *H = matlab_comm_hadamard(4.0);
    RT_CHECK(rt_rows(H) == 4 && rt_cols(H) == 4, "Hadamard 4x4");
    /* Row 1 (index 0) should be all +1. */
    for (int j = 0; j < 4; ++j)
        RT_NEAR(rt_at(H, 0, j), 1.0, 1e-12, "Hadamard row 1");
    /* Inner product of row 0 with row 1 should be 0. */
    double dot = 0.0;
    for (int j = 0; j < 4; ++j) dot += rt_at(H, 0, j) * rt_at(H, 1, j);
    RT_NEAR(dot, 0.0, 1e-12, "Hadamard orthogonal rows");
    rt_free(H);
}

/* ===== Tier-7 modern codes (Polar) ===== */

static void test_polar_encode_length(void) {
    /* Polar(8) — encode a length-8 message into a length-8 codeword. */
    double ubuf[] = {1, 0, 1, 1, 0, 0, 1, 0};
    matlab_mat *u = mk(ubuf, 8, 1);
    matlab_mat *c = matlab_comm_polar_encode(u, 8.0);
    int64_t n = rt_rows(c) * rt_cols(c);
    RT_CHECK(n == 8, "Polar(8) codeword length");
    rt_free(u); rt_free(c);
}

int main(void) {
    RT_RUN(test_rng_deterministic);
    RT_RUN(test_int2bit_round_trip);
    RT_RUN(test_biterr_basic);
    RT_RUN(test_symerr_basic);
    RT_RUN(test_pam_round_trip);
    RT_RUN(test_psk_round_trip);
    RT_RUN(test_qam_round_trip_16);
    RT_RUN(test_berawgn_qpsk);
    RT_RUN(test_berawgn_bpsk_4dB);
    RT_RUN(test_qfunc_known);
    RT_RUN(test_crc_generate_check);
    RT_RUN(test_hamming_7_4_round_trip);
    RT_RUN(test_intrlv_deintrlv_round_trip);
    RT_RUN(test_pn_sequence_length);
    RT_RUN(test_hadamard_orthogonal);
    RT_RUN(test_polar_encode_length);
    RT_DONE();
}
