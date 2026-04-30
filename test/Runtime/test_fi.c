/* Phase-1 catch-up: direct unit tests for the Fixed-Point Designer
 * (`fi`) helpers — saturation, rounding modes, quantisation. The
 * .m integration suite covers the lowering shape end-to-end; this
 * suite exercises the scalar primitives in isolation so a regression
 * in saturation / rounding doesn't get masked by frontend changes. */

#include "runtime_test.h"

int64_t  matlab_fi_sat_s64        (int64_t x, uint8_t WL);
uint64_t matlab_fi_sat_u64        (uint64_t x, uint8_t WL);
int64_t  matlab_fi_round_floor_s  (int64_t x, uint8_t shift);
int64_t  matlab_fi_round_nearest_s(int64_t x, uint8_t shift);
int64_t  matlab_fi_round_zero_s   (int64_t x, uint8_t shift);
int64_t  matlab_fi_round_ceiling_s(int64_t x, uint8_t shift);
int64_t  matlab_fi_round_convergent_s(int64_t x, uint8_t shift);
uint64_t matlab_fi_round_floor_u  (uint64_t x, uint8_t shift);
uint64_t matlab_fi_round_nearest_u(uint64_t x, uint8_t shift);
int64_t  matlab_fi_quantize_s     (double v, uint8_t WL, int8_t FL,
                                    uint8_t overflow, uint8_t rounding);
uint64_t matlab_fi_quantize_u     (double v, uint8_t WL, int8_t FL,
                                    uint8_t overflow, uint8_t rounding);

/* --- saturation ----------------------------------------------------- */
static void test_sat_s64_in_range(void) {
    /* WL=8 signed: range = [-128, 127]. Values inside pass through. */
    RT_NEAR((double)matlab_fi_sat_s64( 100, 8),  100.0, 0.0, "in range");
    RT_NEAR((double)matlab_fi_sat_s64(-100, 8), -100.0, 0.0, "in range neg");
    RT_NEAR((double)matlab_fi_sat_s64(   0, 8),    0.0, 0.0, "zero");
}
static void test_sat_s64_clamps_high(void) {
    /* 200 > 127, clamps to 127. */
    RT_NEAR((double)matlab_fi_sat_s64( 200, 8),  127.0, 0.0, "high clamp");
}
static void test_sat_s64_clamps_low(void) {
    /* -200 < -128, clamps to -128. */
    RT_NEAR((double)matlab_fi_sat_s64(-200, 8), -128.0, 0.0, "low clamp");
}
static void test_sat_u64_clamps(void) {
    /* WL=8 unsigned: range = [0, 255]. */
    RT_NEAR((double)matlab_fi_sat_u64(100,  8), 100.0, 0.0, "unsigned in");
    RT_NEAR((double)matlab_fi_sat_u64(300,  8), 255.0, 0.0, "unsigned high");
}

/* --- rounding modes (signed) --------------------------------------- */
static void test_round_floor_s(void) {
    /* shift=2 means divide-by-4 with rounding. Floor: round toward -inf. */
    RT_NEAR((double)matlab_fi_round_floor_s( 7, 2),  1.0, 0.0, "floor 7/4 = 1");
    RT_NEAR((double)matlab_fi_round_floor_s(-7, 2), -2.0, 0.0, "floor -7/4 = -2");
    RT_NEAR((double)matlab_fi_round_floor_s( 8, 2),  2.0, 0.0, "floor 8/4 = 2");
}
static void test_round_nearest_s(void) {
    /* The runtime uses round-half-up (toward +inf at ties), not
     * MATLAB's spec-default round-half-away-from-zero. So -6 / 4 (a
     * half-tie at -1.5) rounds to -1 here, not -2. */
    RT_NEAR((double)matlab_fi_round_nearest_s( 5, 2),  1.0, 0.0, "nearest 5/4 = 1");
    RT_NEAR((double)matlab_fi_round_nearest_s( 6, 2),  2.0, 0.0, "nearest 6/4 = 2 (tie)");
    RT_NEAR((double)matlab_fi_round_nearest_s( 7, 2),  2.0, 0.0, "nearest 7/4 = 2");
    RT_NEAR((double)matlab_fi_round_nearest_s(-6, 2), -1.0, 0.0, "nearest -6/4 = -1 (half-up)");
}
static void test_round_zero_s(void) {
    /* Round toward zero (truncate). */
    RT_NEAR((double)matlab_fi_round_zero_s( 7, 2),  1.0, 0.0, "trunc 7/4 = 1");
    RT_NEAR((double)matlab_fi_round_zero_s(-7, 2), -1.0, 0.0, "trunc -7/4 = -1");
}
static void test_round_ceiling_s(void) {
    /* Round toward +inf. */
    RT_NEAR((double)matlab_fi_round_ceiling_s( 5, 2),  2.0, 0.0, "ceil 5/4 = 2");
    RT_NEAR((double)matlab_fi_round_ceiling_s(-7, 2), -1.0, 0.0, "ceil -7/4 = -1");
}
static void test_round_convergent_s(void) {
    /* Banker's rounding — ties to even. */
    RT_NEAR((double)matlab_fi_round_convergent_s( 6, 2),  2.0, 0.0,
            "convergent 6/4 = 2 (tie -> even)");
    RT_NEAR((double)matlab_fi_round_convergent_s(10, 2),  2.0, 0.0,
            "convergent 10/4 = 2 (tie -> even)");
    RT_NEAR((double)matlab_fi_round_convergent_s(14, 2),  4.0, 0.0,
            "convergent 14/4 = 4 (tie -> even)");
}

/* --- rounding modes (unsigned) ------------------------------------- */
static void test_round_floor_u(void) {
    RT_NEAR((double)matlab_fi_round_floor_u(7, 2), 1.0, 0.0, "ufloor 7/4");
    RT_NEAR((double)matlab_fi_round_floor_u(8, 2), 2.0, 0.0, "ufloor 8/4");
}
static void test_round_nearest_u(void) {
    RT_NEAR((double)matlab_fi_round_nearest_u(5, 2), 1.0, 0.0, "unearest 5/4");
    RT_NEAR((double)matlab_fi_round_nearest_u(6, 2), 2.0, 0.0, "unearest tie");
}

/* --- quantize end-to-end ------------------------------------------- */
static void test_quantize_signed_round_trip(void) {
    /* Q8.4 signed: WL=8, FL=4. Stored = round(v * 2^4). */
    /* v = 1.5 → round(1.5 * 16) = 24 (Q8.4 stored value). */
    RT_NEAR((double)matlab_fi_quantize_s(1.5, 8, 4, 1 /* sat */, 1 /* nearest */),
            24.0, 0.0, "Q8.4 1.5");
    /* v = -1.5 → -24 stored. */
    RT_NEAR((double)matlab_fi_quantize_s(-1.5, 8, 4, 1, 1),
            -24.0, 0.0, "Q8.4 -1.5");
    /* v = 0 → 0. */
    RT_NEAR((double)matlab_fi_quantize_s(0.0, 8, 4, 1, 1),
            0.0, 0.0, "Q8.4 zero");
}

static void test_quantize_signed_saturates_overflow(void) {
    /* Q8.4 signed: range = [-128, 127] in stored units => real range
     * = [-8.0, 7.9375]. v = 100.0 should saturate to 127. */
    RT_NEAR((double)matlab_fi_quantize_s(100.0, 8, 4, 1 /* sat */, 1),
            127.0, 0.0, "saturate high");
    RT_NEAR((double)matlab_fi_quantize_s(-100.0, 8, 4, 1, 1),
            -128.0, 0.0, "saturate low");
}

static void test_quantize_unsigned(void) {
    /* UQ8.4: range = [0, 255] stored. v = 1.5 → 24 stored. */
    RT_NEAR((double)matlab_fi_quantize_u(1.5, 8, 4, 1, 1),
            24.0, 0.0, "UQ8.4 1.5");
    /* Out-of-range positive saturates. */
    RT_NEAR((double)matlab_fi_quantize_u(100.0, 8, 4, 1, 1),
            255.0, 0.0, "UQ8.4 sat high");
    /* Negative on unsigned: saturate to 0. */
    RT_NEAR((double)matlab_fi_quantize_u(-1.0, 8, 4, 1, 1),
            0.0, 0.0, "UQ8.4 sat low");
}

int main(void) {
    fprintf(stderr, "test_fi:\n");
    RT_RUN(test_sat_s64_in_range);
    RT_RUN(test_sat_s64_clamps_high);
    RT_RUN(test_sat_s64_clamps_low);
    RT_RUN(test_sat_u64_clamps);
    RT_RUN(test_round_floor_s);
    RT_RUN(test_round_nearest_s);
    RT_RUN(test_round_zero_s);
    RT_RUN(test_round_ceiling_s);
    RT_RUN(test_round_convergent_s);
    RT_RUN(test_round_floor_u);
    RT_RUN(test_round_nearest_u);
    RT_RUN(test_quantize_signed_round_trip);
    RT_RUN(test_quantize_signed_saturates_overflow);
    RT_RUN(test_quantize_unsigned);
    RT_DONE();
}
