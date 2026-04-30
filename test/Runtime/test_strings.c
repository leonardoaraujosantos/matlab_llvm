/* Phase-1 catch-up: string + fi-render helpers.
 *
 * Targets:
 *   matlab_string_from_literal / matlab_string_concat /
 *   matlab_string_len / matlab_isstring / matlab_string_disp
 *   matlab_strcat / matlab_strrep / matlab_strtrim
 *   matlab_lower / matlab_upper
 *   matlab_startsWith / matlab_endsWith / matlab_contains
 *   matlab_str2double / matlab_num2str
 *   matlab_sprintf_str / matlab_sprintf_f64
 *   matlab_fi_bin_s/u / matlab_fi_hex_s/u / matlab_fi_dec_s/u
 *
 * The string descriptor is opaque at the public ABI; tests poke at
 * the layout via the local mirror. */

#include "runtime_test.h"

/* Mirror of struct matlab_string_s in matlab_runtime.cpp. */
struct rt_test_string_layout {
    char *data;
    int64_t len;
};

typedef struct matlab_string_s matlab_string;

matlab_string *matlab_string_from_literal(const char *src, int64_t len);
matlab_string *matlab_string_concat(matlab_string *a, matlab_string *b);
double         matlab_string_len   (matlab_string *s);
double         matlab_isstring     (matlab_string *s);
matlab_string *matlab_strcat       (matlab_string *a, matlab_string *b);
matlab_string *matlab_strrep       (matlab_string *s, matlab_string *o, matlab_string *n);
matlab_string *matlab_strtrim      (matlab_string *s);
matlab_string *matlab_upper        (matlab_string *s);
matlab_string *matlab_lower        (matlab_string *s);
double         matlab_startsWith   (matlab_string *s, matlab_string *p);
double         matlab_endsWith     (matlab_string *s, matlab_string *p);
double         matlab_contains     (matlab_string *s, matlab_string *p);
double         matlab_str2double   (matlab_string *s);
matlab_string *matlab_num2str      (double v);
matlab_string *matlab_sprintf_str  (matlab_string *fmt);
matlab_string *matlab_sprintf_f64  (matlab_string *fmt, double v);

void          *matlab_fi_bin_s     (int64_t  stored, uint8_t WL);
void          *matlab_fi_bin_u     (uint64_t stored, uint8_t WL);
void          *matlab_fi_hex_s     (int64_t  stored, uint8_t WL);
void          *matlab_fi_hex_u     (uint64_t stored, uint8_t WL);
void          *matlab_fi_dec_s     (int64_t  stored, uint8_t WL);
void          *matlab_fi_dec_u     (uint64_t stored, uint8_t WL);

/* Workspace + DAP introspection (declared in runtime_debug.cpp). The
 * REPL/DAP regression tests below verify that a "..."-typed value
 * lands at workspace kind=3 (string) instead of being misrouted to
 * kind=1 (matrix), which used to alias matlab_string::data over
 * matlab_mat::data and render `text = "Test"` as `4 x <heap-garbage>
 * double`. */
void        matlab_ws_set_f64    (const char *name, int64_t len, double v);
void        matlab_ws_set_string (const char *name, int64_t len, void *s);
void        matlab_ws_clear      (void);
const char *matlab_string_get_data(void *s, int64_t *len_out);
int64_t     matlab_string_get_len (void *s);
int         matlab_dbg_ws_count  (void);
const char *matlab_dbg_ws_name   (int i, int64_t *len_out);
int         matlab_dbg_ws_kind   (int i);
double      matlab_dbg_ws_f64    (int i);
void       *matlab_dbg_ws_ptr    (int i);

static const char *str_data(matlab_string *s) {
    return s ? ((struct rt_test_string_layout *)s)->data : "";
}
static int64_t str_len(matlab_string *s) {
    return s ? ((struct rt_test_string_layout *)s)->len : 0;
}
static int str_eq(matlab_string *s, const char *expected) {
    int64_t el = (int64_t)strlen(expected);
    if (str_len(s) != el) return 0;
    return memcmp(str_data(s), expected, (size_t)el) == 0;
}
static matlab_string *L(const char *s) {
    return matlab_string_from_literal(s, (int64_t)strlen(s));
}

/* --- basic literal + concat ---------------------------------------- */
static void test_string_literal_and_len(void) {
    matlab_string *s = L("hello");
    RT_NEAR((double)str_len(s), 5.0, 0.0, "len");
    RT_NEAR(matlab_string_len(s), 5.0, 0.0, "matlab_string_len");
    RT_CHECK(str_eq(s, "hello"), "data");
    RT_NEAR(matlab_isstring(s), 1.0, 0.0, "isstring real");
    RT_NEAR(matlab_isstring(NULL), 0.0, 0.0, "isstring NULL");
}

static void test_string_concat(void) {
    matlab_string *a = L("foo");
    matlab_string *b = L("bar");
    matlab_string *c = matlab_string_concat(a, b);
    RT_CHECK(str_eq(c, "foobar"), "concat");
    /* strcat is the same primitive, just the user-visible name. */
    matlab_string *d = matlab_strcat(a, b);
    RT_CHECK(str_eq(d, "foobar"), "strcat");
}

/* --- predicates ---------------------------------------------------- */
static void test_starts_ends_contains(void) {
    matlab_string *s = L("hello world");
    RT_NEAR(matlab_startsWith(s, L("hello")), 1.0, 0.0, "startsWith yes");
    RT_NEAR(matlab_startsWith(s, L("world")), 0.0, 0.0, "startsWith no");
    RT_NEAR(matlab_endsWith  (s, L("world")), 1.0, 0.0, "endsWith yes");
    RT_NEAR(matlab_endsWith  (s, L("hello")), 0.0, 0.0, "endsWith no");
    RT_NEAR(matlab_contains  (s, L("o w")),   1.0, 0.0, "contains yes");
    RT_NEAR(matlab_contains  (s, L("xyz")),   0.0, 0.0, "contains no");
}

/* --- transformations ---------------------------------------------- */
static void test_upper_lower(void) {
    RT_CHECK(str_eq(matlab_upper(L("Hello, 123!")), "HELLO, 123!"), "upper");
    RT_CHECK(str_eq(matlab_lower(L("Hello, 123!")), "hello, 123!"), "lower");
}

static void test_strtrim(void) {
    RT_CHECK(str_eq(matlab_strtrim(L("   abc   ")),  "abc"),  "trim mid");
    RT_CHECK(str_eq(matlab_strtrim(L("abc")),         "abc"),  "trim none");
    RT_CHECK(str_eq(matlab_strtrim(L("\t  ab cd \n")), "ab cd"), "trim mixed");
}

static void test_strrep(void) {
    RT_CHECK(str_eq(matlab_strrep(L("foo bar foo"), L("foo"), L("baz")),
                    "baz bar baz"), "replace");
    RT_CHECK(str_eq(matlab_strrep(L("aaa"), L("a"), L("bb")), "bbbbbb"),
             "expanding replace");
    RT_CHECK(str_eq(matlab_strrep(L("hello"), L("x"), L("y")), "hello"),
             "no-match unchanged");
}

/* --- num <-> string ----------------------------------------------- */
static void test_num2str(void) {
    matlab_string *s = matlab_num2str(3.14);
    /* Format is implementation-defined; just assert it round-trips. */
    RT_NEAR(matlab_str2double(s), 3.14, 1e-3, "num2str round-trip");
    matlab_string *z = matlab_num2str(0.0);
    RT_NEAR(matlab_str2double(z), 0.0, 1e-12, "num2str(0)");
}

static void test_str2double(void) {
    RT_NEAR(matlab_str2double(L("42")),     42.0,   1e-12, "42");
    RT_NEAR(matlab_str2double(L("-1.5")),  -1.5,    1e-12, "negative");
    RT_NEAR(matlab_str2double(L("1e3")),   1000.0,  1e-12, "scientific");
}

/* --- sprintf scalar ---------------------------------------------- */
static void test_sprintf(void) {
    /* sprintf with no args returns the format string verbatim. */
    matlab_string *r = matlab_sprintf_str(L("hi"));
    RT_CHECK(str_eq(r, "hi"), "sprintf_str literal");

    /* sprintf_f64 expands one %g / %d / %f. */
    matlab_string *q = matlab_sprintf_f64(L("v=%g"), 2.5);
    RT_CHECK(str_eq(q, "v=2.5"), "sprintf %g");
}

/* --- fi rendering helpers ----------------------------------------- */
static void test_fi_bin_hex_dec(void) {
    /* binary of 5 with WL=4 = "0101". */
    matlab_string *b = (matlab_string *)matlab_fi_bin_s(5, 4);
    RT_CHECK(str_eq(b, "0101"), "bin_s 5,4");

    /* hex of 255 with WL=8 = "ff". */
    matlab_string *h = (matlab_string *)matlab_fi_hex_s(255, 8);
    RT_CHECK(str_eq(h, "ff"), "hex_s 255,8");

    /* dec of -3 with WL=8 (two's-complement) — runtime emits the
     * decimal stored value as text; just check it round-trips through
     * str2double. */
    matlab_string *d = (matlab_string *)matlab_fi_dec_s(-3, 8);
    RT_NEAR(matlab_str2double(d), -3.0, 1e-12, "dec_s -3");

    /* Unsigned variants. */
    matlab_string *bu = (matlab_string *)matlab_fi_bin_u(0xA, 4);
    RT_CHECK(str_eq(bu, "1010"), "bin_u 10,4");
    matlab_string *hu = (matlab_string *)matlab_fi_hex_u(255, 8);
    RT_CHECK(str_eq(hu, "ff"), "hex_u 255,8");
    matlab_string *du = (matlab_string *)matlab_fi_dec_u(42, 8);
    RT_NEAR(matlab_str2double(du), 42.0, 1e-12, "dec_u 42");
}

/* --- workspace (REPL/DAP) regression -------------------------------- */
/* Locate the workspace slot for `name`, or -1 if it isn't present. */
static int find_ws_slot(const char *name) {
    int n = matlab_dbg_ws_count();
    int64_t want = (int64_t)strlen(name);
    for (int i = 0; i < n; ++i) {
        int64_t got = 0;
        const char *gn = matlab_dbg_ws_name(i, &got);
        if (got == want && memcmp(gn, name, (size_t)want) == 0)
            return i;
    }
    return -1;
}

/* `text = "Test"` regression. Before the fix, the REPL workspace
 * stored matlab_string* under kind=1 (matrix) — the inspector then
 * cast the pointer to matlab_mat* and reported the descriptor's
 * length field as `rows` and the next 8 heap bytes as `cols`,
 * producing the user-visible "4 x -3847672034583807139" garbage.
 *
 * The asserts pin every contract that has to hold for the fix to
 * stick: kind must be 3 (string), the round-tripped pointer must
 * decode back through the opaque matlab_string accessors, and the
 * shape must NOT match the expected matlab_mat-aliasing values that
 * would identify a regression. */
static void test_workspace_string_assignment(void) {
    matlab_ws_clear();
    matlab_string *s = L("Test");
    matlab_ws_set_string("text", 4, s);

    int idx = find_ws_slot("text");
    RT_CHECK(idx >= 0, "text exists in ws");

    int kind = matlab_dbg_ws_kind(idx);
    RT_NEAR((double)kind, 3.0, 0.0,
            "ws kind == 3 (string) — kind=1 would mean matlab_mat alias");
    RT_CHECK(kind != 1,
             "regression: string MUST NOT be stored under kind=1 (matrix)");

    void *roundtrip = matlab_dbg_ws_ptr(idx);
    RT_CHECK(roundtrip == (void *)s, "ws_ptr round-trips the same pointer");
    RT_NEAR((double)matlab_string_get_len(roundtrip), 4.0, 0.0,
            "string len round-trips");
    int64_t got_len = 0;
    const char *got_data = matlab_string_get_data(roundtrip, &got_len);
    RT_NEAR((double)got_len, 4.0, 0.0, "get_data len");
    RT_CHECK(memcmp(got_data, "Test", 4) == 0, "get_data bytes");
}

/* Reassigning the same name to a different kind must update the slot
 * (no stale string state under a now-numeric variable). Mirrors the
 * REPL pattern `text = "Test"; text = 4` — kind must switch to 0,
 * the f64 must be readable, and the string pointer must be cleared. */
static void test_workspace_string_to_f64_reassign(void) {
    matlab_ws_clear();
    matlab_ws_set_string("text", 4, L("Test"));
    int idx = find_ws_slot("text");
    RT_NEAR((double)matlab_dbg_ws_kind(idx), 3.0, 0.0,
            "first assign records kind=3");

    matlab_ws_set_f64("text", 4, 42.0);
    idx = find_ws_slot("text");
    RT_CHECK(idx >= 0, "text still present after reassign");
    RT_NEAR((double)matlab_dbg_ws_kind(idx), 0.0, 0.0,
            "kind switches to 0 after numeric reassign");
    RT_NEAR(matlab_dbg_ws_f64(idx), 42.0, 0.0, "f64 value");
}

/* Multiple distinct strings coexist with their own slots — guards
 * against any single-slot string-table shortcut. */
static void test_workspace_two_strings(void) {
    matlab_ws_clear();
    matlab_ws_set_string("a", 1, L("alpha"));
    matlab_ws_set_string("b", 1, L("beta"));

    int ai = find_ws_slot("a");
    int bi = find_ws_slot("b");
    RT_CHECK(ai >= 0 && bi >= 0 && ai != bi, "two distinct slots");
    RT_NEAR((double)matlab_dbg_ws_kind(ai), 3.0, 0.0, "a is kind=3");
    RT_NEAR((double)matlab_dbg_ws_kind(bi), 3.0, 0.0, "b is kind=3");

    int64_t la = 0, lb = 0;
    const char *da = matlab_string_get_data(matlab_dbg_ws_ptr(ai), &la);
    const char *db = matlab_string_get_data(matlab_dbg_ws_ptr(bi), &lb);
    RT_CHECK(la == 5 && memcmp(da, "alpha", 5) == 0, "a bytes");
    RT_CHECK(lb == 4 && memcmp(db, "beta",  4) == 0, "b bytes");
}

/* matlab_string_get_data on a NULL pointer must yield empty bytes
 * rather than crashing — the DAP inspector will hit this on a slot
 * that hasn't been initialised yet. */
static void test_string_get_data_null(void) {
    int64_t L0 = 9;
    const char *D = matlab_string_get_data(NULL, &L0);
    RT_NEAR((double)L0, 0.0, 0.0, "NULL → len 0");
    RT_CHECK(D != NULL, "NULL → non-NULL empty buffer");
    RT_NEAR((double)matlab_string_get_len(NULL), 0.0, 0.0,
            "NULL get_len → 0");
}

int main(void) {
    fprintf(stderr, "test_strings:\n");
    RT_RUN(test_string_literal_and_len);
    RT_RUN(test_string_concat);
    RT_RUN(test_starts_ends_contains);
    RT_RUN(test_upper_lower);
    RT_RUN(test_strtrim);
    RT_RUN(test_strrep);
    RT_RUN(test_num2str);
    RT_RUN(test_str2double);
    RT_RUN(test_sprintf);
    RT_RUN(test_fi_bin_hex_dec);
    RT_RUN(test_workspace_string_assignment);
    RT_RUN(test_workspace_string_to_f64_reassign);
    RT_RUN(test_workspace_two_strings);
    RT_RUN(test_string_get_data_null);
    RT_DONE();
}
