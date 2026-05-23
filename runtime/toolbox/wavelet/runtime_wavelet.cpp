/* ============================================================================
 * runtime_wavelet.cpp — Wavelet Toolbox runtime
 * ----------------------------------------------------------------------------
 * Tier-1: discrete wavelet core + family-filter catalogue (the Mallat fast
 * wavelet transform).  wfilters / dwt / idwt / wavedec / waverec / appcoef /
 * detcoef / wrcoef / wextend / wkeep / wmaxlev / wentropy / wenergy / qmf /
 * centfrq.
 * Tier-2: denoising + compression (wthresh / thselect / wnoisest / wnoise /
 * ddencmp / wden / wdenoise / wcompress / measerr).
 * Tier-3: continuous wavelet transform (cwt / icwt / scal2frq / wcoherence).
 * Tier-4: undecimated transforms + 2-D (swt / modwt / modwtmra / dwt2 /
 * wavedec2 / waverec2 / wcodemat).
 * Tier-5/6: wavelet packets, EWT/VMD/EMD, matching pursuit, scattering.
 *
 * Representation: signals/coefficients are plain `double` matlab_mat row or
 * column vectors; decomposition structures are the MATLAB `[C, L]` (1-D) and
 * `[C, S]` (2-D) concatenated-coefficient + bookkeeping matrices — no opaque
 * objects in the matrix lane (Tiers 1-2-4).  The classdef tiers (cwtfilterbank
 * / WPTREE / waveletScattering) carry their state in matlab_obj records.
 *
 * Perfect reconstruction: the DWT uses an orthonormal *circular* (periodic)
 * two-channel filter bank.  Analysis is W (rows = double-shifted Lo_D / Hi_D),
 * synthesis is W^T — exact PR for any properly orthogonal QMF filter, which a
 * PR loop test over the whole family catalogue validates.  No external
 * dependency (no PyWavelets / WaveLab); families are hard-coded scaling-filter
 * tables, the same lookup-table precedent as the Comm 5G-NR base matrices and
 * the Image fspecial kernels.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

/* shipped helpers reused. */
extern "C" matlab_mat   *matlab_conv(matlab_mat *u, matlab_mat *v);
extern "C" matlab_mat   *matlab_conv2(matlab_mat *A, matlab_mat *B);
extern "C" matlab_mat_c *matlab_fft_c(void *Aptr);
extern "C" matlab_mat_c *matlab_ifft_c(void *Aptr);
extern "C" matlab_mat   *matlab_randn(double m, double n);

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
namespace {
struct wv_string_s { char *data; int64_t len; };

/* ===== small helpers ===================================================== */

std::string wv_sstr(const void *s) {
    if (!s) return std::string();
    const wv_string_s *p = reinterpret_cast<const wv_string_s *>(s);
    if (!p->data || p->len <= 0 || p->len > 4096) return std::string();
    return std::string(p->data, p->data + p->len);
}
double wv_sc(const matlab_mat *m, double dflt) {
    return (m && m->data && m->rows * m->cols > 0) ? m->data[0] : dflt;
}
/* Flatten a real matrix to a column-major-agnostic vector (row-major read);
 * vectors are read in natural order regardless of orientation. */
std::vector<double> wv_vec(const matlab_mat *m) {
    std::vector<double> v;
    if (!m || !m->data) return v;
    int64_t n = m->rows * m->cols;
    v.assign(m->data, m->data + n);
    return v;
}
bool wv_is_col(const matlab_mat *m) { return m && m->cols == 1 && m->rows > 1; }

/* Wrap a vector back into a matlab_mat with the requested orientation. */
matlab_mat *wv_row(const std::vector<double> &v) {
    matlab_mat *r = mat_alloc(1, static_cast<int64_t>(v.size()));
    if (r && r->data) memcpy(r->data, v.data(), v.size() * sizeof(double));
    return r;
}
matlab_mat *wv_col(const std::vector<double> &v) {
    matlab_mat *r = mat_alloc(static_cast<int64_t>(v.size()), 1);
    if (r && r->data) memcpy(r->data, v.data(), v.size() * sizeof(double));
    return r;
}
matlab_mat *wv_oriented(const std::vector<double> &v, bool col) {
    return col ? wv_col(v) : wv_row(v);
}

/* ===== family-filter catalogue =========================================
 * Stores Lo_D (decomposition low-pass = orthonormal scaling filter, sum √2,
 * sum-of-squares 1) per orthogonal family member.  Hi_D / Lo_R / Hi_R are
 * derived by QMF + reversal so the four filters are guaranteed self-consistent.
 * Biorthogonal (bior/rbio) members carry an explicit dec/rec low-pass pair. */

struct WvFilters { std::vector<double> LoD, HiD, LoR, HiR; bool ortho; };

/* Daubechies / Symlet / Coiflet scaling filters (PyWavelets dec_lo order). */
const double DB1[]  = {0.7071067811865476, 0.7071067811865476};
const double DB2[]  = {-0.12940952255092145, 0.22414386804185735,
                       0.836516303737469, 0.48296291314469025};
const double DB3[]  = {0.035226291882100656, -0.08544127388224149,
                       -0.13501102001039084, 0.4598775021193313,
                       0.8068915093133388, 0.3326705529509569};
const double DB4[]  = {-0.010597401784997278, 0.032883011666982945,
                       0.030841381835986965, -0.18703481171888114,
                       -0.02798376941698385, 0.6308807679295904,
                       0.7148465705525415, 0.23037781330885523};
const double DB5[]  = {0.003335725285001549, -0.012580751999015526,
                       -0.006241490213011705, 0.07757149384006515,
                       -0.03224486958502952, -0.24229488706619015,
                       0.13842814590110342, 0.7243085284385744,
                       0.6038292697974729, 0.160102397974125};
const double DB6[]  = {-0.00107730108499558, 0.004777257511010651,
                       0.0005538422009938016, -0.031582039318031156,
                       0.02752286553001629, 0.09750160558707936,
                       -0.12976686756709563, -0.22626469396516913,
                       0.3152503517092432, 0.7511339080215775,
                       0.4946238903983854, 0.11154074335008017};
const double DB7[]  = {0.0003537138000010399, -0.0018016407039998328,
                       0.00042957797300470274, 0.012550998556013784,
                       -0.01657454163101562, -0.03802993693503463,
                       0.0806126091510659, 0.07130921926705004,
                       -0.22403618499416572, -0.14390600392910627,
                       0.4697822874053586, 0.7291320908465551,
                       0.39653931948230575, 0.07785205408506236};
const double DB8[]  = {-0.00011747678400228192, 0.0006754494059985568,
                       -0.0003917403729959771, -0.00487035299301066,
                       0.008746094047015655, 0.013981027917015516,
                       -0.04408825393106472, -0.01736930100202211,
                       0.128747426620186, 0.00047248457399797254,
                       -0.2840155429624281, -0.015829105256023893,
                       0.5853546836548691, 0.6756307362980128,
                       0.3128715909144659, 0.05441584224308161};
const double DB9[]  = {3.9347319995026124e-05, -0.0002519631889981789,
                       0.00023038576399541288, 0.0018476468829611268,
                       -0.0042815036819047227, -0.004723204757894831,
                       0.022361662123515244, 0.00025094711483145236,
                       -0.06763282905952399, 0.030725681478322865,
                       0.14854074933476008, -0.09684078322087904,
                       -0.29327378327258685, 0.13319738582208895,
                       0.6572880780366389, 0.6048231236767786,
                       0.24383467463766728, 0.03807794736316728};
const double SYM4[] = {-0.07576571478927333, -0.02963552764599851,
                       0.49761866763201545, 0.8037387518059161,
                       0.29785779560527736, -0.09921954357684722,
                       -0.012603967262037833, 0.0322231006040427};
const double SYM5[] = {0.027333068345077982, 0.029519490925774643,
                       -0.039134249302383094, 0.1993975339773936,
                       0.7234076904024206, 0.6339789634582119,
                       0.01660210576452232, -0.17532808990845047,
                       -0.021101834024758855, 0.019538882735286728};
const double SYM6[] = {0.015404109327027373, 0.0034907120842174702,
                       -0.11799011114819057, -0.048311742585633,
                       0.4910559419267466, 0.787641141030194,
                       0.3379294217276218, -0.07263752278646252,
                       -0.021060292512300564, 0.04472490177066578,
                       0.0017677118642428036, -0.007800708325034148};
const double SYM7[] = {0.002681814568257878, -0.0010473848886829163,
                       -0.01263630340325193, 0.03051551316596357,
                       0.0678926935013727, -0.049552834937127255,
                       0.017441255086855827, 0.5361019170917628,
                       0.767764317003164, 0.2886296317515146,
                       -0.14004724044296152, -0.10780823770381774,
                       0.004010244871533663, 0.010268176708511255};
const double SYM8[] = {-0.0033824159510061256, -0.0005421323317911481,
                       0.03169508781149298, 0.007607487324917605,
                       -0.1432942383508097, -0.061273359067658524,
                       0.4813596512583722, 0.7771857517005235,
                       0.3644418948353314, -0.05194583810770904,
                       -0.027219029917056003, 0.049137179673607506,
                       0.003808752013890615, -0.01495225833704823,
                       -0.0003029205147213668, 0.0018899503327594609};
const double COIF1[] = {-0.01565572813546454, -0.0727326195128539,
                        0.38486484686420286, 0.8525720202122554,
                        0.3378976624578092, -0.0727326195128539};
const double COIF2[] = {-0.0007205494453645122, -0.0018232088707029932,
                        0.0056114348193944995, 0.023680171946334084,
                        -0.0594344186464569, -0.0764885990783064,
                        0.41700518442169254, 0.8127236354455423,
                        0.3861100668211622, -0.06737255472196302,
                        -0.04146493678175915, 0.016387336463522112};
const double COIF3[] = {-3.459977283621256e-05, -7.098330313814125e-05,
                        0.0004662169601128863, 0.0011175187708906016,
                        -0.0025745176887502236, -0.00900797613666158,
                        0.015880544863615904, 0.03455502757306163,
                        -0.08230192710688598, -0.07179982161931202,
                        0.42848347637761874, 0.7937772226256206,
                        0.4051769024096150, -0.06112339000267287,
                        -0.0657719112818555, 0.023452696141836267,
                        0.007782596427325418, -0.003793512864491014};
const double COIF4[] = {-1.7849850030882614e-06, -3.2596802368833675e-06,
                        3.1229875865345646e-05, 6.233903446100713e-05,
                        -0.00025997455248771324, -0.0005890207562443383,
                        0.0012665619292989445, 0.003751436157278457,
                        -0.00565828668661072, -0.015211731527946259,
                        0.025082261844864097, 0.03933442712333749,
                        -0.09622044203398798, -0.06662747426342504,
                        0.4343860564914685, 0.782238930920499,
                        0.41530840703043026, -0.05607731331675481,
                        -0.08126669968087875, 0.026682300156053072,
                        0.016068943964776348, -0.0073461663276420935,
                        -0.0016294920126017326, 0.0008923136685823146};
const double COIF5[] = {-9.517657273819165e-08, -1.6744288576823017e-07,
                        2.0637618513646814e-06, 3.7346551751414047e-06,
                        -2.1315026809955787e-05, -4.134043227251251e-05,
                        0.00014054114970203437, 0.00030225958181306315,
                        -0.0006381313430451114, -0.0016628637020130838,
                        0.0024333732126576722, 0.006764185448053083,
                        -0.009164231162481846, -0.01976177894257264,
                        0.03268357426711183, 0.0412892087501817,
                        -0.10557420870333893, -0.06203596396290357,
                        0.4379916261718371, 0.7742896036529562,
                        0.4215662066908515, -0.05204316317624377,
                        -0.09192001055969624, 0.02816802897093635,
                        0.023408156785839195, -0.010131117519849788,
                        -0.004159358781386048, 0.0021782363581090178,
                        0.00035858968789573785, -0.00021208083980379827};

std::vector<double> arr2vec(const double *a, size_t n) {
    return std::vector<double>(a, a + n);
}

/* qmf: reverse + negate even (1-based) entries — the standard MATLAB qmf. */
std::vector<double> wv_qmf(const std::vector<double> &x) {
    size_t n = x.size();
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) y[i] = x[n - 1 - i];
    for (size_t i = 1; i < n; i += 2) y[i] = -y[i];
    return y;
}
std::vector<double> wv_rev(const std::vector<double> &x) {
    return std::vector<double>(x.rbegin(), x.rend());
}

/* Look up the scaling filter for an orthogonal family member. */
bool wv_scaling(const std::string &nm, std::vector<double> &lo) {
    auto S = [&](const double *a, size_t n) { lo = arr2vec(a, n); return true; };
    if (nm == "haar" || nm == "db1") return S(DB1, 2);
    if (nm == "db2" || nm == "sym2") return S(DB2, 4);
    if (nm == "db3" || nm == "sym3") return S(DB3, 6);
    if (nm == "db4") return S(DB4, 8);
    if (nm == "db5") return S(DB5, 10);
    if (nm == "db6") return S(DB6, 12);
    if (nm == "db7") return S(DB7, 14);
    if (nm == "db8") return S(DB8, 16);
    if (nm == "db9") return S(DB9, 18);
    if (nm == "sym4") return S(SYM4, 8);
    if (nm == "sym5") return S(SYM5, 10);
    if (nm == "sym6") return S(SYM6, 12);
    if (nm == "sym7") return S(SYM7, 14);
    if (nm == "sym8") return S(SYM8, 16);
    if (nm == "coif1") return S(COIF1, 6);
    if (nm == "coif2") return S(COIF2, 12);
    if (nm == "coif3") return S(COIF3, 18);
    if (nm == "coif4") return S(COIF4, 24);
    if (nm == "coif5") return S(COIF5, 30);
    return false;
}

/* Build the four filters for a named wavelet. */
WvFilters wv_filters(const std::string &nm) {
    WvFilters f; f.ortho = true;
    std::vector<double> lo;
    if (!wv_scaling(nm, lo)) {
        /* default to db4 for an unknown name, keeping the toolbox usable. */
        wv_scaling("db4", lo);
    }
    /* Normalise to sum √2 (defensive — the tables are already normalised). */
    double s = 0; for (double v : lo) s += v;
    if (fabs(s) > 1e-12) { double k = sqrt(2.0) / s; for (double &v : lo) v *= k; }
    f.LoD = lo;
    f.HiD = wv_qmf(lo);          /* QMF of the dec low-pass            */
    f.LoR = wv_rev(f.LoD);       /* reconstruction = time-reversal     */
    f.HiR = wv_rev(f.HiD);
    return f;
}

/* ===== circular orthonormal one-level DWT / IDWT ========================= */

/* Analysis: cA[n] = Σ_k Lo_D[k] x[(2n+k) mod N], cD likewise with Hi_D.
 * N must be even; on odd input the caller pads to even (recorded in L). */
void wv_dwt1(const std::vector<double> &x, const WvFilters &f,
             std::vector<double> &cA, std::vector<double> &cD) {
    int64_t N = static_cast<int64_t>(x.size());
    int64_t half = N / 2;
    int64_t lf = static_cast<int64_t>(f.LoD.size());
    cA.assign(static_cast<size_t>(half), 0.0);
    cD.assign(static_cast<size_t>(half), 0.0);
    for (int64_t n = 0; n < half; ++n) {
        double a = 0, d = 0;
        for (int64_t k = 0; k < lf; ++k) {
            int64_t idx = (2 * n + k) % N;
            if (idx < 0) idx += N;
            a += f.LoD[static_cast<size_t>(k)] * x[static_cast<size_t>(idx)];
            d += f.HiD[static_cast<size_t>(k)] * x[static_cast<size_t>(idx)];
        }
        cA[static_cast<size_t>(n)] = a;
        cD[static_cast<size_t>(n)] = d;
    }
}

/* Synthesis (W^T): x[m] = Σ_n cA[n] Lo_D[(m-2n) mod N] + cD[n] Hi_D[(m-2n) mod N]. */
std::vector<double> wv_idwt1(const std::vector<double> &cA,
                             const std::vector<double> &cD,
                             const WvFilters &f) {
    int64_t half = static_cast<int64_t>(cA.size());
    int64_t N = 2 * half;
    int64_t lf = static_cast<int64_t>(f.LoD.size());
    std::vector<double> x(static_cast<size_t>(N), 0.0);
    for (int64_t n = 0; n < half; ++n) {
        double a = cA[static_cast<size_t>(n)];
        double d = (n < static_cast<int64_t>(cD.size())) ? cD[static_cast<size_t>(n)] : 0.0;
        for (int64_t k = 0; k < lf; ++k) {
            int64_t m = (2 * n + k) % N;
            if (m < 0) m += N;
            x[static_cast<size_t>(m)] += a * f.LoD[static_cast<size_t>(k)]
                                       + d * f.HiD[static_cast<size_t>(k)];
        }
    }
    return x;
}

/* Pad a vector to even length by replicating the last sample. */
std::vector<double> wv_to_even(const std::vector<double> &x) {
    if (x.size() % 2 == 0) return x;
    std::vector<double> y = x;
    y.push_back(x.empty() ? 0.0 : x.back());
    return y;
}

/* maximum useful decomposition level. */
int64_t wv_maxlev(int64_t n, int64_t lf) {
    if (n < lf || lf < 2) return 0;
    int64_t lev = static_cast<int64_t>(floor(log2(static_cast<double>(n) /
                                               static_cast<double>(lf - 1))));
    return lev < 0 ? 0 : lev;
}

/* Multi-level decomposition: produce concatenated C and the L bookkeeping.
 * Layout matches MATLAB: C = [cA_n, cD_n, cD_{n-1}, ..., cD_1];
 * L = [len(cA_n), len(cD_n), ..., len(cD_1), len(signal)]. */
void wv_wavedec(const std::vector<double> &x0, int64_t lev, const WvFilters &f,
                std::vector<double> &C, std::vector<double> &L) {
    std::vector<double> approx = x0;
    std::vector<std::vector<double>> details;   /* fine → coarse */
    std::vector<int64_t> lens;
    int64_t origLen = static_cast<int64_t>(x0.size());
    for (int64_t l = 0; l < lev; ++l) {
        std::vector<double> ev = wv_to_even(approx);
        std::vector<double> cA, cD;
        wv_dwt1(ev, f, cA, cD);
        details.push_back(cD);
        lens.push_back(static_cast<int64_t>(cD.size()));
        approx = cA;
    }
    /* assemble C: cA (coarsest) then details coarse→fine. */
    C.clear();
    C.insert(C.end(), approx.begin(), approx.end());
    for (auto it = details.rbegin(); it != details.rend(); ++it)
        C.insert(C.end(), it->begin(), it->end());
    /* assemble L: len(cA), len(cD_n)...len(cD_1), origLen. */
    L.clear();
    L.push_back(static_cast<double>(approx.size()));
    for (auto it = lens.rbegin(); it != lens.rend(); ++it)
        L.push_back(static_cast<double>(*it));
    L.push_back(static_cast<double>(origLen));
}

/* Reconstruct the signal from [C, L]. */
std::vector<double> wv_waverec(const std::vector<double> &C,
                               const std::vector<double> &L,
                               const WvFilters &f) {
    if (L.size() < 2) return C;
    int64_t nlev = static_cast<int64_t>(L.size()) - 2;
    int64_t pos = 0;
    int64_t la = static_cast<int64_t>(L[0]);
    std::vector<double> approx(C.begin(), C.begin() + la);
    pos = la;
    for (int64_t l = 0; l < nlev; ++l) {
        int64_t ld = static_cast<int64_t>(L[1 + l]);
        std::vector<double> cD(C.begin() + pos, C.begin() + pos + ld);
        pos += ld;
        /* approx and cD should be equal length at each merge. */
        if (static_cast<int64_t>(approx.size()) > ld)
            approx.resize(static_cast<size_t>(ld));
        else while (static_cast<int64_t>(approx.size()) < ld) approx.push_back(0.0);
        approx = wv_idwt1(approx, cD, f);
    }
    int64_t origLen = static_cast<int64_t>(L.back());
    if (static_cast<int64_t>(approx.size()) > origLen)
        approx.resize(static_cast<size_t>(origLen));
    return approx;
}

/* Soft / hard threshold of a value. */
double wv_thr_one(double x, double t, bool soft) {
    if (soft) {
        double a = fabs(x) - t;
        return (a > 0) ? ((x > 0) ? a : -a) : 0.0;
    }
    return (fabs(x) > t) ? x : 0.0;
}

double wv_median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  /* anonymous namespace */

/* ===========================================================================
 * Public ABI — Tier-1
 * ==========================================================================*/
extern "C" {

/* ----- wfilters (4-return) -------------------------------------------------*/
matlab_mat *matlab_wavelet_wf_lod(void *wn) { return wv_row(wv_filters(wv_sstr(wn)).LoD); }
matlab_mat *matlab_wavelet_wf_hid(void *wn) { return wv_row(wv_filters(wv_sstr(wn)).HiD); }
matlab_mat *matlab_wavelet_wf_lor(void *wn) { return wv_row(wv_filters(wv_sstr(wn)).LoR); }
matlab_mat *matlab_wavelet_wf_hir(void *wn) { return wv_row(wv_filters(wv_sstr(wn)).HiR); }

/* ----- qmf -----------------------------------------------------------------*/
matlab_mat *matlab_wavelet_qmf(matlab_mat *x) {
    std::vector<double> v = wv_vec(x);
    return wv_oriented(wv_qmf(v), wv_is_col(x));
}

/* ----- dwt (2-return) ------------------------------------------------------*/
matlab_mat *matlab_wavelet_dwt_cA(matlab_mat *x, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> ev = wv_to_even(wv_vec(x)), cA, cD;
    wv_dwt1(ev, f, cA, cD);
    return wv_oriented(cA, wv_is_col(x));
}
matlab_mat *matlab_wavelet_dwt_cD(matlab_mat *x, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> ev = wv_to_even(wv_vec(x)), cA, cD;
    wv_dwt1(ev, f, cA, cD);
    return wv_oriented(cD, wv_is_col(x));
}

/* ----- idwt ----------------------------------------------------------------*/
matlab_mat *matlab_wavelet_idwt(matlab_mat *cA, matlab_mat *cD, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> a = wv_vec(cA), d = wv_vec(cD);
    size_t half = std::max(a.size(), d.size());
    a.resize(half, 0.0); d.resize(half, 0.0);
    return wv_oriented(wv_idwt1(a, d, f), wv_is_col(cA));
}

/* ----- wavedec (2-return) --------------------------------------------------*/
matlab_mat *matlab_wavelet_wavedec_C(matlab_mat *x, double lev, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> C, L;
    wv_wavedec(wv_vec(x), static_cast<int64_t>(lev), f, C, L);
    return wv_oriented(C, wv_is_col(x));
}
matlab_mat *matlab_wavelet_wavedec_L(matlab_mat *x, double lev, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> C, L;
    wv_wavedec(wv_vec(x), static_cast<int64_t>(lev), f, C, L);
    return wv_col(L);   /* MATLAB L is a column vector */
}

/* ----- waverec -------------------------------------------------------------*/
matlab_mat *matlab_wavelet_waverec(matlab_mat *C, matlab_mat *L, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    return wv_row(wv_waverec(wv_vec(C), wv_vec(L), f));
}

/* ----- appcoef / detcoef ---------------------------------------------------*/
matlab_mat *matlab_wavelet_appcoef(matlab_mat *C, matlab_mat *L, void *wn, double n) {
    (void)wn;
    std::vector<double> c = wv_vec(C), l = wv_vec(L);
    if (l.size() < 2) return mat_alloc(0, 0);
    int64_t nlev = static_cast<int64_t>(l.size()) - 2;
    int64_t want = static_cast<int64_t>(n);
    if (want <= 0 || want > nlev) want = nlev;   /* default: coarsest */
    /* approximation only available at the coarsest level in [C,L]; for a
     * higher requested level, re-synthesise by partial reconstruction. */
    int64_t la = static_cast<int64_t>(l[0]);
    std::vector<double> approx(c.begin(), c.begin() + la);
    if (want == nlev) return wv_row(approx);
    /* rebuild down to `want` by inverse-transforming detail levels nlev..want+1 */
    WvFilters f = wv_filters(wv_sstr(wn));
    int64_t pos = la;
    for (int64_t lev = nlev; lev > want; --lev) {
        int64_t ld = static_cast<int64_t>(l[1 + (nlev - lev)]);
        std::vector<double> cD(c.begin() + pos, c.begin() + pos + ld);
        pos += ld;
        approx.resize(static_cast<size_t>(ld), 0.0);
        approx = wv_idwt1(approx, cD, f);
    }
    return wv_row(approx);
}
matlab_mat *matlab_wavelet_detcoef(matlab_mat *C, matlab_mat *L, double n) {
    std::vector<double> c = wv_vec(C), l = wv_vec(L);
    if (l.size() < 2) return mat_alloc(0, 0);
    int64_t nlev = static_cast<int64_t>(l.size()) - 2;
    int64_t want = static_cast<int64_t>(n);
    if (want <= 0) want = 1;
    if (want > nlev) want = nlev;
    /* details are stored coarse→fine after the approximation. Level `nlev`
     * (coarsest) sits first, level 1 (finest) last. */
    int64_t pos = static_cast<int64_t>(l[0]);
    for (int64_t lev = nlev; lev >= 1; --lev) {
        int64_t ld = static_cast<int64_t>(l[1 + (nlev - lev)]);
        if (lev == want) {
            std::vector<double> cD(c.begin() + pos, c.begin() + pos + ld);
            return wv_row(cD);
        }
        pos += ld;
    }
    return mat_alloc(0, 0);
}

/* ----- wrcoef --------------------------------------------------------------*/
matlab_mat *matlab_wavelet_wrcoef(void *typ, matlab_mat *C, matlab_mat *L,
                                  void *wn, double n) {
    std::string t = wv_sstr(typ);
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> c = wv_vec(C), l = wv_vec(L);
    if (l.size() < 2) return mat_alloc(0, 0);
    int64_t nlev = static_cast<int64_t>(l.size()) - 2;
    int64_t want = static_cast<int64_t>(n);
    bool isApprox = (!t.empty() && (t[0] == 'a' || t[0] == 'A'));
    if (isApprox && want <= 0) want = nlev;
    if (!isApprox && want <= 0) want = 1;
    /* zero out everything except the requested branch, then waverec. */
    std::vector<double> Cm = c;
    int64_t la = static_cast<int64_t>(l[0]);
    if (!isApprox) {
        for (int64_t i = 0; i < la; ++i) Cm[static_cast<size_t>(i)] = 0;  /* drop approx */
    }
    int64_t pos = la;
    for (int64_t lev = nlev; lev >= 1; --lev) {
        int64_t ld = static_cast<int64_t>(l[1 + (nlev - lev)]);
        bool keep = (!isApprox && lev == want);
        if (!keep)
            for (int64_t i = 0; i < ld; ++i) Cm[static_cast<size_t>(pos + i)] = 0;
        pos += ld;
    }
    if (isApprox) {
        /* keep only approx + drop all details. */
        pos = la;
        for (int64_t lev = nlev; lev >= 1; --lev) {
            int64_t ld = static_cast<int64_t>(l[1 + (nlev - lev)]);
            for (int64_t i = 0; i < ld; ++i) Cm[static_cast<size_t>(pos + i)] = 0;
            pos += ld;
        }
    }
    return wv_row(wv_waverec(Cm, l, f));
}

/* ----- upcoef (single-branch upward reconstruction) ------------------------*/
matlab_mat *matlab_wavelet_upcoef(void *typ, matlab_mat *x, void *wn, double n) {
    std::string t = wv_sstr(typ);
    WvFilters f = wv_filters(wv_sstr(wn));
    std::vector<double> v = wv_vec(x);
    int64_t lev = static_cast<int64_t>(n); if (lev < 1) lev = 1;
    bool isApprox = (!t.empty() && (t[0] == 'a' || t[0] == 'A'));
    for (int64_t l = 0; l < lev; ++l) {
        std::vector<double> zero(v.size(), 0.0);
        v = isApprox ? wv_idwt1(v, zero, f) : wv_idwt1(zero, v, f);
    }
    return wv_oriented(v, wv_is_col(x));
}

/* ----- wmaxlev -------------------------------------------------------------*/
double matlab_wavelet_wmaxlev(matlab_mat *n, void *wn) {
    int64_t len;
    if (n && n->rows * n->cols >= 2) {
        /* size vector — use the larger dimension */
        len = static_cast<int64_t>(std::max(n->data[0], n->data[1]));
    } else {
        len = static_cast<int64_t>(wv_sc(n, 0));
    }
    WvFilters f = wv_filters(wv_sstr(wn));
    return static_cast<double>(wv_maxlev(len, static_cast<int64_t>(f.LoD.size())));
}

/* ----- wextend -------------------------------------------------------------*/
matlab_mat *matlab_wavelet_wextend(void *typ, void *mode, matlab_mat *x, double len) {
    (void)typ;
    std::string m = wv_sstr(mode);
    std::vector<double> v = wv_vec(x);
    int64_t N = static_cast<int64_t>(v.size());
    int64_t p = static_cast<int64_t>(len);
    if (N == 0 || p <= 0) return wv_oriented(v, wv_is_col(x));
    std::vector<double> out;
    auto at = [&](int64_t i) -> double {
        /* clamp helper used by zpd */
        if (i < 0 || i >= N) return 0.0;
        return v[static_cast<size_t>(i)];
    };
    if (m == "zpd") {
        for (int64_t i = 0; i < p; ++i) out.push_back(0.0);
        out.insert(out.end(), v.begin(), v.end());
        for (int64_t i = 0; i < p; ++i) out.push_back(0.0);
    } else if (m == "per") {
        std::vector<double> ve = wv_to_even(v);
        int64_t M = static_cast<int64_t>(ve.size());
        for (int64_t i = 0; i < p; ++i) out.push_back(ve[static_cast<size_t>(((M - p + i) % M + M) % M)]);
        out.insert(out.end(), ve.begin(), ve.end());
        for (int64_t i = 0; i < p; ++i) out.push_back(ve[static_cast<size_t>(i % M)]);
    } else if (m == "ppd") {
        for (int64_t i = 0; i < p; ++i) out.push_back(v[static_cast<size_t>(((N - p + i) % N + N) % N)]);
        out.insert(out.end(), v.begin(), v.end());
        for (int64_t i = 0; i < p; ++i) out.push_back(v[static_cast<size_t>(i % N)]);
    } else {
        /* default 'sym' (half-point symmetric) */
        for (int64_t i = 0; i < p; ++i) out.push_back(at(p - 1 - i < N ? p - 1 - i : N - 1));
        /* careful symmetric reflection of first p samples */
        out.clear();
        for (int64_t i = p - 1; i >= 0; --i) out.push_back(v[static_cast<size_t>(i % N)]);
        out.insert(out.end(), v.begin(), v.end());
        for (int64_t i = 0; i < p; ++i) out.push_back(v[static_cast<size_t>((N - 1 - (i % N)))]);
    }
    return wv_oriented(out, wv_is_col(x));
}

/* ----- wkeep ---------------------------------------------------------------*/
matlab_mat *matlab_wavelet_wkeep(matlab_mat *x, double len) {
    std::vector<double> v = wv_vec(x);
    int64_t N = static_cast<int64_t>(v.size());
    int64_t k = static_cast<int64_t>(len);
    if (k <= 0 || k >= N) return wv_oriented(v, wv_is_col(x));
    int64_t start = (N - k) / 2;     /* center crop */
    std::vector<double> out(v.begin() + start, v.begin() + start + k);
    return wv_oriented(out, wv_is_col(x));
}

/* ----- centfrq -------------------------------------------------------------*/
double matlab_wavelet_centfrq(void *wn) {
    /* center frequency from the dominant FFT bin of the wavelet function;
     * approximated from the high-pass filter peak for the orthogonal lane. */
    WvFilters f = wv_filters(wv_sstr(wn));
    int64_t L = static_cast<int64_t>(f.HiR.size());
    /* upsample-cascade a few iterations to approximate psi, then FFT-peak.
     * Cheap proxy: peak of |FFT(Hi_R)| normalised. */
    int64_t Nf = 1; while (Nf < 8 * L) Nf <<= 1;
    std::vector<std::complex<double>> H(static_cast<size_t>(Nf), {0, 0});
    for (int64_t k = 0; k < L; ++k) H[static_cast<size_t>(k)] = f.HiR[static_cast<size_t>(k)];
    /* naive DFT magnitude, find peak bin in [0, Nf/2) */
    double best = -1; int64_t bestk = 1;
    for (int64_t kf = 1; kf < Nf / 2; ++kf) {
        std::complex<double> s(0, 0);
        for (int64_t n = 0; n < L; ++n) {
            double ang = -2.0 * M_PI * static_cast<double>(kf) * static_cast<double>(n) /
                         static_cast<double>(Nf);
            s += f.HiR[static_cast<size_t>(n)] * std::complex<double>(cos(ang), sin(ang));
        }
        double mag = std::abs(s);
        if (mag > best) { best = mag; bestk = kf; }
    }
    return static_cast<double>(bestk) / static_cast<double>(Nf);
}

/* ----- wentropy ------------------------------------------------------------*/
double matlab_wavelet_wentropy(matlab_mat *x, void *typ) {
    std::string t = wv_sstr(typ);
    std::vector<double> v = wv_vec(x);
    double e = 0;
    if (t == "shannon" || t.empty()) {
        for (double a : v) { double s = a * a; if (s > 1e-300) e -= s * log(s); }
    } else if (t == "log energy" || t == "logenergy" || t == "log") {
        for (double a : v) { double s = a * a; if (s > 1e-300) e += log(s); }
    } else if (t == "norm") {
        for (double a : v) e += fabs(a);   /* p=1 norm by default */
    } else if (t == "threshold") {
        for (double a : v) if (fabs(a) > 0.2) e += 1;
    } else if (t == "sure") {
        for (double a : v) e += std::min(a * a, 1.0);
    } else {
        for (double a : v) { double s = a * a; if (s > 1e-300) e -= s * log(s); }
    }
    return e;
}

/* ----- wenergy -------------------------------------------------------------*/
matlab_mat *matlab_wavelet_wenergy(matlab_mat *C, matlab_mat *L) {
    std::vector<double> c = wv_vec(C), l = wv_vec(L);
    if (l.size() < 2) return mat_alloc(0, 0);
    int64_t nlev = static_cast<int64_t>(l.size()) - 2;
    double total = 0; for (double a : c) total += a * a;
    if (total < 1e-300) total = 1;
    std::vector<double> pct;
    int64_t pos = 0;
    int64_t la = static_cast<int64_t>(l[0]);
    double ea = 0; for (int64_t i = 0; i < la; ++i) ea += c[static_cast<size_t>(i)] * c[static_cast<size_t>(i)];
    pos = la;
    /* Ea then Ed per level (fine→coarse to match MATLAB's [Ea, Ed]) */
    std::vector<double> ed(static_cast<size_t>(nlev), 0.0);
    for (int64_t lev = nlev; lev >= 1; --lev) {
        int64_t ld = static_cast<int64_t>(l[1 + (nlev - lev)]);
        double e = 0; for (int64_t i = 0; i < ld; ++i) { double a = c[static_cast<size_t>(pos + i)]; e += a * a; }
        ed[static_cast<size_t>(lev - 1)] = 100.0 * e / total;
        pos += ld;
    }
    pct.push_back(100.0 * ea / total);
    for (int64_t lev = 1; lev <= nlev; ++lev) pct.push_back(ed[static_cast<size_t>(lev - 1)]);
    return wv_row(pct);
}

/* ----- dwtmode (global border mode; stored, mostly informational) ----------*/
static std::string g_dwtmode = "per";
matlab_mat *matlab_wavelet_dwtmode(void *mode) {
    std::string m = wv_sstr(mode);
    if (!m.empty() && m != "status") g_dwtmode = m;
    return wv_row(std::vector<double>{});   /* returns nothing meaningful */
}

}  /* extern "C" */

/* ===========================================================================
 * Tier-2 — denoising + nonparametric estimation + compression
 * ==========================================================================*/
namespace {

/* threshold-selection rule for a coefficient vector (assumes unit-variance
 * noise; the caller multiplies by the estimated σ).  Returns the threshold. */
double wv_thselect(const std::vector<double> &x, const std::string &rule) {
    int64_t n = static_cast<int64_t>(x.size());
    if (n <= 0) return 0.0;
    double universal = sqrt(2.0 * log(static_cast<double>(n)));
    if (rule == "sqtwolog" || rule.empty()) return universal;
    if (rule == "minimaxi") {
        if (n <= 32) return 0.0;
        return 0.3936 + 0.1829 * (log(static_cast<double>(n)) / log(2.0));
    }
    /* rigrsure — Stein's Unbiased Risk Estimate. */
    if (rule == "rigrsure" || rule == "heursure") {
        std::vector<double> sx2(x.size());
        for (size_t i = 0; i < x.size(); ++i) sx2[i] = x[i] * x[i];
        std::sort(sx2.begin(), sx2.end());
        double bestRisk = 1e300; double bestThr = universal;
        double cum = 0;
        for (int64_t k = 0; k < n; ++k) {
            cum += sx2[static_cast<size_t>(k)];
            double risk = (static_cast<double>(n) - 2.0 * static_cast<double>(k + 1) +
                           cum + static_cast<double>(n - (k + 1)) * sx2[static_cast<size_t>(k)]) /
                          static_cast<double>(n);
            if (risk < bestRisk) { bestRisk = risk; bestThr = sqrt(sx2[static_cast<size_t>(k)]); }
        }
        if (rule == "rigrsure") return bestThr;
        /* heursure: pick between SURE and universal by an energy test. */
        double sumsq = cum;
        double eta = (sumsq - static_cast<double>(n)) / static_cast<double>(n);
        double crit = pow(log(static_cast<double>(n)) / log(2.0), 1.5) /
                      sqrt(static_cast<double>(n));
        if (eta < crit) return universal;
        return std::min(bestThr, universal);
    }
    return universal;
}

/* robust noise σ = median(|d|) / 0.6745. */
double wv_madsigma(const std::vector<double> &d) {
    std::vector<double> a(d.size());
    for (size_t i = 0; i < d.size(); ++i) a[i] = fabs(d[i]);
    return wv_median(a) / 0.6745;
}

/* full denoising over [C,L]: keep approx, threshold details level-by-level. */
std::vector<double> wv_denoise_core(const std::vector<double> &x, int64_t level,
                                    const std::string &wname,
                                    const std::string &rule, bool soft,
                                    bool levelDependent) {
    WvFilters f = wv_filters(wname);
    std::vector<double> C, L;
    if (level <= 0) level = wv_maxlev(static_cast<int64_t>(x.size()),
                                      static_cast<int64_t>(f.LoD.size()));
    if (level < 1) level = 1;
    wv_wavedec(x, level, f, C, L);
    int64_t nlev = static_cast<int64_t>(L.size()) - 2;
    int64_t la = static_cast<int64_t>(L[0]);
    /* global σ from finest details (level 1, the last block in C). */
    int64_t pos = la;
    std::vector<std::pair<int64_t, int64_t>> blocks;   /* (start,len) per detail, coarse→fine */
    for (int64_t lev = nlev; lev >= 1; --lev) {
        int64_t ld = static_cast<int64_t>(L[1 + (nlev - lev)]);
        blocks.push_back({pos, ld});
        pos += ld;
    }
    /* finest details = last block. */
    std::vector<double> finest(C.begin() + blocks.back().first,
                               C.begin() + blocks.back().first + blocks.back().second);
    double sigmaGlobal = wv_madsigma(finest);
    for (auto &b : blocks) {
        std::vector<double> d(C.begin() + b.first, C.begin() + b.first + b.second);
        double sigma = levelDependent ? wv_madsigma(d) : sigmaGlobal;
        if (sigma < 1e-30) sigma = sigmaGlobal;
        double thr = sigma * wv_thselect(d, rule);
        for (int64_t i = 0; i < b.second; ++i)
            C[static_cast<size_t>(b.first + i)] =
                wv_thr_one(C[static_cast<size_t>(b.first + i)], thr, soft);
    }
    return wv_waverec(C, L, f);
}

/* ----- test-signal generators (wnoise FUN 1..6 on t ∈ [0,1]) -------------- */
std::vector<double> wv_testsig(int fun, int64_t N) {
    std::vector<double> x(static_cast<size_t>(N), 0.0);
    auto tval = [&](int64_t i) { return static_cast<double>(i) / static_cast<double>(N); };
    if (fun == 1) {   /* Blocks */
        double pos[] = {0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81};
        double hgt[] = {4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 5.1, -4.2};
        for (int64_t i = 0; i < N; ++i) {
            double t = tval(i), s = 0;
            for (int j = 0; j < 11; ++j) s += hgt[j] * (1.0 + ((t - pos[j]) >= 0 ? 1.0 : -1.0)) / 2.0;
            x[static_cast<size_t>(i)] = s;
        }
    } else if (fun == 2) {   /* Bumps */
        double pos[] = {0.1, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81};
        double hgt[] = {4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2};
        double wid[] = {0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005};
        for (int64_t i = 0; i < N; ++i) {
            double t = tval(i), s = 0;
            for (int j = 0; j < 11; ++j) {
                double u = fabs(t - pos[j]) / wid[j];
                s += hgt[j] / pow(1.0 + u, 4.0);
            }
            x[static_cast<size_t>(i)] = s;
        }
    } else if (fun == 3) {   /* HeavySine */
        for (int64_t i = 0; i < N; ++i) {
            double t = tval(i);
            x[static_cast<size_t>(i)] = 4.0 * sin(4.0 * M_PI * t)
                - ((t - 0.3) >= 0 ? 1.0 : -1.0) - ((0.72 - t) >= 0 ? 1.0 : -1.0);
        }
    } else if (fun == 4) {   /* Doppler */
        for (int64_t i = 0; i < N; ++i) {
            double t = tval(i);
            x[static_cast<size_t>(i)] = sqrt(t * (1.0 - t)) *
                sin(2.0 * M_PI * 1.05 / (t + 0.05));
        }
    } else if (fun == 5) {   /* Quadchirp */
        for (int64_t i = 0; i < N; ++i) {
            double t = tval(i);
            x[static_cast<size_t>(i)] = sin(M_PI / 3.0 * t * (N * t * t));
        }
    } else {                 /* Mishmash (default) */
        for (int64_t i = 0; i < N; ++i) {
            double t = tval(i);
            x[static_cast<size_t>(i)] = sin(M_PI / 3.0 * t * (N * t * t))
                + sin(M_PI * t * 0.6902 * N) + sin(M_PI * t * (0.125 * N) * t);
        }
    }
    return x;
}

/* thread-local cache so [x,xn]=wnoise(...) returns a consistent pair. */
thread_local std::vector<double> g_wnoise_clean, g_wnoise_noisy;

}  /* anonymous namespace */

extern "C" {

/* ----- wthresh -------------------------------------------------------------*/
matlab_mat *matlab_wavelet_wthresh(matlab_mat *x, void *sorh, double t) {
    std::string s = wv_sstr(sorh);
    bool soft = !(s.size() && (s[0] == 'h' || s[0] == 'H'));
    std::vector<double> v = wv_vec(x);
    for (double &a : v) a = wv_thr_one(a, t, soft);
    return wv_oriented(v, wv_is_col(x));
}

/* ----- thselect ------------------------------------------------------------*/
double matlab_wavelet_thselect(matlab_mat *x, void *rule) {
    return wv_thselect(wv_vec(x), wv_sstr(rule));
}

/* ----- wnoisest ------------------------------------------------------------*/
double matlab_wavelet_wnoisest1(matlab_mat *x) { return wv_madsigma(wv_vec(x)); }
double matlab_wavelet_wnoisest3(matlab_mat *C, matlab_mat *L, double level) {
    matlab_mat *d = matlab_wavelet_detcoef(C, L, level <= 0 ? 1.0 : level);
    double s = wv_madsigma(wv_vec(d));
    return s;
}

/* ----- wnoise --------------------------------------------------------------*/
matlab_mat *matlab_wavelet_wnoise_x(double fun, double n) {
    int64_t N = static_cast<int64_t>(pow(2.0, n));
    return wv_row(wv_testsig(static_cast<int>(fun), N));
}
matlab_mat *matlab_wavelet_wnoise_x3(double fun, double n, double snr) {
    int64_t N = static_cast<int64_t>(pow(2.0, n));
    std::vector<double> x = wv_testsig(static_cast<int>(fun), N);
    /* scale so std(x) == snr. */
    double mean = 0; for (double a : x) mean += a; mean /= static_cast<double>(N);
    double var = 0; for (double a : x) var += (a - mean) * (a - mean); var /= static_cast<double>(N);
    double sd = sqrt(var); if (sd < 1e-30) sd = 1;
    double k = snr / sd; for (double &a : x) a *= k;
    g_wnoise_clean = x;
    matlab_mat *noise = matlab_randn(1, static_cast<double>(N));
    g_wnoise_noisy = x;
    if (noise && noise->data)
        for (int64_t i = 0; i < N; ++i) g_wnoise_noisy[static_cast<size_t>(i)] += noise->data[i];
    return wv_row(g_wnoise_clean);
}
matlab_mat *matlab_wavelet_wnoise_xn3(double fun, double n, double snr) {
    if (g_wnoise_noisy.empty()) matlab_wavelet_wnoise_x3(fun, n, snr);
    return wv_row(g_wnoise_noisy);
}

/* ----- wden (legacy automatic denoising, signal form) ----------------------*/
matlab_mat *matlab_wavelet_wden(matlab_mat *x, void *tptr, void *sorh,
                                void *scal, double n, void *wname) {
    std::string rule = wv_sstr(tptr);
    std::string s = wv_sstr(sorh);
    std::string sc = wv_sstr(scal);
    bool soft = !(s.size() && (s[0] == 'h' || s[0] == 'H'));
    bool levelDep = (sc == "mln" || sc == "sln" ? (sc == "mln") : false);
    std::vector<double> out = wv_denoise_core(wv_vec(x), static_cast<int64_t>(n),
                                              wv_sstr(wname), rule, soft, levelDep);
    return wv_oriented(out, wv_is_col(x));
}

/* ----- wdenoise (modern API, positional wavelet) ---------------------------*/
matlab_mat *matlab_wavelet_wdenoise3(matlab_mat *x, double level, void *wname) {
    std::vector<double> out = wv_denoise_core(wv_vec(x), static_cast<int64_t>(level),
                                              wv_sstr(wname), "sqtwolog", true, false);
    return wv_oriented(out, wv_is_col(x));
}
matlab_mat *matlab_wavelet_wdenoise2(matlab_mat *x, double level) {
    std::vector<double> out = wv_denoise_core(wv_vec(x), static_cast<int64_t>(level),
                                              "sym4", "sqtwolog", true, false);
    return wv_oriented(out, wv_is_col(x));
}

/* ----- wcompress (coefficient-thresholding compression) --------------------*/
matlab_mat *matlab_wavelet_wcompress(matlab_mat *x, double level, void *wname) {
    /* keep the largest-magnitude coefficients capturing ~99.9% energy. */
    WvFilters f = wv_filters(wv_sstr(wname));
    std::vector<double> C, L;
    int64_t lev = static_cast<int64_t>(level); if (lev < 1) lev = 5;
    wv_wavedec(wv_vec(x), lev, f, C, L);
    std::vector<double> mag(C.size());
    for (size_t i = 0; i < C.size(); ++i) mag[i] = C[i] * C[i];
    double total = 0; for (double m : mag) total += m;
    std::vector<double> sorted = mag;
    std::sort(sorted.rbegin(), sorted.rend());
    double cum = 0, thr = 0;
    for (double m : sorted) { cum += m; if (cum >= 0.999 * total) { thr = m; break; } }
    for (size_t i = 0; i < C.size(); ++i) if (mag[i] < thr) C[i] = 0;
    std::vector<double> out = wv_waverec(C, L, f);
    matlab_mat *r = mat_alloc(1, static_cast<int64_t>(out.size()));
    if (r && r->data) memcpy(r->data, out.data(), out.size() * sizeof(double));
    return r;
}

/* ----- measerr (PSNR by default) -------------------------------------------*/
double matlab_wavelet_measerr(matlab_mat *xref, matlab_mat *xapp) {
    std::vector<double> a = wv_vec(xref), b = wv_vec(xapp);
    size_t n = std::min(a.size(), b.size());
    if (n == 0) return 0.0;
    double mse = 0, peak = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = a[i] - b[i]; mse += e * e;
        if (fabs(a[i]) > peak) peak = fabs(a[i]);
    }
    mse /= static_cast<double>(n);
    if (mse < 1e-300) return 1e3;
    if (peak < 1e-30) peak = 1;
    return 10.0 * log10(peak * peak / mse);
}

}  /* extern "C" */

/* ===========================================================================
 * Tier-3 — continuous wavelet transform + time-frequency
 * ==========================================================================*/
extern "C" matlab_mat_c *matlab_ifft_c(void *Aptr);

namespace {

const double WV_MORLET_W0 = 6.0;   /* Morlet center frequency (rad/sample) */
const int    WV_NV        = 12;    /* voices per octave */

/* deterministic log-spaced scale set for a length-N signal. */
std::vector<double> wv_cwt_scales(int64_t N) {
    double s0 = 2.0;
    double smax = static_cast<double>(N) / 8.0;
    if (smax < s0 * 4) smax = s0 * 4;
    int numOct = static_cast<int>(floor(log2(smax / s0)));
    if (numOct < 1) numOct = 1;
    int ns = numOct * WV_NV;
    std::vector<double> a(static_cast<size_t>(ns));
    for (int j = 0; j < ns; ++j)
        a[static_cast<size_t>(j)] = s0 * pow(2.0, static_cast<double>(j) / WV_NV);
    return a;
}

/* one-shot FFT of a real vector via the shipped complex FFT. */
void wv_fft(const std::vector<double> &x, std::vector<double> &re,
            std::vector<double> &im) {
    int64_t N = static_cast<int64_t>(x.size());
    matlab_mat *m = wv_row(x);
    matlab_mat_c *F = matlab_fft_c(m);
    re.assign(static_cast<size_t>(N), 0.0);
    im.assign(static_cast<size_t>(N), 0.0);
    if (F && mat_is_complex(F) && F->re && F->im)
        for (int64_t i = 0; i < N; ++i) { re[static_cast<size_t>(i)] = F->re[i]; im[static_cast<size_t>(i)] = F->im[i]; }
}

/* compute the complex CWT coefficient matrix (ns × N), row 0 = highest freq. */
matlab_mat_c *wv_cwt_coeffs(const std::vector<double> &x) {
    int64_t N = static_cast<int64_t>(x.size());
    if (N < 4) return mat_c_alloc(0, 0);
    std::vector<double> scales = wv_cwt_scales(N);
    int64_t ns = static_cast<int64_t>(scales.size());
    std::vector<double> xr, xi;
    wv_fft(x, xr, xi);
    matlab_mat_c *W = mat_c_alloc(ns, N);
    if (!W) return W;
    std::vector<double> pr(static_cast<size_t>(N)), pi(static_cast<size_t>(N));
    for (int64_t j = 0; j < ns; ++j) {
        double a = scales[static_cast<size_t>(j)];
        /* analytic Morlet frequency response, multiply spectrum. */
        for (int64_t k = 0; k < N; ++k) {
            double wk = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(N);
            double psi = 0.0;
            if (k > 0 && k <= N / 2) {
                double arg = a * wk - WV_MORLET_W0;
                psi = pow(M_PI, -0.25) * sqrt(2.0 * a) * exp(-0.5 * arg * arg);
            }
            pr[static_cast<size_t>(k)] = xr[static_cast<size_t>(k)] * psi;
            pi[static_cast<size_t>(k)] = xi[static_cast<size_t>(k)] * psi;
        }
        /* ifft → row of W. */
        matlab_mat_c *in = mat_c_alloc(1, N);
        memcpy(in->re, pr.data(), static_cast<size_t>(N) * sizeof(double));
        memcpy(in->im, pi.data(), static_cast<size_t>(N) * sizeof(double));
        matlab_mat_c *row = matlab_ifft_c(in);
        if (row && row->re && row->im)
            for (int64_t k = 0; k < N; ++k) {
                W->re[j * N + k] = row->re[k];
                W->im[j * N + k] = row->im[k];
            }
    }
    return W;
}

}  /* anonymous namespace */

extern "C" {

/* ----- cwt (complex coefficient matrix + frequency vector) -----------------*/
matlab_mat_c *matlab_wavelet_cwt_mag(matlab_mat *x, double fs) {
    (void)fs;
    return wv_cwt_coeffs(wv_vec(x));
}
matlab_mat *matlab_wavelet_cwt_f(matlab_mat *x, double fs) {
    int64_t N = x ? x->rows * x->cols : 0;
    std::vector<double> scales = wv_cwt_scales(N);
    std::vector<double> f(scales.size());
    for (size_t j = 0; j < scales.size(); ++j)
        f[j] = WV_MORLET_W0 * fs / (2.0 * M_PI * scales[j]);
    return wv_col(f);
}

/* ----- icwt (approximate Torrence-Compo synthesis) -------------------------*/
matlab_mat *matlab_wavelet_icwt(matlab_mat *Wptr) {
    if (!Wptr || !mat_is_complex(Wptr)) return mat_alloc(0, 0);
    matlab_mat_c *W = reinterpret_cast<matlab_mat_c *>(Wptr);
    int64_t ns = W->rows, N = W->cols;
    if (ns <= 0 || N <= 0) return mat_alloc(0, 0);
    std::vector<double> scales = wv_cwt_scales(N);
    if (static_cast<int64_t>(scales.size()) != ns) scales = wv_cwt_scales(N);
    std::vector<double> x(static_cast<size_t>(N), 0.0);
    double dj = 1.0 / WV_NV;
    double Cdelta = 0.776, psi0 = pow(M_PI, -0.25);
    double pref = dj * 1.0 / (Cdelta * psi0);
    for (int64_t k = 0; k < N; ++k) {
        double s = 0;
        for (int64_t j = 0; j < ns; ++j)
            s += W->re[j * N + k] / sqrt(scales[static_cast<size_t>(j)]);
        x[static_cast<size_t>(k)] = pref * s;
    }
    return wv_row(x);
}

/* ----- scal2frq / freq2scal ------------------------------------------------*/
matlab_mat *matlab_wavelet_scal2frq(matlab_mat *a, void *wn, double dt) {
    double fc = matlab_wavelet_centfrq(wn);
    if (dt <= 0) dt = 1.0;
    std::vector<double> av = wv_vec(a), f(av.size());
    for (size_t i = 0; i < av.size(); ++i)
        f[i] = (av[i] != 0) ? fc / (av[i] * dt) : 0.0;
    return wv_oriented(f, wv_is_col(a));
}
matlab_mat *matlab_wavelet_freq2scal(matlab_mat *f, void *wn, double dt) {
    double fc = matlab_wavelet_centfrq(wn);
    if (dt <= 0) dt = 1.0;
    std::vector<double> fv = wv_vec(f), a(fv.size());
    for (size_t i = 0; i < fv.size(); ++i)
        a[i] = (fv[i] != 0) ? fc / (fv[i] * dt) : 0.0;
    return wv_oriented(a, wv_is_col(f));
}

/* ----- wcoherence (smoothed magnitude-squared coherence) -------------------*/
matlab_mat *matlab_wavelet_wcoherence(matlab_mat *x, matlab_mat *y) {
    std::vector<double> xv = wv_vec(x), yv = wv_vec(y);
    size_t N = std::min(xv.size(), yv.size());
    xv.resize(N); yv.resize(N);
    matlab_mat_c *Wx = wv_cwt_coeffs(xv);
    matlab_mat_c *Wy = wv_cwt_coeffs(yv);
    if (!Wx || !Wy) return mat_alloc(0, 0);
    int64_t ns = Wx->rows, n = Wx->cols;
    matlab_mat *R = mat_alloc(ns, n);
    /* boxcar smoothing in time (window 8) of cross/auto spectra. */
    int win = 8;
    for (int64_t j = 0; j < ns; ++j) {
        for (int64_t k = 0; k < n; ++k) {
            double sxy_r = 0, sxy_i = 0, sxx = 0, syy = 0;
            for (int64_t w = -win; w <= win; ++w) {
                int64_t idx = k + w; if (idx < 0 || idx >= n) continue;
                double xr = Wx->re[j * n + idx], xi = Wx->im[j * n + idx];
                double yr = Wy->re[j * n + idx], yi = Wy->im[j * n + idx];
                sxy_r += xr * yr + xi * yi;
                sxy_i += xi * yr - xr * yi;
                sxx += xr * xr + xi * xi;
                syy += yr * yr + yi * yi;
            }
            double num = sxy_r * sxy_r + sxy_i * sxy_i;
            double den = sxx * syy;
            R->data[j * n + k] = (den > 1e-30) ? num / den : 0.0;
        }
    }
    return R;
}

}  /* extern "C" */

/* ===========================================================================
 * Tier-4 — undecimated transforms (SWT / MODWT) + 2-D
 * ==========================================================================*/
namespace {

/* one MODWT level: filter V_{j-1} with the level-j (à-trous) filters. */
void wv_modwt_level(const std::vector<double> &V, const std::vector<double> &ht,
                    const std::vector<double> &gt, int64_t upfac,
                    std::vector<double> &W, std::vector<double> &Vout) {
    int64_t N = static_cast<int64_t>(V.size());
    int64_t L = static_cast<int64_t>(ht.size());
    W.assign(static_cast<size_t>(N), 0.0);
    Vout.assign(static_cast<size_t>(N), 0.0);
    for (int64_t t = 0; t < N; ++t) {
        double w = 0, v = 0;
        for (int64_t l = 0; l < L; ++l) {
            int64_t idx = (t - upfac * l) % N; if (idx < 0) idx += N;
            w += ht[static_cast<size_t>(l)] * V[static_cast<size_t>(idx)];
            v += gt[static_cast<size_t>(l)] * V[static_cast<size_t>(idx)];
        }
        W[static_cast<size_t>(t)] = w;
        Vout[static_cast<size_t>(t)] = v;
    }
}

/* full MODWT → (J+1) × N row-major matrix (rows W_1..W_J, then V_J). */
matlab_mat *wv_modwt(const std::vector<double> &x, const std::string &wname, int64_t J) {
    int64_t N = static_cast<int64_t>(x.size());
    if (N < 2) return mat_alloc(0, 0);
    if (J <= 0) J = static_cast<int64_t>(floor(log2(static_cast<double>(N))));
    if (J < 1) J = 1;
    WvFilters f = wv_filters(wname);
    std::vector<double> ht(f.HiD.size()), gt(f.LoD.size());
    for (size_t i = 0; i < ht.size(); ++i) ht[i] = f.HiD[i] / sqrt(2.0);
    for (size_t i = 0; i < gt.size(); ++i) gt[i] = f.LoD[i] / sqrt(2.0);
    matlab_mat *out = mat_alloc(J + 1, N);
    std::vector<double> V = x, W, Vout;
    for (int64_t j = 1; j <= J; ++j) {
        int64_t upfac = static_cast<int64_t>(1) << (j - 1);
        wv_modwt_level(V, ht, gt, upfac, W, Vout);
        for (int64_t t = 0; t < N; ++t) out->data[(j - 1) * N + t] = W[static_cast<size_t>(t)];
        V = Vout;
    }
    for (int64_t t = 0; t < N; ++t) out->data[J * N + t] = V[static_cast<size_t>(t)];
    return out;
}

/* inverse MODWT from a (J+1) × N matrix. */
std::vector<double> wv_imodwt(const matlab_mat *Wm, const std::string &wname) {
    if (!Wm || Wm->rows < 2) return {};
    int64_t J = Wm->rows - 1, N = Wm->cols;
    WvFilters f = wv_filters(wname);
    std::vector<double> ht(f.HiD.size()), gt(f.LoD.size());
    for (size_t i = 0; i < ht.size(); ++i) ht[i] = f.HiD[i] / sqrt(2.0);
    for (size_t i = 0; i < gt.size(); ++i) gt[i] = f.LoD[i] / sqrt(2.0);
    int64_t L = static_cast<int64_t>(ht.size());
    std::vector<double> V(static_cast<size_t>(N));
    for (int64_t t = 0; t < N; ++t) V[static_cast<size_t>(t)] = Wm->data[J * N + t];
    for (int64_t j = J; j >= 1; --j) {
        int64_t upfac = static_cast<int64_t>(1) << (j - 1);
        std::vector<double> Vprev(static_cast<size_t>(N), 0.0);
        for (int64_t t = 0; t < N; ++t) {
            double v = 0;
            for (int64_t l = 0; l < L; ++l) {
                int64_t idx = (t + upfac * l) % N; if (idx < 0) idx += N;
                v += ht[static_cast<size_t>(l)] * Wm->data[(j - 1) * N + idx]
                   + gt[static_cast<size_t>(l)] * V[static_cast<size_t>(idx)];
            }
            Vprev[static_cast<size_t>(t)] = v;
        }
        V = Vprev;
    }
    return V;
}

/* ----- separable 2-D one-level analysis / synthesis (circular) ------------ */
struct Mat2 { std::vector<double> d; int64_t r, c; };
Mat2 mat2_from(const matlab_mat *m) {
    Mat2 M; M.r = m ? m->rows : 0; M.c = m ? m->cols : 0;
    M.d.assign(M.r * M.c, 0.0);
    if (m && m->data) memcpy(M.d.data(), m->data, static_cast<size_t>(M.r * M.c) * sizeof(double));
    return M;
}
matlab_mat *mat2_to(const Mat2 &M) {
    matlab_mat *r = mat_alloc(M.r, M.c);
    if (r && r->data) memcpy(r->data, M.d.data(), static_cast<size_t>(M.r * M.c) * sizeof(double));
    return r;
}

/* analysis: returns the four R/2 × C/2 subbands. */
void wv_dwt2_one(const Mat2 &M, const WvFilters &f,
                 Mat2 &cA, Mat2 &cH, Mat2 &cV, Mat2 &cD) {
    int64_t R = M.r, C = M.c;
    int64_t Rh = R / 2, Ch = C / 2;
    /* row pass → L (R × Ch) and H (R × Ch). */
    Mat2 Lr{std::vector<double>(R * Ch, 0.0), R, Ch};
    Mat2 Hr{std::vector<double>(R * Ch, 0.0), R, Ch};
    for (int64_t i = 0; i < R; ++i) {
        std::vector<double> row(static_cast<size_t>(C));
        for (int64_t k = 0; k < C; ++k) row[static_cast<size_t>(k)] = M.d[i * C + k];
        std::vector<double> a, d; wv_dwt1(row, f, a, d);
        for (int64_t k = 0; k < Ch; ++k) { Lr.d[i * Ch + k] = a[static_cast<size_t>(k)]; Hr.d[i * Ch + k] = d[static_cast<size_t>(k)]; }
    }
    /* column pass on L and H. */
    cA = {std::vector<double>(Rh * Ch, 0.0), Rh, Ch};
    cH = {std::vector<double>(Rh * Ch, 0.0), Rh, Ch};
    cV = {std::vector<double>(Rh * Ch, 0.0), Rh, Ch};
    cD = {std::vector<double>(Rh * Ch, 0.0), Rh, Ch};
    for (int64_t k = 0; k < Ch; ++k) {
        std::vector<double> colL(static_cast<size_t>(R)), colH(static_cast<size_t>(R));
        for (int64_t i = 0; i < R; ++i) { colL[static_cast<size_t>(i)] = Lr.d[i * Ch + k]; colH[static_cast<size_t>(i)] = Hr.d[i * Ch + k]; }
        std::vector<double> aL, dL, aH, dH;
        wv_dwt1(colL, f, aL, dL);
        wv_dwt1(colH, f, aH, dH);
        for (int64_t i = 0; i < Rh; ++i) {
            cA.d[i * Ch + k] = aL[static_cast<size_t>(i)];
            cH.d[i * Ch + k] = dL[static_cast<size_t>(i)];
            cV.d[i * Ch + k] = aH[static_cast<size_t>(i)];
            cD.d[i * Ch + k] = dH[static_cast<size_t>(i)];
        }
    }
}

/* synthesis: rebuild R × C from the four subbands. */
Mat2 wv_idwt2_one(const Mat2 &cA, const Mat2 &cH, const Mat2 &cV, const Mat2 &cD,
                  const WvFilters &f) {
    int64_t Rh = cA.r, Ch = cA.c, R = 2 * Rh, C = 2 * Ch;
    Mat2 Lr{std::vector<double>(R * Ch, 0.0), R, Ch};
    Mat2 Hr{std::vector<double>(R * Ch, 0.0), R, Ch};
    for (int64_t k = 0; k < Ch; ++k) {
        std::vector<double> aL(static_cast<size_t>(Rh)), dL(static_cast<size_t>(Rh)),
                            aH(static_cast<size_t>(Rh)), dH(static_cast<size_t>(Rh));
        for (int64_t i = 0; i < Rh; ++i) {
            aL[static_cast<size_t>(i)] = cA.d[i * Ch + k]; dL[static_cast<size_t>(i)] = cH.d[i * Ch + k];
            aH[static_cast<size_t>(i)] = cV.d[i * Ch + k]; dH[static_cast<size_t>(i)] = cD.d[i * Ch + k];
        }
        std::vector<double> colL = wv_idwt1(aL, dL, f);
        std::vector<double> colH = wv_idwt1(aH, dH, f);
        for (int64_t i = 0; i < R; ++i) { Lr.d[i * Ch + k] = colL[static_cast<size_t>(i)]; Hr.d[i * Ch + k] = colH[static_cast<size_t>(i)]; }
    }
    Mat2 out{std::vector<double>(R * C, 0.0), R, C};
    for (int64_t i = 0; i < R; ++i) {
        std::vector<double> a(static_cast<size_t>(Ch)), d(static_cast<size_t>(Ch));
        for (int64_t k = 0; k < Ch; ++k) { a[static_cast<size_t>(k)] = Lr.d[i * Ch + k]; d[static_cast<size_t>(k)] = Hr.d[i * Ch + k]; }
        std::vector<double> row = wv_idwt1(a, d, f);
        for (int64_t k = 0; k < C; ++k) out.d[i * C + k] = row[static_cast<size_t>(k)];
    }
    return out;
}

}  /* anonymous namespace */

extern "C" {

/* ----- modwt / imodwt / modwtmra / modwtvar --------------------------------*/
matlab_mat *matlab_wavelet_modwt3(matlab_mat *x, void *wn, double J) {
    return wv_modwt(wv_vec(x), wv_sstr(wn), static_cast<int64_t>(J));
}
matlab_mat *matlab_wavelet_modwt2(matlab_mat *x, void *wn) {
    return wv_modwt(wv_vec(x), wv_sstr(wn), 0);
}
matlab_mat *matlab_wavelet_imodwt2(matlab_mat *W, void *wn) {
    return wv_row(wv_imodwt(W, wv_sstr(wn)));
}
matlab_mat *matlab_wavelet_imodwt1(matlab_mat *W) {
    return wv_row(wv_imodwt(W, "sym4"));
}
matlab_mat *matlab_wavelet_modwtmra2(matlab_mat *W, void *wn) {
    if (!W || W->rows < 2) return mat_alloc(0, 0);
    std::string wname = wv_sstr(wn);
    int64_t rows = W->rows, N = W->cols;
    matlab_mat *out = mat_alloc(rows, N);
    for (int64_t j = 0; j < rows; ++j) {
        matlab_mat *tmp = mat_alloc(rows, N);
        for (int64_t t = 0; t < N; ++t) tmp->data[j * N + t] = W->data[j * N + t];
        std::vector<double> comp = wv_imodwt(tmp, wname);
        for (int64_t t = 0; t < N; ++t) out->data[j * N + t] = comp[static_cast<size_t>(t)];
    }
    return out;
}
matlab_mat *matlab_wavelet_modwtmra1(matlab_mat *W) { return matlab_wavelet_modwtmra2(W, nullptr); }
matlab_mat *matlab_wavelet_modwtvar(matlab_mat *W) {
    if (!W || W->rows < 1) return mat_alloc(0, 0);
    int64_t rows = W->rows, N = W->cols;
    std::vector<double> v(static_cast<size_t>(rows));
    for (int64_t j = 0; j < rows; ++j) {
        double s = 0; for (int64_t t = 0; t < N; ++t) { double a = W->data[j * N + t]; s += a * a; }
        v[static_cast<size_t>(j)] = s / static_cast<double>(N);
    }
    return wv_col(v);
}

/* ----- swt / iswt (undecimated; reuses the MODWT engine) -------------------*/
matlab_mat *matlab_wavelet_swt(matlab_mat *x, double n, void *wn) {
    return wv_modwt(wv_vec(x), wv_sstr(wn), static_cast<int64_t>(n));
}
matlab_mat *matlab_wavelet_iswt(matlab_mat *swc, void *wn) {
    return wv_row(wv_imodwt(swc, wv_sstr(wn)));
}

/* ----- dwt2 / idwt2 --------------------------------------------------------*/
matlab_mat *matlab_wavelet_dwt2_cA(matlab_mat *x, void *wn) {
    Mat2 cA, cH, cV, cD; wv_dwt2_one(mat2_from(x), wv_filters(wv_sstr(wn)), cA, cH, cV, cD);
    return mat2_to(cA);
}
matlab_mat *matlab_wavelet_dwt2_cH(matlab_mat *x, void *wn) {
    Mat2 cA, cH, cV, cD; wv_dwt2_one(mat2_from(x), wv_filters(wv_sstr(wn)), cA, cH, cV, cD);
    return mat2_to(cH);
}
matlab_mat *matlab_wavelet_dwt2_cV(matlab_mat *x, void *wn) {
    Mat2 cA, cH, cV, cD; wv_dwt2_one(mat2_from(x), wv_filters(wv_sstr(wn)), cA, cH, cV, cD);
    return mat2_to(cV);
}
matlab_mat *matlab_wavelet_dwt2_cD(matlab_mat *x, void *wn) {
    Mat2 cA, cH, cV, cD; wv_dwt2_one(mat2_from(x), wv_filters(wv_sstr(wn)), cA, cH, cV, cD);
    return mat2_to(cD);
}
matlab_mat *matlab_wavelet_idwt2(matlab_mat *cA, matlab_mat *cH, matlab_mat *cV,
                                 matlab_mat *cD, void *wn) {
    return mat2_to(wv_idwt2_one(mat2_from(cA), mat2_from(cH), mat2_from(cV),
                                mat2_from(cD), wv_filters(wv_sstr(wn))));
}

/* ----- wavedec2 / waverec2 (multilevel 2-D) --------------------------------
 * C = [cA_n(:); cH_n(:); cV_n(:); cD_n(:); ... cH_1; cV_1; cD_1] as a row.
 * S = [(rn,cn); (rn,cn); ...; (r1,c1); (R,C)] size bookkeeping. */
matlab_mat *matlab_wavelet_wavedec2_C(matlab_mat *x, double lev, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    int64_t n = static_cast<int64_t>(lev); if (n < 1) n = 1;
    Mat2 A = mat2_from(x);
    std::vector<double> C;
    std::vector<Mat2> details;   /* fine→coarse triples flattened later */
    std::vector<std::array<Mat2,3>> dets;
    for (int64_t l = 0; l < n; ++l) {
        Mat2 cA, cH, cV, cD; wv_dwt2_one(A, f, cA, cH, cV, cD);
        dets.push_back({cH, cV, cD});
        A = cA;
    }
    C.insert(C.end(), A.d.begin(), A.d.end());      /* coarsest approx */
    for (auto it = dets.rbegin(); it != dets.rend(); ++it) {
        C.insert(C.end(), (*it)[0].d.begin(), (*it)[0].d.end());
        C.insert(C.end(), (*it)[1].d.begin(), (*it)[1].d.end());
        C.insert(C.end(), (*it)[2].d.begin(), (*it)[2].d.end());
    }
    return wv_row(C);
}
matlab_mat *matlab_wavelet_wavedec2_S(matlab_mat *x, double lev, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    int64_t n = static_cast<int64_t>(lev); if (n < 1) n = 1;
    Mat2 A = mat2_from(x);
    std::vector<std::array<int64_t,2>> sizes;   /* coarse→fine */
    int64_t R0 = A.r, C0 = A.c;
    std::vector<std::array<int64_t,2>> detSizes;
    for (int64_t l = 0; l < n; ++l) {
        Mat2 cA, cH, cV, cD; wv_dwt2_one(A, f, cA, cH, cV, cD);
        detSizes.push_back({cH.r, cH.c});
        A = cA;
    }
    /* S rows: approx size, then detail sizes coarse→fine, then original. */
    std::vector<double> S;
    S.push_back(static_cast<double>(A.r));   /* will reshape to 2 cols below */
    /* build as (n+2) × 2 matrix row-major */
    std::vector<std::array<int64_t,2>> rows;
    rows.push_back({A.r, A.c});
    for (auto it = detSizes.rbegin(); it != detSizes.rend(); ++it) rows.push_back(*it);
    rows.push_back({R0, C0});
    matlab_mat *Sm = mat_alloc(static_cast<int64_t>(rows.size()), 2);
    for (size_t i = 0; i < rows.size(); ++i) {
        Sm->data[i * 2 + 0] = static_cast<double>(rows[i][0]);
        Sm->data[i * 2 + 1] = static_cast<double>(rows[i][1]);
    }
    return Sm;
}
matlab_mat *matlab_wavelet_waverec2(matlab_mat *C, matlab_mat *S, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    if (!C || !S || S->rows < 2) return mat_alloc(0, 0);
    int64_t nrows = S->rows;          /* (n+2) */
    int64_t n = nrows - 2;
    auto sr = [&](int64_t i) { return static_cast<int64_t>(S->data[i * 2 + 0]); };
    auto sc = [&](int64_t i) { return static_cast<int64_t>(S->data[i * 2 + 1]); };
    int64_t pos = 0;
    int64_t ra = sr(0), ca = sc(0);
    Mat2 A{std::vector<double>(C->data + pos, C->data + pos + ra * ca), ra, ca};
    pos += ra * ca;
    for (int64_t l = 0; l < n; ++l) {
        int64_t rd = sr(1 + l), cd = sc(1 + l);
        int64_t cnt = rd * cd;
        Mat2 cH{std::vector<double>(C->data + pos, C->data + pos + cnt), rd, cd}; pos += cnt;
        Mat2 cV{std::vector<double>(C->data + pos, C->data + pos + cnt), rd, cd}; pos += cnt;
        Mat2 cD{std::vector<double>(C->data + pos, C->data + pos + cnt), rd, cd}; pos += cnt;
        /* approx may differ in size from detail by ±1 — crop/pad. */
        if (A.r != rd || A.c != cd) {
            Mat2 A2{std::vector<double>(rd * cd, 0.0), rd, cd};
            for (int64_t i = 0; i < std::min(A.r, rd); ++i)
                for (int64_t k = 0; k < std::min(A.c, cd); ++k)
                    A2.d[i * cd + k] = A.d[i * A.c + k];
            A = A2;
        }
        A = wv_idwt2_one(A, cH, cV, cD, f);
    }
    int64_t R = sr(nrows - 1), Cc = sc(nrows - 1);
    if (A.r != R || A.c != Cc) {
        Mat2 out{std::vector<double>(R * Cc, 0.0), R, Cc};
        for (int64_t i = 0; i < std::min(A.r, R); ++i)
            for (int64_t k = 0; k < std::min(A.c, Cc); ++k)
                out.d[i * Cc + k] = A.d[i * A.c + k];
        return mat2_to(out);
    }
    return mat2_to(A);
}

/* ----- wcodemat (rescale matrix for display) -------------------------------*/
matlab_mat *matlab_wavelet_wcodemat2(matlab_mat *x, double nbcol) {
    std::vector<double> v = wv_vec(x);
    double lo = 1e300, hi = -1e300;
    for (double a : v) { lo = std::min(lo, a); hi = std::max(hi, a); }
    double rng = (hi - lo); if (rng < 1e-30) rng = 1;
    if (nbcol <= 0) nbcol = 16;
    matlab_mat *r = mat_alloc(x->rows, x->cols);
    for (size_t i = 0; i < v.size(); ++i)
        r->data[i] = 1.0 + floor((nbcol - 1.0) * (v[i] - lo) / rng);
    return r;
}
matlab_mat *matlab_wavelet_wcodemat1(matlab_mat *x) { return matlab_wavelet_wcodemat2(x, 16); }

}  /* extern "C" */

/* ===========================================================================
 * Tier-5 — wavelet packets (matrix lane: terminal-node coefficient matrix in
 * natural / Paley order, 2^n rows of length N/2^n; PR-exact with wprec).
 * ==========================================================================*/
extern "C" {

/* wpdec: full binary-tree decomposition to depth n. */
matlab_mat *matlab_wavelet_wpdec(matlab_mat *x, double depth, void *wn) {
    WvFilters f = wv_filters(wv_sstr(wn));
    int64_t n = static_cast<int64_t>(depth); if (n < 1) n = 1;
    std::vector<std::vector<double>> nodes;
    nodes.push_back(wv_vec(x));
    for (int64_t l = 0; l < n; ++l) {
        std::vector<std::vector<double>> nxt;
        for (auto &nd : nodes) {
            std::vector<double> ev = wv_to_even(nd), a, d;
            wv_dwt1(ev, f, a, d);
            nxt.push_back(a); nxt.push_back(d);
        }
        nodes = nxt;
    }
    int64_t rows = static_cast<int64_t>(nodes.size());
    int64_t cols = rows ? static_cast<int64_t>(nodes[0].size()) : 0;
    matlab_mat *out = mat_alloc(rows, cols);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t k = 0; k < cols && k < static_cast<int64_t>(nodes[static_cast<size_t>(i)].size()); ++k)
            out->data[i * cols + k] = nodes[static_cast<size_t>(i)][static_cast<size_t>(k)];
    return out;
}

/* wprec: reconstruct from a terminal-node matrix (rows = 2^n nodes). */
matlab_mat *matlab_wavelet_wprec(matlab_mat *T, void *wn) {
    if (!T || T->rows < 1) return mat_alloc(0, 0);
    WvFilters f = wv_filters(wv_sstr(wn));
    int64_t rows = T->rows, cols = T->cols;
    std::vector<std::vector<double>> nodes(static_cast<size_t>(rows));
    for (int64_t i = 0; i < rows; ++i)
        nodes[static_cast<size_t>(i)].assign(T->data + i * cols, T->data + (i + 1) * cols);
    while (nodes.size() > 1) {
        std::vector<std::vector<double>> parents;
        for (size_t k = 0; k + 1 < nodes.size(); k += 2)
            parents.push_back(wv_idwt1(nodes[k], nodes[k + 1], f));
        nodes = parents;
    }
    return wv_row(nodes[0]);
}

/* wpcoef: read the coefficients at a terminal node (0-based natural order). */
matlab_mat *matlab_wavelet_wpcoef(matlab_mat *T, double node) {
    if (!T || T->rows < 1) return mat_alloc(0, 0);
    int64_t r = static_cast<int64_t>(node);
    if (r < 0 || r >= T->rows) r = 0;
    std::vector<double> row(T->data + r * T->cols, T->data + (r + 1) * T->cols);
    return wv_row(row);
}

/* besttree: full tree is a valid basis in the matrix lane (true entropy
 * pruning to a sub-tree object is a documented carve-down) — identity. */
matlab_mat *matlab_wavelet_besttree(matlab_mat *T) {
    matlab_mat *r = mat_alloc(T ? T->rows : 0, T ? T->cols : 0);
    if (r && T && T->data) memcpy(r->data, T->data, static_cast<size_t>(T->rows * T->cols) * sizeof(double));
    return r;
}

/* wenergy(packet): per-node energy as a percentage of the total. */
matlab_mat *matlab_wavelet_wenergy_wp(matlab_mat *T) {
    if (!T || T->rows < 1) return mat_alloc(0, 0);
    int64_t rows = T->rows, cols = T->cols;
    std::vector<double> e(static_cast<size_t>(rows), 0.0);
    double total = 0;
    for (int64_t i = 0; i < rows; ++i) {
        double s = 0; for (int64_t k = 0; k < cols; ++k) { double a = T->data[i * cols + k]; s += a * a; }
        e[static_cast<size_t>(i)] = s; total += s;
    }
    if (total < 1e-300) total = 1;
    for (double &v : e) v = 100.0 * v / total;
    return wv_row(e);
}

}  /* extern "C" */

/* ===========================================================================
 * Tier-6 — special topics (EWT / VMD / EMD) + matching pursuit + scattering
 * ==========================================================================*/
namespace {

/* natural cubic spline through (xk, yk), evaluated at xq (all sorted xk). */
std::vector<double> wv_spline(const std::vector<double> &xk, const std::vector<double> &yk,
                              const std::vector<double> &xq) {
    int64_t n = static_cast<int64_t>(xk.size());
    std::vector<double> yq(xq.size(), 0.0);
    if (n < 2) { for (auto &v : yq) v = n == 1 ? yk[0] : 0.0; return yq; }
    if (n == 2) {
        for (size_t i = 0; i < xq.size(); ++i) {
            double t = (xq[i] - xk[0]) / (xk[1] - xk[0]);
            yq[i] = yk[0] + t * (yk[1] - yk[0]);
        }
        return yq;
    }
    std::vector<double> h(n - 1), alpha(n, 0.0), l(n), mu(n), z(n), c(n, 0.0), b(n - 1), d(n - 1);
    for (int64_t i = 0; i < n - 1; ++i) h[static_cast<size_t>(i)] = xk[static_cast<size_t>(i + 1)] - xk[static_cast<size_t>(i)];
    for (int64_t i = 1; i < n - 1; ++i)
        alpha[static_cast<size_t>(i)] = 3.0 / h[static_cast<size_t>(i)] * (yk[static_cast<size_t>(i + 1)] - yk[static_cast<size_t>(i)])
                       - 3.0 / h[static_cast<size_t>(i - 1)] * (yk[static_cast<size_t>(i)] - yk[static_cast<size_t>(i - 1)]);
    l[0] = 1; mu[0] = 0; z[0] = 0;
    for (int64_t i = 1; i < n - 1; ++i) {
        l[static_cast<size_t>(i)] = 2.0 * (xk[static_cast<size_t>(i + 1)] - xk[static_cast<size_t>(i - 1)]) - h[static_cast<size_t>(i - 1)] * mu[static_cast<size_t>(i - 1)];
        mu[static_cast<size_t>(i)] = h[static_cast<size_t>(i)] / l[static_cast<size_t>(i)];
        z[static_cast<size_t>(i)] = (alpha[static_cast<size_t>(i)] - h[static_cast<size_t>(i - 1)] * z[static_cast<size_t>(i - 1)]) / l[static_cast<size_t>(i)];
    }
    l[n - 1] = 1; z[n - 1] = 0; c[n - 1] = 0;
    for (int64_t j = n - 2; j >= 0; --j) {
        c[static_cast<size_t>(j)] = z[static_cast<size_t>(j)] - mu[static_cast<size_t>(j)] * c[static_cast<size_t>(j + 1)];
        b[static_cast<size_t>(j)] = (yk[static_cast<size_t>(j + 1)] - yk[static_cast<size_t>(j)]) / h[static_cast<size_t>(j)]
                   - h[static_cast<size_t>(j)] * (c[static_cast<size_t>(j + 1)] + 2.0 * c[static_cast<size_t>(j)]) / 3.0;
        d[static_cast<size_t>(j)] = (c[static_cast<size_t>(j + 1)] - c[static_cast<size_t>(j)]) / (3.0 * h[static_cast<size_t>(j)]);
    }
    for (size_t q = 0; q < xq.size(); ++q) {
        double x = xq[q];
        int64_t i = n - 2;
        for (int64_t s = 0; s < n - 1; ++s) if (x <= xk[static_cast<size_t>(s + 1)]) { i = s; break; }
        double dx = x - xk[static_cast<size_t>(i)];
        yq[q] = yk[static_cast<size_t>(i)] + b[static_cast<size_t>(i)] * dx + c[static_cast<size_t>(i)] * dx * dx + d[static_cast<size_t>(i)] * dx * dx * dx;
    }
    return yq;
}

/* one EMD sift: subtract the mean of cubic-spline envelopes. */
bool wv_extrema_envelope(const std::vector<double> &h, std::vector<double> &up,
                         std::vector<double> &lo) {
    int64_t N = static_cast<int64_t>(h.size());
    std::vector<double> xmax, ymax, xmin, ymin;
    for (int64_t i = 1; i < N - 1; ++i) {
        if (h[static_cast<size_t>(i)] > h[static_cast<size_t>(i - 1)] && h[static_cast<size_t>(i)] >= h[static_cast<size_t>(i + 1)]) { xmax.push_back(static_cast<double>(i)); ymax.push_back(h[static_cast<size_t>(i)]); }
        if (h[static_cast<size_t>(i)] < h[static_cast<size_t>(i - 1)] && h[static_cast<size_t>(i)] <= h[static_cast<size_t>(i + 1)]) { xmin.push_back(static_cast<double>(i)); ymin.push_back(h[static_cast<size_t>(i)]); }
    }
    if (xmax.size() < 2 || xmin.size() < 2) return false;
    /* anchor endpoints. */
    xmax.insert(xmax.begin(), 0.0); ymax.insert(ymax.begin(), h[0]);
    xmax.push_back(static_cast<double>(N - 1)); ymax.push_back(h[static_cast<size_t>(N - 1)]);
    xmin.insert(xmin.begin(), 0.0); ymin.insert(ymin.begin(), h[0]);
    xmin.push_back(static_cast<double>(N - 1)); ymin.push_back(h[static_cast<size_t>(N - 1)]);
    std::vector<double> xq(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) xq[static_cast<size_t>(i)] = static_cast<double>(i);
    up = wv_spline(xmax, ymax, xq);
    lo = wv_spline(xmin, ymin, xq);
    return true;
}

}  /* anonymous namespace */

extern "C" {

/* ----- emd (empirical mode decomposition) ---------------------------------*/
matlab_mat *matlab_wavelet_emd(matlab_mat *x, double maxmodes) {
    std::vector<double> r = wv_vec(x);
    int64_t N = static_cast<int64_t>(r.size());
    int64_t K = static_cast<int64_t>(maxmodes); if (K < 1) K = 6;
    std::vector<std::vector<double>> imfs;
    for (int64_t m = 0; m < K; ++m) {
        std::vector<double> h = r;
        bool ok = true;
        for (int it = 0; it < 30; ++it) {
            std::vector<double> up, lo;
            if (!wv_extrema_envelope(h, up, lo)) { ok = false; break; }
            double sd = 0, en = 0;
            std::vector<double> hn(static_cast<size_t>(N));
            for (int64_t i = 0; i < N; ++i) {
                double mean = 0.5 * (up[static_cast<size_t>(i)] + lo[static_cast<size_t>(i)]);
                hn[static_cast<size_t>(i)] = h[static_cast<size_t>(i)] - mean;
                sd += (h[static_cast<size_t>(i)] - hn[static_cast<size_t>(i)]) * (h[static_cast<size_t>(i)] - hn[static_cast<size_t>(i)]);
                en += h[static_cast<size_t>(i)] * h[static_cast<size_t>(i)];
            }
            h = hn;
            if (en > 1e-30 && sd / en < 1e-4) break;
        }
        if (!ok) break;
        imfs.push_back(h);
        for (int64_t i = 0; i < N; ++i) r[static_cast<size_t>(i)] -= h[static_cast<size_t>(i)];
    }
    imfs.push_back(r);   /* residual */
    int64_t rows = static_cast<int64_t>(imfs.size());
    matlab_mat *out = mat_alloc(rows, N);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t k = 0; k < N; ++k) out->data[i * N + k] = imfs[static_cast<size_t>(i)][static_cast<size_t>(k)];
    return out;
}

/* ----- vmd (variational mode decomposition, frequency-domain ADMM) --------*/
matlab_mat *matlab_wavelet_vmd(matlab_mat *x, double Kd) {
    std::vector<double> sig = wv_vec(x);
    int64_t N = static_cast<int64_t>(sig.size());
    int64_t K = static_cast<int64_t>(Kd); if (K < 1) K = 3;
    if (N < 4) return mat_alloc(K, N);
    double alpha = 2000.0, tau = 0.0;   /* moderate bandwidth, no noise-slack */
    /* spectrum of the (one-sided) signal. */
    std::vector<double> fr, fi; wv_fft(sig, fr, fi);
    int64_t half = N / 2 + 1;
    std::vector<double> freq(static_cast<size_t>(half));
    for (int64_t k = 0; k < half; ++k) freq[static_cast<size_t>(k)] = static_cast<double>(k) / static_cast<double>(N);
    std::vector<std::vector<double>> ur(static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(half), 0.0));
    std::vector<std::vector<double>> ui(static_cast<size_t>(K), std::vector<double>(static_cast<size_t>(half), 0.0));
    std::vector<double> omega(static_cast<size_t>(K));
    for (int64_t k = 0; k < K; ++k) omega[static_cast<size_t>(k)] = 0.5 * static_cast<double>(k + 1) / static_cast<double>(K);
    std::vector<double> lr(static_cast<size_t>(half), 0.0), li(static_cast<size_t>(half), 0.0);
    for (int iter = 0; iter < 200; ++iter) {
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t f = 0; f < half; ++f) {
                double sumr = fr[static_cast<size_t>(f)], sumi = fi[static_cast<size_t>(f)];
                for (int64_t i = 0; i < K; ++i) if (i != k) { sumr -= ur[static_cast<size_t>(i)][static_cast<size_t>(f)]; sumi -= ui[static_cast<size_t>(i)][static_cast<size_t>(f)]; }
                sumr += 0.5 * lr[static_cast<size_t>(f)]; sumi += 0.5 * li[static_cast<size_t>(f)];
                double df = freq[static_cast<size_t>(f)] - omega[static_cast<size_t>(k)];
                double denom = 1.0 + 2.0 * alpha * df * df;
                ur[static_cast<size_t>(k)][static_cast<size_t>(f)] = sumr / denom;
                ui[static_cast<size_t>(k)][static_cast<size_t>(f)] = sumi / denom;
            }
            double num = 0, den = 0;
            for (int64_t f = 0; f < half; ++f) {
                double p = ur[static_cast<size_t>(k)][static_cast<size_t>(f)] * ur[static_cast<size_t>(k)][static_cast<size_t>(f)]
                         + ui[static_cast<size_t>(k)][static_cast<size_t>(f)] * ui[static_cast<size_t>(k)][static_cast<size_t>(f)];
                num += freq[static_cast<size_t>(f)] * p; den += p;
            }
            if (den > 1e-30) omega[static_cast<size_t>(k)] = num / den;
        }
        for (int64_t f = 0; f < half; ++f) {
            double sr = fr[static_cast<size_t>(f)], si = fi[static_cast<size_t>(f)];
            for (int64_t k = 0; k < K; ++k) { sr -= ur[static_cast<size_t>(k)][static_cast<size_t>(f)]; si -= ui[static_cast<size_t>(k)][static_cast<size_t>(f)]; }
            lr[static_cast<size_t>(f)] += tau * sr; li[static_cast<size_t>(f)] += tau * si;
        }
    }
    /* reconstruct each mode by Hermitian-symmetric ifft. */
    matlab_mat *out = mat_alloc(K, N);
    for (int64_t k = 0; k < K; ++k) {
        matlab_mat_c *in = mat_c_alloc(1, N);
        for (int64_t f = 0; f < N; ++f) {
            if (f < half) { in->re[f] = ur[static_cast<size_t>(k)][static_cast<size_t>(f)]; in->im[f] = ui[static_cast<size_t>(k)][static_cast<size_t>(f)]; }
            else { int64_t m = N - f; in->re[f] = ur[static_cast<size_t>(k)][static_cast<size_t>(m)]; in->im[f] = -ui[static_cast<size_t>(k)][static_cast<size_t>(m)]; }
        }
        matlab_mat_c *rec = matlab_ifft_c(in);
        if (rec && rec->re) for (int64_t i = 0; i < N; ++i) out->data[k * N + i] = rec->re[i];
    }
    return out;
}

/* ----- ewt (empirical wavelet transform) ----------------------------------*/
matlab_mat *matlab_wavelet_ewt(matlab_mat *x, double Nb) {
    std::vector<double> sig = wv_vec(x);
    int64_t N = static_cast<int64_t>(sig.size());
    int64_t nb = static_cast<int64_t>(Nb); if (nb < 1) nb = 3;
    if (N < 4) return mat_alloc(nb, N);
    std::vector<double> fr, fi; wv_fft(sig, fr, fi);
    int64_t half = N / 2;
    std::vector<double> mag(static_cast<size_t>(half));
    for (int64_t k = 0; k < half; ++k) mag[static_cast<size_t>(k)] = sqrt(fr[static_cast<size_t>(k)] * fr[static_cast<size_t>(k)] + fi[static_cast<size_t>(k)] * fi[static_cast<size_t>(k)]);
    /* find the (nb) largest spectral maxima, boundaries = midpoints between
     * consecutive sorted peak positions. */
    std::vector<std::pair<double,int64_t>> peaks;
    for (int64_t k = 1; k < half - 1; ++k)
        if (mag[static_cast<size_t>(k)] > mag[static_cast<size_t>(k - 1)] && mag[static_cast<size_t>(k)] >= mag[static_cast<size_t>(k + 1)])
            peaks.push_back({mag[static_cast<size_t>(k)], k});
    std::sort(peaks.rbegin(), peaks.rend());
    std::vector<int64_t> pk;
    for (int64_t i = 0; i < nb && i < static_cast<int64_t>(peaks.size()); ++i) pk.push_back(peaks[static_cast<size_t>(i)].second);
    std::sort(pk.begin(), pk.end());
    std::vector<int64_t> bnd; bnd.push_back(0);
    for (size_t i = 0; i + 1 < pk.size(); ++i) bnd.push_back((pk[i] + pk[i + 1]) / 2);
    bnd.push_back(half);
    int64_t actualBands = static_cast<int64_t>(bnd.size()) - 1; if (actualBands < 1) actualBands = 1;
    matlab_mat *out = mat_alloc(actualBands, N);
    for (int64_t b = 0; b < actualBands; ++b) {
        int64_t lo = bnd[static_cast<size_t>(b)], hi = bnd[static_cast<size_t>(b + 1)];
        matlab_mat_c *in = mat_c_alloc(1, N);
        for (int64_t f = 0; f < N; ++f) {
            bool keep = (f >= lo && f < hi) || (f > 0 && (N - f) >= lo && (N - f) < hi);
            in->re[f] = keep ? fr[static_cast<size_t>(f)] : 0.0;
            in->im[f] = keep ? fi[static_cast<size_t>(f)] : 0.0;
        }
        matlab_mat_c *rec = matlab_ifft_c(in);
        if (rec && rec->re) for (int64_t i = 0; i < N; ++i) out->data[b * N + i] = rec->re[i];
    }
    return out;
}

/* ----- matchingPursuit (orthogonal MP over a dictionary) ------------------
 * D: M×P (atoms in columns), y: M-vector, K iterations → P coefficient vector. */
matlab_mat *matlab_wavelet_omp(matlab_mat *D, matlab_mat *y, double Kd) {
    if (!D || !y) return mat_alloc(0, 0);
    int64_t M = D->rows, P = D->cols;
    int64_t K = static_cast<int64_t>(Kd); if (K < 1) K = 1; if (K > P) K = P;
    std::vector<double> r = wv_vec(y); r.resize(static_cast<size_t>(M), 0.0);
    std::vector<double> coeff(static_cast<size_t>(P), 0.0);
    std::vector<int64_t> sel;
    /* column 2-norms for normalised projection. */
    std::vector<double> nrm(static_cast<size_t>(P), 0.0);
    for (int64_t j = 0; j < P; ++j) { double s = 0; for (int64_t i = 0; i < M; ++i) s += D->data[i * P + j] * D->data[i * P + j]; nrm[static_cast<size_t>(j)] = sqrt(s) + 1e-30; }
    for (int64_t k = 0; k < K; ++k) {
        int64_t best = -1; double bestv = -1;
        for (int64_t j = 0; j < P; ++j) {
            bool used = false; for (int64_t s : sel) if (s == j) { used = true; break; }
            if (used) continue;
            double p = 0; for (int64_t i = 0; i < M; ++i) p += D->data[i * P + j] * r[static_cast<size_t>(i)];
            double v = fabs(p) / nrm[static_cast<size_t>(j)];
            if (v > bestv) { bestv = v; best = j; }
        }
        if (best < 0) break;
        sel.push_back(best);
        /* least squares over selected columns via normal equations. */
        int64_t s = static_cast<int64_t>(sel.size());
        std::vector<double> G(static_cast<size_t>(s * s), 0.0), b(static_cast<size_t>(s), 0.0);
        for (int64_t a = 0; a < s; ++a) {
            for (int64_t c = 0; c < s; ++c) { double g = 0; for (int64_t i = 0; i < M; ++i) g += D->data[i * P + sel[static_cast<size_t>(a)]] * D->data[i * P + sel[static_cast<size_t>(c)]]; G[static_cast<size_t>(a * s + c)] = g; }
            double bb = 0; for (int64_t i = 0; i < M; ++i) bb += D->data[i * P + sel[static_cast<size_t>(a)]] * (wv_vec(y))[static_cast<size_t>(i)]; b[static_cast<size_t>(a)] = bb;
        }
        /* Gaussian elimination. */
        for (int64_t c = 0; c < s; ++c) {
            int64_t piv = c; for (int64_t rr = c + 1; rr < s; ++rr) if (fabs(G[static_cast<size_t>(rr * s + c)]) > fabs(G[static_cast<size_t>(piv * s + c)])) piv = rr;
            for (int64_t cc = 0; cc < s; ++cc) std::swap(G[static_cast<size_t>(c * s + cc)], G[static_cast<size_t>(piv * s + cc)]);
            std::swap(b[static_cast<size_t>(c)], b[static_cast<size_t>(piv)]);
            double d = G[static_cast<size_t>(c * s + c)]; if (fabs(d) < 1e-30) d = 1e-30;
            for (int64_t cc = 0; cc < s; ++cc) G[static_cast<size_t>(c * s + cc)] /= d; b[static_cast<size_t>(c)] /= d;
            for (int64_t rr = 0; rr < s; ++rr) if (rr != c) { double fct = G[static_cast<size_t>(rr * s + c)]; for (int64_t cc = 0; cc < s; ++cc) G[static_cast<size_t>(rr * s + cc)] -= fct * G[static_cast<size_t>(c * s + cc)]; b[static_cast<size_t>(rr)] -= fct * b[static_cast<size_t>(c)]; }
        }
        for (double &v : coeff) v = 0;
        for (int64_t a = 0; a < s; ++a) coeff[static_cast<size_t>(sel[static_cast<size_t>(a)])] = b[static_cast<size_t>(a)];
        std::vector<double> yy = wv_vec(y);
        for (int64_t i = 0; i < M; ++i) { double pr = 0; for (int64_t a = 0; a < s; ++a) pr += D->data[i * P + sel[static_cast<size_t>(a)]] * b[static_cast<size_t>(a)]; r[static_cast<size_t>(i)] = yy[static_cast<size_t>(i)] - pr; }
    }
    return wv_col(coeff);
}

/* ----- waveletScattering (S0 + S1 invariant features) ----------------------
 * Time-averaged |CWT| per scale + the signal mean — a simplified scattering
 * feature vector (the S2 second-order layer is a documented carve-down). */
matlab_mat *matlab_wavelet_scatter(matlab_mat *x) {
    std::vector<double> sig = wv_vec(x);
    int64_t N = static_cast<int64_t>(sig.size());
    if (N < 4) return mat_alloc(0, 0);
    matlab_mat_c *W = wv_cwt_coeffs(sig);
    if (!W) return mat_alloc(0, 0);
    int64_t ns = W->rows, n = W->cols;
    std::vector<double> feat;
    double s0 = 0; for (double a : sig) s0 += a; feat.push_back(s0 / static_cast<double>(N));
    for (int64_t j = 0; j < ns; ++j) {
        double s = 0;
        for (int64_t k = 0; k < n; ++k)
            s += sqrt(W->re[j * n + k] * W->re[j * n + k] + W->im[j * n + k] * W->im[j * n + k]);
        feat.push_back(s / static_cast<double>(n));
    }
    return wv_col(feat);
}

}  /* extern "C" */
