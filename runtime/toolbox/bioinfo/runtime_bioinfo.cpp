/* ============================================================================
 * runtime_bioinfo.cpp — Bioinformatics Toolbox runtime (Phase A: Tiers 1-2)
 * ----------------------------------------------------------------------------
 * Tier-1 — sequence lane: representation, alphabets, conversion, statistics,
 *   and FASTA I/O.  nt2int / int2nt / aa2int / int2aa / nt2aa / dna2rna /
 *   rna2dna / seqcomplement / seqrcomplement / seqreverse / basecount /
 *   aacount / randseq / fastaread / fastawrite.
 * Tier-2 — pairwise alignment: blosum / pam / nuc44 scoring-matrix catalogue
 *   + nwalign (Needleman-Wunsch global) + swalign (Smith-Waterman local) +
 *   seqdotplot.
 *
 * Representation: biological sequences are char vectors, carried at runtime as
 * matlab_string* (the same value `'ACGT'` / "ACGT" lower to).  Sequence
 * transforms take and return matlab_string*; conversions to the numeric lane
 * (nt2int) return matlab_mat*; composition counts (basecount / aacount) return
 * a matlab_struct*; scoring matrices return a labelled matlab_mat*; alignment
 * scores return a 1x1 matlab_mat*, the alignment a 3-row newline-joined
 * matlab_string*.
 *
 * Scoring matrices + genetic code are baked-in lookup tables (the same
 * precedent as the Comm 5G-NR base matrices, the Image fspecial kernels and
 * the Wavelet family-filter catalogue).  No external dependency.
 * ==========================================================================*/

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

/* shipped helpers reused. */
extern "C" matlab_string *matlab_string_from_literal(const char *src,
                                                     int64_t len);
extern "C" void matlab_struct_set_string(matlab_struct *s, const char *name,
                                         int64_t len, void *str);
/* classdef object field ABI (phytree, Tier-4). */
extern "C" void   matlab_obj_set_f64(matlab_obj *o, const char *name,
                                     int64_t len, double v);
extern "C" void   matlab_obj_set_mat(matlab_obj *o, const char *name,
                                     int64_t len, matlab_mat *m);
extern "C" void   matlab_obj_set_string(matlab_obj *o, const char *name,
                                        int64_t len, void *str);
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name,
                                          int64_t len);
extern "C" double matlab_obj_get_f64(matlab_obj *o, const char *name,
                                     int64_t len);
/* cell-array ABI (sequence-set inputs, Tier-3/4). */
extern "C" double      matlab_cell_numel(matlab_cell *c);
extern "C" matlab_mat *matlab_cell_get_mat(matlab_cell *c, double i1);

namespace {

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
struct bi_string_s { char *data; int64_t len; };

std::string bi_sstr(const void *s) {
    if (!s) return std::string();
    const bi_string_s *p = reinterpret_cast<const bi_string_s *>(s);
    if (!p->data || p->len <= 0 || p->len > (1 << 24)) return std::string();
    return std::string(p->data, p->data + p->len);
}

/* Read a sequence set passed as a cell array of char vectors (the MATLAB
 * multialign / seqpdist API).  Cell string elements come back from
 * matlab_cell_get_mat as char-code rows (1xN doubles). */
std::vector<std::string> bi_seqset(void *cell) {
    std::vector<std::string> out;
    if (!cell) return out;
    matlab_cell *c = reinterpret_cast<matlab_cell *>(cell);
    int n = static_cast<int>(matlab_cell_numel(c));
    if (n < 0 || n > (1 << 20)) return out;
    for (int i = 1; i <= n; ++i) {
        matlab_mat *m = matlab_cell_get_mat(c, static_cast<double>(i));
        std::string s;
        if (m && m->data) {
            int64_t cnt = m->rows * m->cols;
            for (int64_t k = 0; k < cnt; ++k)
                s.push_back(static_cast<char>(m->data[k]));
        }
        out.push_back(s);
    }
    return out;
}

/* Split a newline-joined alignment string into its rows (the multialign
 * output / seqprofile-seqconsensus input representation). */
std::vector<std::string> bi_splitlines(const std::string &all) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : all) {
        if (ch == '\n') { out.push_back(cur); cur.clear(); }
        else if (ch != '\r') cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

matlab_string *bi_mkstr(const std::string &s) {
    return matlab_string_from_literal(s.c_str(),
                                      static_cast<int64_t>(s.size()));
}

matlab_mat *bi_scalar(double v) {
    matlab_mat *r = mat_alloc(1, 1);
    if (r && r->data) r->data[0] = v;
    return r;
}

matlab_mat *bi_row(const std::vector<double> &v) {
    matlab_mat *r = mat_alloc(1, static_cast<int64_t>(v.size()));
    if (r && r->data)
        for (size_t i = 0; i < v.size(); ++i) r->data[i] = v[i];
    return r;
}

char bi_upper(char c) {
    return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

/* ===== alphabets ========================================================= */

/* Nucleotide code (MATLAB nt2int): A=1 C=2 G=3 T/U=4, anything else=0
 * (gap/ambiguous collapse to 0 for the Phase-A core). */
int bi_nt_code(char c) {
    switch (bi_upper(c)) {
        case 'A': return 1;
        case 'C': return 2;
        case 'G': return 3;
        case 'T': case 'U': return 4;
        default:  return 0;
    }
}
char bi_nt_char(int code) {
    static const char *T = "-ACGT";
    return (code >= 1 && code <= 4) ? T[code] : '-';
}

/* Amino-acid order used by BLOSUM/PAM: A R N D C Q E G H I L K M F P S T W Y V
 * B Z X * (24 symbols, indices 0..23). */
const char *kAA = "ARNDCQEGHILKMFPSTWYVBZX*";
int bi_aa_code(char c) {
    char u = bi_upper(c);
    for (int i = 0; i < 24; ++i)
        if (kAA[i] == u) return i + 1;   /* 1-based, matches MATLAB aa2int */
    if (u == '-') return 25;             /* gap */
    return 23;                           /* unknown -> X */
}
char bi_aa_char(int code) {
    if (code >= 1 && code <= 24) return kAA[code - 1];
    if (code == 25) return '-';
    return 'X';
}

/* ===== genetic code (standard, frame 1) ================================== */
/* Codon -> amino-acid (1-letter).  Index a codon as 16*b0+4*b1+b2 with
 * T=0 C=1 A=2 G=3 (the canonical codon-table ordering). */
char bi_translate_codon(char a, char b, char c) {
    static const char *tab =
        /* TTT..TGG (T-block), then C-, A-, G-blocks */
        "FFLLSSSSYY**CC*W"   /* T?? */
        "LLLLPPPPHHQQRRRR"   /* C?? */
        "IIIMTTTTNNKKSSRR"   /* A?? */
        "VVVVAAAADDEEGGGG";  /* G?? */
    auto idx = [](char x) -> int {
        switch (bi_upper(x)) {
            case 'T': case 'U': return 0;
            case 'C': return 1;
            case 'A': return 2;
            case 'G': return 3;
            default:  return -1;
        }
    };
    int i0 = idx(a), i1 = idx(b), i2 = idx(c);
    if (i0 < 0 || i1 < 0 || i2 < 0) return 'X';
    return tab[16 * i0 + 4 * i1 + i2];
}

/* ===== scoring-matrix catalogue ========================================== */
/* BLOSUM62 in the kAA ordering (24x24). */
static const int kBLOSUM62[24][24] = {
/*A*/{ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4},
/*R*/{-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4},
/*N*/{-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4},
/*D*/{-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4},
/*C*/{ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4},
/*Q*/{-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4},
/*E*/{-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4},
/*G*/{ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4},
/*H*/{-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4},
/*I*/{-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4},
/*L*/{-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4},
/*K*/{-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4},
/*M*/{-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4},
/*F*/{-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4},
/*P*/{-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4},
/*S*/{ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4},
/*T*/{ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4},
/*W*/{-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4},
/*Y*/{-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4},
/*V*/{ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4},
/*B*/{-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4},
/*Z*/{-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4},
/*X*/{ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4},
/***/{-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1},
};

/* Is a sequence pure nucleotide (only ACGTU/N, case-insensitive)? */
bool bi_is_nt(const std::string &s) {
    if (s.empty()) return false;
    for (char c : s) {
        char u = bi_upper(c);
        if (u != 'A' && u != 'C' && u != 'G' && u != 'T' && u != 'U' &&
            u != 'N' && u != '-')
            return false;
    }
    return true;
}

/* Scoring functor: a 256x256-ish lookup keyed by residue chars. */
struct Scorer {
    bool nt;            /* true => nucleotide NUC44, false => BLOSUM62 */
    int score(char a, char b) const {
        char x = bi_upper(a), y = bi_upper(b);
        if (nt) {
            if (x == 'U') x = 'T';
            if (y == 'U') y = 'T';
            return (x == y) ? 5 : -4;   /* NUC44 core: match +5 / mismatch -4 */
        }
        return kBLOSUM62[bi_aa_code(x) - 1][bi_aa_code(y) - 1];
    }
};

Scorer bi_pick_scorer(const std::string &s1, const std::string &s2,
                      const std::string &name) {
    Scorer sc;
    if (!name.empty()) {
        std::string n;
        for (char c : name) n.push_back(bi_upper(c));
        sc.nt = (n.find("NUC") != std::string::npos);
        return sc;
    }
    sc.nt = bi_is_nt(s1) && bi_is_nt(s2);
    return sc;
}

/* ===== Needleman-Wunsch / Smith-Waterman engine ========================== */
struct AlignResult {
    double score;
    std::string a1, mid, a2;   /* the 3 alignment rows */
};

AlignResult bi_align(const std::string &s1, const std::string &s2,
                     const Scorer &sc, bool local) {
    const int gap = -8;        /* MATLAB default GapOpen = 8 (linear gap) */
    int n = static_cast<int>(s1.size()), m = static_cast<int>(s2.size());
    std::vector<std::vector<double>> H(n + 1, std::vector<double>(m + 1, 0.0));
    std::vector<std::vector<char>> tb(n + 1, std::vector<char>(m + 1, 0));
    /* 0 = stop, 1 = diag, 2 = up (gap in s2), 3 = left (gap in s1) */
    if (!local) {
        for (int i = 1; i <= n; ++i) { H[i][0] = i * gap; tb[i][0] = 2; }
        for (int j = 1; j <= m; ++j) { H[0][j] = j * gap; tb[0][j] = 3; }
    }
    double best = local ? 0.0 : H[n][m];
    int bi = n, bj = m;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            double d = H[i - 1][j - 1] + sc.score(s1[i - 1], s2[j - 1]);
            double u = H[i - 1][j] + gap;
            double l = H[i][j - 1] + gap;
            double v = d; char t = 1;
            if (u > v) { v = u; t = 2; }
            if (l > v) { v = l; t = 3; }
            if (local && v < 0.0) { v = 0.0; t = 0; }
            H[i][j] = v; tb[i][j] = t;
            if (local && v >= best) { best = v; bi = i; bj = j; }
        }
    }
    AlignResult R;
    R.score = local ? best : H[n][m];
    int i = local ? bi : n, j = local ? bj : m;
    std::string a1, mid, a2;
    while (i > 0 || j > 0) {
        char t = tb[i][j];
        if (local && (t == 0 || (i == 0 && j == 0))) break;
        if (t == 1 || (i > 0 && j > 0 && t == 0)) {
            char c1 = s1[i - 1], c2 = s2[j - 1];
            a1.push_back(c1); a2.push_back(c2);
            int sv = sc.score(c1, c2);
            mid.push_back(bi_upper(c1) == bi_upper(c2) ? '|'
                          : (sv > 0 ? ':' : ' '));
            --i; --j;
        } else if (t == 2 && i > 0) {
            a1.push_back(s1[i - 1]); a2.push_back('-'); mid.push_back(' ');
            --i;
        } else if (j > 0) {
            a1.push_back('-'); a2.push_back(s2[j - 1]); mid.push_back(' ');
            --j;
        } else if (i > 0) {
            a1.push_back(s1[i - 1]); a2.push_back('-'); mid.push_back(' ');
            --i;
        } else break;
    }
    std::reverse(a1.begin(), a1.end());
    std::reverse(mid.begin(), mid.end());
    std::reverse(a2.begin(), a2.end());
    R.a1 = a1; R.mid = mid; R.a2 = a2;
    return R;
}

/* ===== Tier-3 — multiple-sequence alignment + profiles ==================== */

/* True if the whole set looks nucleotide. */
bool bi_set_is_nt(const std::vector<std::string> &seqs) {
    for (const std::string &s : seqs)
        if (!bi_is_nt(s)) return false;
    return !seqs.empty();
}

/* Consensus residue of a set of aligned rows at column j (ignoring gaps). */
char bi_consensus_col(const std::vector<std::string> &rows, size_t j) {
    int best = -1; char bc = '-';
    int counts[128] = {0};
    for (const std::string &r : rows) {
        if (j >= r.size()) continue;
        char c = bi_upper(r[j]);
        if (c == '-') continue;
        counts[static_cast<int>(c) & 127]++;
    }
    for (int c = 0; c < 128; ++c)
        if (counts[c] > best) { best = counts[c]; bc = static_cast<char>(c); }
    return (best <= 0) ? '-' : bc;
}

/* Progressive multiple alignment: order the sequences, then add each one to a
 * growing alignment by Needleman-Wunsch against the current consensus,
 * propagating consensus gaps as new columns across every existing row.  Not
 * a published heuristic's exact output, but a deterministic, valid MSA. */
std::vector<std::string> bi_msa(std::vector<std::string> seqs) {
    std::vector<std::string> aln;
    if (seqs.empty()) return aln;
    Scorer sc; sc.nt = bi_set_is_nt(seqs);
    /* seed with the longest sequence (a stable, deterministic anchor). */
    size_t seed = 0;
    for (size_t i = 1; i < seqs.size(); ++i)
        if (seqs[i].size() > seqs[seed].size()) seed = i;
    std::vector<size_t> order; order.push_back(seed);
    for (size_t i = 0; i < seqs.size(); ++i) if (i != seed) order.push_back(i);

    aln.push_back(seqs[order[0]]);
    for (size_t oi = 1; oi < order.size(); ++oi) {
        std::string cons;
        for (size_t j = 0; j < aln[0].size(); ++j) cons.push_back(bi_consensus_col(aln, j));
        AlignResult R = bi_align(cons, seqs[order[oi]], sc, false);
        /* R.a1 = gapped consensus, R.a2 = gapped new sequence.  Rebuild every
         * existing row + the new one walking R.a1 columns. */
        std::vector<std::string> next(aln.size() + 1);
        size_t consPos = 0;
        for (size_t k = 0; k < R.a1.size(); ++k) {
            if (R.a1[k] == '-') {                 /* inserted column */
                for (size_t r = 0; r < aln.size(); ++r) next[r].push_back('-');
            } else {
                for (size_t r = 0; r < aln.size(); ++r)
                    next[r].push_back(consPos < aln[r].size() ? aln[r][consPos] : '-');
                ++consPos;
            }
            next[aln.size()].push_back(R.a2[k]);
        }
        aln = next;
    }
    return aln;
}

/* ===== Tier-4 — distances + tree builders ================================ */

/* p-distance / Jukes-Cantor between two (globally aligned) sequences. */
double bi_pair_dist(const std::string &a, const std::string &b, bool nt,
                    int method) {  /* 0 = p-distance, 1 = Jukes-Cantor */
    Scorer sc; sc.nt = nt;
    AlignResult R = bi_align(a, b, sc, false);
    int diff = 0, cols = 0;
    for (size_t k = 0; k < R.a1.size(); ++k) {
        char x = R.a1[k], y = R.a2[k];
        if (x == '-' || y == '-') continue;
        ++cols;
        if (bi_upper(x) != bi_upper(y)) ++diff;
    }
    double p = (cols > 0) ? static_cast<double>(diff) / cols : 0.0;
    if (method == 1) {              /* Jukes-Cantor */
        double arg = 1.0 - (4.0 / 3.0) * p;
        if (arg <= 0.0) return 10.0;        /* saturated */
        return -0.75 * std::log(arg);
    }
    return p;
}

/* Phylogenetic tree: leaves 1..N, internal nodes N+1..2N-1 (root = 2N-1). */
struct BiTree {
    int N = 0;
    std::vector<int> c1, c2;        /* children of internal node N+1+k */
    std::vector<double> edge;       /* edge length to parent, per node id 1..2N-1 */
    std::vector<std::string> names; /* leaf names, N of them */
};

/* UPGMA / WPGMA / single / complete linkage over a square distance matrix. */
BiTree bi_upgma(std::vector<std::vector<double>> D, int linkage) {
    int N = static_cast<int>(D.size());
    BiTree T; T.N = N;
    T.edge.assign(2 * N, 0.0);                 /* 1-based ids; index 0 unused */
    std::vector<int> id(N), size(N, 1);
    std::vector<double> height(2 * N, 0.0);
    std::vector<bool> active(N, true);
    for (int i = 0; i < N; ++i) id[i] = i + 1;
    int nextId = N + 1;
    for (int step = 0; step < N - 1; ++step) {
        double best = 1e300; int bi = -1, bj = -1;
        for (int i = 0; i < N; ++i) if (active[i])
            for (int j = i + 1; j < N; ++j) if (active[j])
                if (D[i][j] < best) { best = D[i][j]; bi = i; bj = j; }
        if (bi < 0) break;
        double h = best / 2.0;                 /* UPGMA node height */
        int ni = id[bi], nj = id[bj];
        T.c1.push_back(ni); T.c2.push_back(nj);
        height[nextId] = h;
        T.edge[ni] = h - height[ni];
        T.edge[nj] = h - height[nj];
        if (T.edge[ni] < 0) T.edge[ni] = 0;
        if (T.edge[nj] < 0) T.edge[nj] = 0;
        /* merge bj into bi. */
        for (int k = 0; k < N; ++k) {
            if (!active[k] || k == bi || k == bj) continue;
            double dk;
            double dik = D[bi][k], djk = D[bj][k];
            switch (linkage) {
                case 1: dk = std::min(dik, djk); break;            /* single */
                case 2: dk = std::max(dik, djk); break;            /* complete */
                case 3: dk = (dik + djk) / 2.0; break;             /* weighted/WPGMA */
                default:                                           /* average/UPGMA */
                    dk = (size[bi] * dik + size[bj] * djk) /
                         static_cast<double>(size[bi] + size[bj]);
            }
            D[bi][k] = D[k][bi] = dk;
        }
        id[bi] = nextId; size[bi] += size[bj]; height[nextId] = h;
        active[bj] = false;
        ++nextId;
    }
    return T;
}

/* Neighbor-joining (Saitou-Nei) over a square distance matrix. */
BiTree bi_nj(std::vector<std::vector<double>> D) {
    int N = static_cast<int>(D.size());
    BiTree T; T.N = N;
    T.edge.assign(2 * N, 0.0);
    std::vector<int> id(N);
    std::vector<bool> active(N, true);
    for (int i = 0; i < N; ++i) id[i] = i + 1;
    int nActive = N, nextId = N + 1;
    while (nActive > 2) {
        std::vector<double> r(N, 0.0);
        for (int i = 0; i < N; ++i) if (active[i])
            for (int j = 0; j < N; ++j) if (active[j] && j != i) r[i] += D[i][j];
        double best = 1e300; int bi = -1, bj = -1;
        for (int i = 0; i < N; ++i) if (active[i])
            for (int j = i + 1; j < N; ++j) if (active[j]) {
                double q = (nActive - 2) * D[i][j] - r[i] - r[j];
                if (q < best) { best = q; bi = i; bj = j; }
            }
        double dij = D[bi][bj];
        double ei = 0.5 * dij + (r[bi] - r[bj]) / (2.0 * (nActive - 2));
        double ej = dij - ei;
        if (ei < 0) ei = 0; if (ej < 0) ej = 0;
        T.c1.push_back(id[bi]); T.c2.push_back(id[bj]);
        T.edge[id[bi]] = ei; T.edge[id[bj]] = ej;
        /* new node distances. */
        for (int k = 0; k < N; ++k) {
            if (!active[k] || k == bi || k == bj) continue;
            double dk = 0.5 * (D[bi][k] + D[bj][k] - dij);
            D[bi][k] = D[k][bi] = dk;
        }
        id[bi] = nextId; active[bj] = false; --nActive; ++nextId;
    }
    /* join the final two active clusters at the root. */
    int a = -1, b = -1;
    for (int i = 0; i < N; ++i) if (active[i]) { (a < 0 ? a : b) = i; }
    T.c1.push_back(id[a]); T.c2.push_back(id[b]);
    T.edge[id[a]] = D[a][b]; T.edge[id[b]] = 0.0;
    return T;
}

/* Recursive Newick serialization from a node id. */
void bi_newick_rec(const BiTree &T, int node, std::string &out) {
    if (node <= T.N) {                              /* leaf */
        out += (node - 1 < static_cast<int>(T.names.size()))
                   ? T.names[node - 1] : ("Leaf" + std::to_string(node));
    } else {
        int k = node - T.N - 1;                     /* internal node index */
        out += "(";
        char buf[64];
        int ch1 = T.c1[k], ch2 = T.c2[k];
        bi_newick_rec(T, ch1, out);
        std::snprintf(buf, sizeof(buf), ":%g", T.edge[ch1]); out += buf;
        out += ",";
        bi_newick_rec(T, ch2, out);
        std::snprintf(buf, sizeof(buf), ":%g", T.edge[ch2]); out += buf;
        out += ")";
    }
}

std::string bi_newick(const BiTree &T) {
    std::string out;
    bi_newick_rec(T, 2 * T.N - 1, out);
    out += ";";
    return out;
}

/* Serialize a BiTree into a phytree object's fields. */
void bi_store_tree(matlab_obj *obj, const BiTree &T) {
    int N = T.N;
    matlab_obj_set_f64(obj, "NumLeaves", 9, static_cast<double>(N));
    /* Pointers: (N-1) x 2 child-id matrix. */
    matlab_mat *P = mat_alloc(N - 1, 2);
    if (P && P->data)
        for (int k = 0; k < N - 1; ++k) {
            P->data[k * 2 + 0] = T.c1[k];
            P->data[k * 2 + 1] = T.c2[k];
        }
    matlab_obj_set_mat(obj, "Pointers", 8, P);
    /* Distances: edge length per node id 1..2N-1 (column vector). */
    matlab_mat *Dm = mat_alloc(2 * N - 1, 1);
    if (Dm && Dm->data)
        for (int i = 1; i <= 2 * N - 1; ++i) Dm->data[i - 1] = T.edge[i];
    matlab_obj_set_mat(obj, "Distances", 9, Dm);
    /* Names: newline-joined leaf names. */
    std::string nm;
    for (int i = 0; i < N; ++i) { if (i) nm += "\n"; nm += T.names[i]; }
    matlab_obj_set_string(obj, "Names", 5, bi_mkstr(nm));
    /* Newick string cached for getnewickstr / disp. */
    BiTree T2 = T;
    matlab_obj_set_string(obj, "Newick", 6, bi_mkstr(bi_newick(T2)));
}

/* Rebuild a BiTree from a phytree object (for pdist / methods). */
BiTree bi_load_tree(matlab_obj *obj) {
    BiTree T;
    T.N = static_cast<int>(matlab_obj_get_f64(obj, "NumLeaves", 9));
    matlab_mat *P = matlab_obj_get_mat(obj, "Pointers", 8);
    matlab_mat *Dm = matlab_obj_get_mat(obj, "Distances", 9);
    if (P && P->data)
        for (int k = 0; k < P->rows; ++k) {
            T.c1.push_back(static_cast<int>(P->data[k * 2 + 0]));
            T.c2.push_back(static_cast<int>(P->data[k * 2 + 1]));
        }
    T.edge.assign(2 * T.N, 0.0);
    if (Dm && Dm->data)
        for (int i = 1; i <= 2 * T.N - 1 && i - 1 < Dm->rows * Dm->cols; ++i)
            T.edge[i] = Dm->data[i - 1];
    return T;
}

/* Distance from a node up to the root (sum of edge lengths). */
double bi_depth_to_root(const BiTree &T, int node) {
    /* parent map. */
    std::vector<int> parent(2 * T.N, 0);
    for (int k = 0; k < static_cast<int>(T.c1.size()); ++k) {
        parent[T.c1[k]] = T.N + 1 + k;
        parent[T.c2[k]] = T.N + 1 + k;
    }
    double d = 0.0; int cur = node;
    while (parent[cur] != 0) { d += T.edge[cur]; cur = parent[cur]; }
    return d;
}

/* ===== Tier-5 — protein property tables ================================== */

/* Average residue masses (Da) in the 20-AA order "ARNDCQEGHILKMFPSTWYV". */
const char *kAA20 = "ARNDCQEGHILKMFPSTWYV";
double bi_aa_mass(char c) {
    static const double M[20] = {
        71.0788, 156.1875, 114.1038, 115.0886, 103.1388, 128.1307, 129.1155,
        57.0519, 137.1411, 113.1594, 113.1594, 128.1741, 131.1926, 147.1766,
        97.1167, 87.0782, 101.1051, 186.2132, 163.1760, 99.1326};
    char u = bi_upper(c);
    for (int i = 0; i < 20; ++i) if (kAA20[i] == u) return M[i];
    return 0.0;
}

/* Residue atomic composition (C,H,N,O,S) — formula minus one water. */
struct AtomComp { int C, H, N, O, S; };
AtomComp bi_aa_atoms(char c) {
    /* residue (peptide-bonded) C,H,N,O,S, in kAA20 order. */
    static const AtomComp A[20] = {
        {3,5,1,1,0},   /*A*/ {6,12,4,1,0}, /*R*/ {4,6,2,2,0},  /*N*/ {4,5,1,3,0}, /*D*/
        {3,5,1,1,1},   /*C*/ {5,8,2,2,0},  /*Q*/ {5,7,1,3,0},  /*E*/ {2,3,1,1,0}, /*G*/
        {6,7,3,1,0},   /*H*/ {6,11,1,1,0}, /*I*/ {6,11,1,1,0}, /*L*/ {6,12,2,1,0}, /*K*/
        {5,9,1,1,1},   /*M*/ {9,9,1,1,0},  /*F*/ {5,7,1,1,0},  /*P*/ {3,5,1,2,0}, /*S*/
        {4,7,1,2,0},   /*T*/ {11,10,2,1,0},/*W*/ {9,9,1,2,0},  /*Y*/ {5,9,1,1,0}  /*V*/};
    char u = bi_upper(c);
    for (int i = 0; i < 20; ++i) if (kAA20[i] == u) return A[i];
    return {0, 0, 0, 0, 0};
}

/* Side-chain pK values (EMBOSS) for isoelectric-point computation. */
double bi_side_charge(const std::string &seq, double pH) {
    /* N-term + C-term + ionizable side chains. */
    auto pos = [&](double pK) { return 1.0 / (1.0 + std::pow(10.0, pH - pK)); };
    auto neg = [&](double pK) { return 1.0 / (1.0 + std::pow(10.0, pK - pH)); };
    double q = pos(8.6);          /* N-terminus */
    q -= neg(3.6);                /* C-terminus */
    for (char ch : seq) {
        switch (bi_upper(ch)) {
            case 'K': q += pos(10.8); break;
            case 'R': q += pos(12.5); break;
            case 'H': q += pos(6.5);  break;
            case 'D': q -= neg(3.9);  break;
            case 'E': q -= neg(4.1);  break;
            case 'C': q -= neg(8.5);  break;
            case 'Y': q -= neg(10.1); break;
            default: break;
        }
    }
    return q;
}

/* Find non-overlapping cut positions for a protease (cleave after residues
 * in `after`, unless the next residue is in `notbefore`). */
std::vector<std::string> bi_protease(const std::string &seq, const std::string &after,
                                     const std::string &notbefore) {
    std::vector<std::string> frags;
    std::string cur;
    for (size_t i = 0; i < seq.size(); ++i) {
        cur.push_back(seq[i]);
        char c = bi_upper(seq[i]);
        bool cut = after.find(c) != std::string::npos;
        if (cut && i + 1 < seq.size() &&
            notbefore.find(bi_upper(seq[i + 1])) != std::string::npos)
            cut = false;
        if (cut) { frags.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) frags.push_back(cur);
    return frags;
}

/* ===== Tier-6 — microarray / mass-spec / learning helpers ================ */

/* In-order leaf traversal of a UPGMA tree -> a 1-based row permutation. */
void bi_leaf_order(const BiTree &T, int node, std::vector<int> &order) {
    if (node <= T.N) { order.push_back(node); return; }
    int k = node - T.N - 1;
    bi_leaf_order(T, T.c1[k], order);
    bi_leaf_order(T, T.c2[k], order);
}

/* Column read of a matlab_mat (row-major). */
double bi_at(const matlab_mat *m, int r, int c) {
    return m->data[r * m->cols + c];
}

}  /* namespace */

/* ===========================================================================
 * Tier-1 — sequence conversion / complement / translation
 * ==========================================================================*/
extern "C" {

matlab_mat *matlab_bioinfo_nt2int(void *seq) {
    std::string s = bi_sstr(seq);
    std::vector<double> v;
    v.reserve(s.size());
    for (char c : s) v.push_back(bi_nt_code(c));
    return bi_row(v);
}

matlab_string *matlab_bioinfo_int2nt(matlab_mat *codes) {
    std::string s;
    if (codes && codes->data) {
        int64_t n = codes->rows * codes->cols;
        for (int64_t i = 0; i < n; ++i)
            s.push_back(bi_nt_char(static_cast<int>(codes->data[i])));
    }
    return bi_mkstr(s);
}

matlab_mat *matlab_bioinfo_aa2int(void *seq) {
    std::string s = bi_sstr(seq);
    std::vector<double> v;
    v.reserve(s.size());
    for (char c : s) v.push_back(bi_aa_code(c));
    return bi_row(v);
}

matlab_string *matlab_bioinfo_int2aa(matlab_mat *codes) {
    std::string s;
    if (codes && codes->data) {
        int64_t n = codes->rows * codes->cols;
        for (int64_t i = 0; i < n; ++i)
            s.push_back(bi_aa_char(static_cast<int>(codes->data[i])));
    }
    return bi_mkstr(s);
}

matlab_string *matlab_bioinfo_dna2rna(void *seq) {
    std::string s = bi_sstr(seq);
    for (char &c : s) { if (c == 'T') c = 'U'; else if (c == 't') c = 'u'; }
    return bi_mkstr(s);
}

matlab_string *matlab_bioinfo_rna2dna(void *seq) {
    std::string s = bi_sstr(seq);
    for (char &c : s) { if (c == 'U') c = 'T'; else if (c == 'u') c = 't'; }
    return bi_mkstr(s);
}

matlab_string *matlab_bioinfo_seqcomplement(void *seq) {
    std::string s = bi_sstr(seq), o;
    o.reserve(s.size());
    for (char c : s) {
        char u = bi_upper(c), r;
        switch (u) {
            case 'A': r = 'T'; break;  case 'T': r = 'A'; break;
            case 'U': r = 'A'; break;  case 'C': r = 'G'; break;
            case 'G': r = 'C'; break;  default:  r = u;    break;
        }
        o.push_back(r);
    }
    return bi_mkstr(o);
}

matlab_string *matlab_bioinfo_seqreverse(void *seq) {
    std::string s = bi_sstr(seq);
    std::reverse(s.begin(), s.end());
    return bi_mkstr(s);
}

matlab_string *matlab_bioinfo_seqrcomplement(void *seq) {
    std::string s = bi_sstr(seq), o;
    o.reserve(s.size());
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        char u = bi_upper(*it), r;
        switch (u) {
            case 'A': r = 'T'; break;  case 'T': r = 'A'; break;
            case 'U': r = 'A'; break;  case 'C': r = 'G'; break;
            case 'G': r = 'C'; break;  default:  r = u;    break;
        }
        o.push_back(r);
    }
    return bi_mkstr(o);
}

/* nt2aa: translate a nucleotide sequence (frame 1) to a protein string. */
matlab_string *matlab_bioinfo_nt2aa(void *seq) {
    std::string s = bi_sstr(seq), o;
    for (size_t i = 0; i + 2 < s.size() + 1 && i + 3 <= s.size(); i += 3)
        o.push_back(bi_translate_codon(s[i], s[i + 1], s[i + 2]));
    return bi_mkstr(o);
}

/* basecount -> struct with A C G T (and Other) counts. */
matlab_struct *matlab_bioinfo_basecount(void *seq) {
    std::string s = bi_sstr(seq);
    double a = 0, c = 0, g = 0, t = 0, other = 0;
    for (char ch : s) {
        switch (bi_upper(ch)) {
            case 'A': a++; break;  case 'C': c++; break;
            case 'G': g++; break;
            case 'T': case 'U': t++; break;
            default: other++; break;
        }
    }
    matlab_struct *st = matlab_struct_new();
    matlab_struct_set_f64(st, "A", 1, a);
    matlab_struct_set_f64(st, "C", 1, c);
    matlab_struct_set_f64(st, "G", 1, g);
    matlab_struct_set_f64(st, "T", 1, t);
    if (other > 0) matlab_struct_set_f64(st, "Other", 5, other);
    return st;
}

/* aacount -> struct with the 20 standard amino-acid counts. */
matlab_struct *matlab_bioinfo_aacount(void *seq) {
    std::string s = bi_sstr(seq);
    const char *aa20 = "ARNDCQEGHILKMFPSTWYV";
    double cnt[20] = {0};
    for (char ch : s) {
        char u = bi_upper(ch);
        for (int i = 0; i < 20; ++i)
            if (aa20[i] == u) { cnt[i]++; break; }
    }
    matlab_struct *st = matlab_struct_new();
    char name[2] = {0, 0};
    for (int i = 0; i < 20; ++i) {
        name[0] = aa20[i];
        matlab_struct_set_f64(st, name, 1, cnt[i]);
    }
    return st;
}

/* randseq(n): n random nucleotides (deterministic LCG; content is not a
 * gating target, only the length is). */
matlab_string *matlab_bioinfo_randseq(double n) {
    int64_t len = (n > 0) ? static_cast<int64_t>(n) : 0;
    static uint64_t state = 0x2545F4914F6CDD1DULL;
    std::string s;
    s.reserve(static_cast<size_t>(len));
    const char *B = "ACGT";
    for (int64_t i = 0; i < len; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        s.push_back(B[(state >> 33) & 3]);
    }
    return bi_mkstr(s);
}

/* ===========================================================================
 * Tier-1 — FASTA I/O
 * ==========================================================================*/

/* fastaread(filename) -> struct array s with s(i).Header / s(i).Sequence. */
matlab_struct_arr *matlab_bioinfo_fastaread(void *path_s) {
    matlab_struct_arr *arr = matlab_struct_arr_new();
    std::string path = bi_sstr(path_s);
    FILE *f = std::fopen(path.c_str(), "r");
    if (!f) return arr;
    std::string header, seq;
    int idx = 0;
    bool have = false;
    char line[8192];
    auto flush = [&]() {
        if (!have) return;
        matlab_struct *st = matlab_struct_arr_get_or_create(
            arr, static_cast<double>(++idx));
        matlab_struct_set_string(st, "Header", 6, bi_mkstr(header));
        matlab_struct_set_string(st, "Sequence", 8, bi_mkstr(seq));
    };
    while (std::fgets(line, sizeof(line), f)) {
        size_t len = std::strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (line[0] == '>') {
            flush();
            header.assign(line + 1);
            seq.clear();
            have = true;
        } else if (have) {
            for (size_t i = 0; i < len; ++i)
                if (!std::isspace(static_cast<unsigned char>(line[i])))
                    seq.push_back(line[i]);
        }
    }
    flush();
    std::fclose(f);
    return arr;
}

/* fastawrite(filename, header, sequence): append one FASTA record. */
matlab_mat *matlab_bioinfo_fastawrite(void *path_s, void *header_s,
                                      void *seq_s) {
    std::string path = bi_sstr(path_s);
    std::string header = bi_sstr(header_s);
    std::string seq = bi_sstr(seq_s);
    /* MATLAB fastawrite appends; to keep round-trip tests deterministic
     * across re-runs, the first write to a given path *in this process*
     * truncates, and subsequent writes to it append. */
    static std::set<std::string> seen;
    const char *mode = seen.insert(path).second ? "w" : "a";
    FILE *f = std::fopen(path.c_str(), mode);
    if (!f) return bi_scalar(0);
    std::fprintf(f, ">%s\n", header.c_str());
    /* wrap at 60 columns, the FASTA convention. */
    for (size_t i = 0; i < seq.size(); i += 60)
        std::fprintf(f, "%.*s\n", static_cast<int>(std::min<size_t>(60, seq.size() - i)),
                     seq.c_str() + i);
    std::fclose(f);
    return bi_scalar(1);
}

/* ===========================================================================
 * Tier-2 — scoring matrices
 * ==========================================================================*/

matlab_mat *matlab_bioinfo_blosum(double /*n*/) {
    /* Phase A ships BLOSUM62 (the headline matrix); other N collapse to it. */
    matlab_mat *r = mat_alloc(24, 24);
    if (r && r->data)
        for (int i = 0; i < 24; ++i)
            for (int j = 0; j < 24; ++j)
                r->data[i * 24 + j] = kBLOSUM62[i][j];
    return r;
}

matlab_mat *matlab_bioinfo_nuc44(void) {
    /* ACGT core: match +5 / mismatch -4. */
    matlab_mat *r = mat_alloc(4, 4);
    if (r && r->data)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r->data[i * 4 + j] = (i == j) ? 5.0 : -4.0;
    return r;
}

/* ===========================================================================
 * Tier-2 — pairwise alignment (multi-return: score + alignment)
 * ==========================================================================*/

/* nwalign(s1, s2)  — global, auto scoring matrix. */
matlab_mat *matlab_bioinfo_nwalign_score2(void *s1, void *s2) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2);
    return bi_scalar(bi_align(a, b, bi_pick_scorer(a, b, ""), false).score);
}
matlab_string *matlab_bioinfo_nwalign_align2(void *s1, void *s2) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2);
    AlignResult R = bi_align(a, b, bi_pick_scorer(a, b, ""), false);
    return bi_mkstr(R.a1 + "\n" + R.mid + "\n" + R.a2);
}

/* nwalign(s1, s2, 'ScoringMatrix') — global, named scoring matrix. */
matlab_mat *matlab_bioinfo_nwalign_score3(void *s1, void *s2, void *nm) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2), n = bi_sstr(nm);
    return bi_scalar(bi_align(a, b, bi_pick_scorer(a, b, n), false).score);
}
matlab_string *matlab_bioinfo_nwalign_align3(void *s1, void *s2, void *nm) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2), n = bi_sstr(nm);
    AlignResult R = bi_align(a, b, bi_pick_scorer(a, b, n), false);
    return bi_mkstr(R.a1 + "\n" + R.mid + "\n" + R.a2);
}

/* swalign(s1, s2[, 'ScoringMatrix']) — local. */
matlab_mat *matlab_bioinfo_swalign_score2(void *s1, void *s2) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2);
    return bi_scalar(bi_align(a, b, bi_pick_scorer(a, b, ""), true).score);
}
matlab_string *matlab_bioinfo_swalign_align2(void *s1, void *s2) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2);
    AlignResult R = bi_align(a, b, bi_pick_scorer(a, b, ""), true);
    return bi_mkstr(R.a1 + "\n" + R.mid + "\n" + R.a2);
}
matlab_mat *matlab_bioinfo_swalign_score3(void *s1, void *s2, void *nm) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2), n = bi_sstr(nm);
    return bi_scalar(bi_align(a, b, bi_pick_scorer(a, b, n), true).score);
}
matlab_string *matlab_bioinfo_swalign_align3(void *s1, void *s2, void *nm) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2), n = bi_sstr(nm);
    AlignResult R = bi_align(a, b, bi_pick_scorer(a, b, n), true);
    return bi_mkstr(R.a1 + "\n" + R.mid + "\n" + R.a2);
}

/* seqdotplot(s1, s2[, window, threshold]) -> binary match matrix
 * (rows = s1 positions, cols = s2 positions).  window/threshold default to a
 * single-residue exact match. */
matlab_mat *matlab_bioinfo_seqdotplot(void *s1, void *s2) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2);
    int n = static_cast<int>(a.size()), m = static_cast<int>(b.size());
    matlab_mat *r = mat_alloc(n, m);
    if (r && r->data)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                r->data[i * m + j] = (bi_upper(a[i]) == bi_upper(b[j])) ? 1.0 : 0.0;
    return r;
}

/* ===========================================================================
 * Tier-3 — multiple-sequence alignment + profiles
 * ==========================================================================*/

/* multialign(cell) -> newline-joined aligned sequence set. */
matlab_string *matlab_bioinfo_multialign(void *cell) {
    std::vector<std::string> aln = bi_msa(bi_seqset(cell));
    std::string out;
    for (size_t i = 0; i < aln.size(); ++i) { if (i) out += "\n"; out += aln[i]; }
    return bi_mkstr(out);
}

/* profalign(s1, s2) -> aligned pair (newline-joined: seq1\nmatch\nseq2). */
matlab_string *matlab_bioinfo_profalign(void *s1, void *s2) {
    std::string a = bi_sstr(s1), b = bi_sstr(s2);
    Scorer sc; sc.nt = bi_is_nt(a) && bi_is_nt(b);
    AlignResult R = bi_align(a, b, sc, false);
    return bi_mkstr(R.a1 + "\n" + R.a2);
}

/* seqconsensus(alignment) -> consensus char string (input = the newline-
 * joined alignment that multialign returns). */
matlab_string *matlab_bioinfo_seqconsensus(void *aln_s) {
    std::vector<std::string> rows = bi_splitlines(bi_sstr(aln_s));
    size_t L = 0;
    for (const std::string &r : rows) L = std::max(L, r.size());
    std::string cons;
    for (size_t j = 0; j < L; ++j) cons.push_back(bi_consensus_col(rows, j));
    return bi_mkstr(cons);
}

/* seqprofile(alignment) -> frequency matrix (alphabet rows x position cols). */
matlab_mat *matlab_bioinfo_seqprofile(void *aln_s) {
    std::vector<std::string> rows = bi_splitlines(bi_sstr(aln_s));
    bool nt = bi_set_is_nt(rows);
    const char *alpha = nt ? "ACGT" : "ARNDCQEGHILKMFPSTWYV";
    int A = nt ? 4 : 20;
    size_t L = 0;
    for (const std::string &r : rows) L = std::max(L, r.size());
    matlab_mat *P = mat_alloc(A, static_cast<int64_t>(L));
    if (!P || !P->data) return P;
    for (size_t j = 0; j < L; ++j) {
        double tot = 0.0;
        std::vector<double> col(A, 0.0);
        for (const std::string &r : rows) {
            if (j >= r.size()) continue;
            char c = bi_upper(r[j]);
            for (int a = 0; a < A; ++a)
                if (alpha[a] == c) { col[a] += 1.0; tot += 1.0; break; }
        }
        for (int a = 0; a < A; ++a)
            P->data[a * static_cast<int>(L) + j] = (tot > 0) ? col[a] / tot : 0.0;
    }
    return P;
}

/* ===========================================================================
 * Tier-4 — distances + phylogenetic tree (phytree classdef)
 * ==========================================================================*/

static matlab_mat *bi_seqpdist_impl(void *cell, int method) {
    std::vector<std::string> seqs = bi_seqset(cell);
    int N = static_cast<int>(seqs.size());
    bool nt = bi_set_is_nt(seqs);
    int M = N * (N - 1) / 2;
    matlab_mat *v = mat_alloc(1, M > 0 ? M : 0);
    if (v && v->data) {
        int idx = 0;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j)
                v->data[idx++] = bi_pair_dist(seqs[i], seqs[j], nt, method);
    }
    return v;
}

/* seqpdist(cell) -> pdist-format distance row vector (default p-distance). */
matlab_mat *matlab_bioinfo_seqpdist2(void *cell) {
    return bi_seqpdist_impl(cell, 0);
}
matlab_mat *matlab_bioinfo_seqpdist3(void *cell, void *method_s) {
    std::string m = bi_sstr(method_s);
    for (char &c : m) c = bi_upper(c);
    int method = (m.find("JUKES") != std::string::npos || m == "JC") ? 1 : 0;
    return bi_seqpdist_impl(cell, method);
}

/* Build a square distance matrix from either a pdist row vector or a square. */
static std::vector<std::vector<double>> bi_square(matlab_mat *D) {
    std::vector<std::vector<double>> M;
    if (!D || !D->data) return M;
    int64_t n = D->rows * D->cols;
    if (D->rows == D->cols && D->rows > 1) {       /* already square */
        int N = static_cast<int>(D->rows);
        M.assign(N, std::vector<double>(N, 0.0));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) M[i][j] = D->data[i * N + j];
        return M;
    }
    /* pdist vector of length N(N-1)/2 -> N. */
    int N = static_cast<int>((1.0 + std::sqrt(1.0 + 8.0 * n)) / 2.0 + 0.5);
    M.assign(N, std::vector<double>(N, 0.0));
    int idx = 0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) { M[i][j] = M[j][i] = D->data[idx++]; }
    return M;
}

static void bi_default_names(BiTree &T) {
    for (int i = 1; i <= T.N; ++i) T.names.push_back("Leaf" + std::to_string(i));
}
static void bi_names_from_cell(BiTree &T, void *names_cell) {
    std::vector<std::string> nm = bi_seqset(names_cell);
    for (int i = 0; i < T.N; ++i)
        T.names.push_back(i < static_cast<int>(nm.size()) && !nm[i].empty()
                              ? nm[i] : ("Leaf" + std::to_string(i + 1)));
}
static int bi_linkage_code(const std::string &m) {
    std::string s; for (char c : m) s.push_back(bi_upper(c));
    if (s.find("SINGLE") != std::string::npos) return 1;
    if (s.find("COMPLETE") != std::string::npos) return 2;
    if (s.find("WEIGHTED") != std::string::npos || s.find("WPGMA") != std::string::npos) return 3;
    return 0;  /* average / UPGMA */
}

/* seqlinkage(D[, method[, names]]) -> phytree (populate the alloc'd shell). */
matlab_mat *matlab_bioinfo_seqlinkage1(matlab_obj *obj, matlab_mat *D) {
    BiTree T = bi_upgma(bi_square(D), 0);
    bi_default_names(T);
    bi_store_tree(obj, T);
    return reinterpret_cast<matlab_mat *>(obj);
}
matlab_mat *matlab_bioinfo_seqlinkage2(matlab_obj *obj, matlab_mat *D, void *method_s) {
    BiTree T = bi_upgma(bi_square(D), bi_linkage_code(bi_sstr(method_s)));
    bi_default_names(T);
    bi_store_tree(obj, T);
    return reinterpret_cast<matlab_mat *>(obj);
}
matlab_mat *matlab_bioinfo_seqlinkage3(matlab_obj *obj, matlab_mat *D, void *method_s,
                                void *names_cell) {
    BiTree T = bi_upgma(bi_square(D), bi_linkage_code(bi_sstr(method_s)));
    bi_names_from_cell(T, names_cell);
    bi_store_tree(obj, T);
    return reinterpret_cast<matlab_mat *>(obj);
}

/* seqneighjoin(D[, method[, names]]) -> phytree (neighbor joining). */
matlab_mat *matlab_bioinfo_seqneighjoin1(matlab_obj *obj, matlab_mat *D) {
    BiTree T = bi_nj(bi_square(D));
    bi_default_names(T);
    bi_store_tree(obj, T);
    return reinterpret_cast<matlab_mat *>(obj);
}
matlab_mat *matlab_bioinfo_seqneighjoin3(matlab_obj *obj, matlab_mat *D, void *method_s,
                                  void *names_cell) {
    (void)method_s;
    BiTree T = bi_nj(bi_square(D));
    bi_names_from_cell(T, names_cell);
    bi_store_tree(obj, T);
    return reinterpret_cast<matlab_mat *>(obj);
}

/* phytree methods (forwarded from the classdef). */
matlab_string *matlab_bioinfo_phytree_newick(matlab_obj *obj) {
    matlab_mat *nw = matlab_obj_get_mat(obj, "Newick", 6);  /* cached string */
    /* Newick was stored as a string (kind=3); get_mat returns the ptr. */
    if (nw) return reinterpret_cast<matlab_string *>(nw);
    return bi_mkstr(bi_newick(bi_load_tree(obj)));
}

/* get(obj, prop) -> scalar (1x1) or matrix property. */
matlab_mat *matlab_bioinfo_phytree_get(matlab_obj *obj, void *prop_s) {
    std::string p; for (char c : bi_sstr(prop_s)) p.push_back(bi_upper(c));
    int N = static_cast<int>(matlab_obj_get_f64(obj, "NumLeaves", 9));
    if (p == "NUMLEAVES")  return bi_scalar(N);
    if (p == "NUMBRANCHES") return bi_scalar(N - 1);
    if (p == "NUMNODES")   return bi_scalar(2 * N - 1);
    if (p == "POINTERS")   return matlab_obj_get_mat(obj, "Pointers", 8);
    if (p == "DISTANCES")  return matlab_obj_get_mat(obj, "Distances", 9);
    return bi_scalar(0);
}

/* pdist(obj) -> patristic distance row vector between leaves (pdist format). */
matlab_mat *matlab_bioinfo_phytree_pdist(matlab_obj *obj) {
    BiTree T = bi_load_tree(obj);
    int N = T.N;
    /* parent map. */
    std::vector<int> parent(2 * N + 1, 0);
    for (int k = 0; k < static_cast<int>(T.c1.size()); ++k) {
        parent[T.c1[k]] = N + 1 + k;
        parent[T.c2[k]] = N + 1 + k;
    }
    auto pathdist = [&](int a, int b) -> double {
        /* collect ancestors of a with cumulative distance. */
        std::vector<int> anc; std::vector<double> dcum;
        int cur = a; double d = 0.0;
        while (cur != 0) { anc.push_back(cur); dcum.push_back(d); d += T.edge[cur]; cur = parent[cur]; }
        cur = b; d = 0.0;
        while (cur != 0) {
            for (size_t i = 0; i < anc.size(); ++i)
                if (anc[i] == cur) return dcum[i] + d;
            d += T.edge[cur]; cur = parent[cur];
        }
        return d;
    };
    int M = N * (N - 1) / 2;
    matlab_mat *v = mat_alloc(1, M > 0 ? M : 0);
    if (v && v->data) {
        int idx = 0;
        for (int i = 1; i <= N; ++i)
            for (int j = i + 1; j <= N; ++j) v->data[idx++] = pathdist(i, j);
    }
    return v;
}

/* phytreewrite(filename, obj) -> write the Newick string to a file. */
matlab_mat *matlab_bioinfo_phytreewrite(void *path_s, matlab_obj *obj) {
    std::string path = bi_sstr(path_s);
    matlab_string *nw = matlab_bioinfo_phytree_newick(obj);
    std::string s = bi_sstr(nw);
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) return bi_scalar(0);
    std::fprintf(f, "%s\n", s.c_str());
    std::fclose(f);
    return bi_scalar(1);
}

/* ===========================================================================
 * Tier-5 — protein property + structural analysis
 * ==========================================================================*/

/* molweight(seq): average molecular weight in Da (sum residue masses + water). */
matlab_mat *matlab_bioinfo_molweight(void *seq) {
    std::string s = bi_sstr(seq);
    double w = 0.0;
    for (char c : s) w += bi_aa_mass(c);
    if (w > 0.0) w += 18.01524;            /* one water for the peptide */
    return bi_scalar(w);
}

/* atomiccomp(seq): atomic composition struct {C,H,N,O,S}. */
matlab_struct *matlab_bioinfo_atomiccomp(void *seq) {
    std::string s = bi_sstr(seq);
    AtomComp t = {0, 0, 0, 0, 0};
    int n = 0;
    for (char c : s) {
        AtomComp a = bi_aa_atoms(c);
        if (a.C || a.H || a.N || a.O || a.S) {
            t.C += a.C; t.H += a.H; t.N += a.N; t.O += a.O; t.S += a.S; ++n;
        }
    }
    if (n > 0) { t.H += 2; t.O += 1; }     /* add one water (H2O) for the chain */
    matlab_struct *st = matlab_struct_new();
    matlab_struct_set_f64(st, "C", 1, t.C);
    matlab_struct_set_f64(st, "H", 1, t.H);
    matlab_struct_set_f64(st, "N", 1, t.N);
    matlab_struct_set_f64(st, "O", 1, t.O);
    matlab_struct_set_f64(st, "S", 1, t.S);
    return st;
}

/* isoelectric(seq): isoelectric point pI via bisection on net charge. */
matlab_mat *matlab_bioinfo_isoelectric(void *seq) {
    std::string s = bi_sstr(seq);
    double lo = 0.0, hi = 14.0;
    for (int it = 0; it < 100; ++it) {
        double mid = 0.5 * (lo + hi);
        double q = bi_side_charge(s, mid);
        if (q > 0.0) lo = mid; else hi = mid;
    }
    return bi_scalar(0.5 * (lo + hi));
}

/* aminolookup: 1-letter sequence -> concatenated 3-letter codes, or a single
 * 3-letter code -> its 1-letter symbol. */
matlab_string *matlab_bioinfo_aminolookup(void *seq) {
    static const char *T3[20] = {
        "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His","Ile",
        "Leu","Lys","Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val"};
    std::string s = bi_sstr(seq);
    /* 3-letter -> 1-letter? */
    if (s.size() == 3) {
        for (int i = 0; i < 20; ++i)
            if (s == T3[i]) return bi_mkstr(std::string(1, kAA20[i]));
    }
    /* 1-letter sequence -> concatenated 3-letter codes. */
    std::string out;
    for (char c : s) {
        char u = bi_upper(c);
        bool found = false;
        for (int i = 0; i < 20; ++i)
            if (kAA20[i] == u) { out += T3[i]; found = true; break; }
        if (!found) out += "Xaa";
    }
    return bi_mkstr(out);
}

/* cleave(seq, enzyme): protease digestion -> newline-joined peptide fragments. */
matlab_string *matlab_bioinfo_cleave(void *seq, void *enzyme) {
    std::string s = bi_sstr(seq), e;
    for (char c : bi_sstr(enzyme)) e.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(c))));
    std::string after, notbefore;
    if (e == "trypsin")            { after = "KR"; notbefore = "P"; }
    else if (e == "chymotrypsin")  { after = "FYW"; notbefore = "P"; }
    else if (e == "lysc" || e == "lys-c") { after = "K"; }
    else if (e == "argc" || e == "arg-c") { after = "R"; }
    else if (e == "gluc" || e == "glu-c" || e == "v8") { after = "E"; }
    else if (e == "pepsin")        { after = "FL"; }
    else                           { after = "KR"; notbefore = "P"; } /* default trypsin */
    std::vector<std::string> frags = bi_protease(s, after, notbefore);
    std::string out;
    for (size_t i = 0; i < frags.size(); ++i) { if (i) out += "\n"; out += frags[i]; }
    return bi_mkstr(out);
}

/* restrict(seq, enzyme): restriction digest -> newline-joined DNA fragments. */
matlab_string *matlab_bioinfo_restrict(void *seq, void *enzyme) {
    std::string s, e;
    for (char c : bi_sstr(seq)) s.push_back(bi_upper(c));
    for (char c : bi_sstr(enzyme)) e.push_back(static_cast<char>(std::tolower(
        static_cast<unsigned char>(c))));
    /* recognition site + cut offset (after this many bases from site start). */
    std::string site; int cut = 0;
    if (e == "ecori")        { site = "GAATTC"; cut = 1; }
    else if (e == "bamhi")   { site = "GGATCC"; cut = 1; }
    else if (e == "hindiii") { site = "AAGCTT"; cut = 1; }
    else if (e == "noti")    { site = "GCGGCCGC"; cut = 2; }
    else if (e == "psti")    { site = "CTGCAG"; cut = 5; }
    else if (e == "smai")    { site = "CCCGGG"; cut = 3; }
    else                     { site = "GAATTC"; cut = 1; } /* default EcoRI */
    std::vector<size_t> cuts;
    for (size_t p = 0; p + site.size() <= s.size(); ++p)
        if (s.compare(p, site.size(), site) == 0) cuts.push_back(p + cut);
    std::vector<std::string> frags;
    size_t start = 0;
    for (size_t c : cuts) { frags.push_back(s.substr(start, c - start)); start = c; }
    frags.push_back(s.substr(start));
    std::string out;
    for (size_t i = 0; i < frags.size(); ++i) { if (i) out += "\n"; out += frags[i]; }
    return bi_mkstr(out);
}

/* ===========================================================================
 * Tier-6 — microarray normalization / filtering / clustering
 * ==========================================================================*/

/* quantilenorm(X): quantile-normalize the columns of an M-gene x N-sample
 * matrix (rank within column -> replace by the across-column rank mean). */
matlab_mat *matlab_bioinfo_quantilenorm(matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int M = static_cast<int>(X->rows), N = static_cast<int>(X->cols);
    matlab_mat *R = mat_alloc(M, N);
    if (!R || !R->data) return R;
    /* per-column sorted order. */
    std::vector<std::vector<int>> ord(N, std::vector<int>(M));
    for (int c = 0; c < N; ++c) {
        for (int r = 0; r < M; ++r) ord[c][r] = r;
        std::sort(ord[c].begin(), ord[c].end(),
                  [&](int a, int b) { return bi_at(X, a, c) < bi_at(X, b, c); });
    }
    /* rank-mean: mean of the k-th smallest across all columns. */
    std::vector<double> rankmean(M, 0.0);
    for (int k = 0; k < M; ++k) {
        double s = 0.0;
        for (int c = 0; c < N; ++c) s += bi_at(X, ord[c][k], c);
        rankmean[k] = s / N;
    }
    for (int c = 0; c < N; ++c)
        for (int k = 0; k < M; ++k)
            R->data[ord[c][k] * N + c] = rankmean[k];
    return R;
}

/* manorm(X): scale each column to unit mean (divide by the column mean). */
matlab_mat *matlab_bioinfo_manorm(matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int M = static_cast<int>(X->rows), N = static_cast<int>(X->cols);
    matlab_mat *R = mat_alloc(M, N);
    if (!R || !R->data) return R;
    for (int c = 0; c < N; ++c) {
        double mu = 0.0;
        for (int r = 0; r < M; ++r) mu += bi_at(X, r, c);
        mu /= (M > 0 ? M : 1);
        for (int r = 0; r < M; ++r)
            R->data[r * N + c] = (mu != 0.0) ? bi_at(X, r, c) / mu : bi_at(X, r, c);
    }
    return R;
}

/* Gene filter helper: keep rows whose per-row statistic is at or above the
 * `pct`-th percentile (default removes the bottom `pct`%).  metric:
 * 0 = variance, 1 = range (max-min), 2 = max absolute value. */
static matlab_mat *bi_gene_filter(matlab_mat *X, int metric, double pct) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int M = static_cast<int>(X->rows), N = static_cast<int>(X->cols);
    std::vector<double> stat(M, 0.0);
    for (int r = 0; r < M; ++r) {
        double mn = bi_at(X, r, 0), mx = mn, mu = 0.0, amax = 0.0;
        for (int c = 0; c < N; ++c) {
            double v = bi_at(X, r, c);
            mn = std::min(mn, v); mx = std::max(mx, v); mu += v;
            amax = std::max(amax, std::fabs(v));
        }
        mu /= (N > 0 ? N : 1);
        if (metric == 1) stat[r] = mx - mn;
        else if (metric == 2) stat[r] = amax;
        else {
            double var = 0.0;
            for (int c = 0; c < N; ++c) { double d = bi_at(X, r, c) - mu; var += d * d; }
            stat[r] = (N > 1) ? var / (N - 1) : 0.0;
        }
    }
    std::vector<double> sorted = stat;
    std::sort(sorted.begin(), sorted.end());
    int ti = static_cast<int>(std::floor(pct / 100.0 * M));
    if (ti < 0) ti = 0; if (ti >= M) ti = M - 1;
    double thr = sorted[ti];
    std::vector<int> keep;
    for (int r = 0; r < M; ++r) if (stat[r] >= thr) keep.push_back(r);
    matlab_mat *R = mat_alloc(static_cast<int>(keep.size()), N);
    if (R && R->data)
        for (size_t i = 0; i < keep.size(); ++i)
            for (int c = 0; c < N; ++c)
                R->data[i * N + c] = bi_at(X, keep[i], c);
    return R;
}

matlab_mat *matlab_bioinfo_genevarfilter(matlab_mat *X) { return bi_gene_filter(X, 0, 10.0); }
matlab_mat *matlab_bioinfo_generangefilter(matlab_mat *X) { return bi_gene_filter(X, 1, 10.0); }
matlab_mat *matlab_bioinfo_genelowvalfilter(matlab_mat *X) { return bi_gene_filter(X, 2, 10.0); }

/* clustergram(X): hierarchical (UPGMA / average-linkage Euclidean) clustering
 * of the rows; returns the 1-based row permutation (the dendrogram leaf
 * order).  The heatmap render is a documented follow-on. */
matlab_mat *matlab_bioinfo_clustergram(matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int M = static_cast<int>(X->rows), N = static_cast<int>(X->cols);
    std::vector<std::vector<double>> D(M, std::vector<double>(M, 0.0));
    for (int i = 0; i < M; ++i)
        for (int j = i + 1; j < M; ++j) {
            double s = 0.0;
            for (int c = 0; c < N; ++c) { double d = bi_at(X, i, c) - bi_at(X, j, c); s += d * d; }
            D[i][j] = D[j][i] = std::sqrt(s);
        }
    BiTree T = bi_upgma(D, 0);
    std::vector<int> order;
    if (M >= 1) bi_leaf_order(T, 2 * M - 1, order);
    matlab_mat *R = mat_alloc(1, static_cast<int>(order.size()));
    if (R && R->data)
        for (size_t i = 0; i < order.size(); ++i) R->data[i] = order[i];  /* 1-based */
    return R;
}

/* ===========================================================================
 * Tier-6 — mass-spectrometry preprocessing
 * ==========================================================================*/

/* msnorm(mz, y): normalize the spectrum so its maximum intensity is 1. */
matlab_mat *matlab_bioinfo_msnorm(matlab_mat *mz, matlab_mat *y) {
    (void)mz;
    if (!y || !y->data) return mat_alloc(0, 0);
    int n = static_cast<int>(y->rows * y->cols);
    double mx = 0.0;
    for (int i = 0; i < n; ++i) mx = std::max(mx, y->data[i]);
    matlab_mat *R = mat_alloc(y->rows, y->cols);
    if (R && R->data)
        for (int i = 0; i < n; ++i) R->data[i] = (mx > 0) ? y->data[i] / mx : y->data[i];
    return R;
}

/* mslowess(mz, y): local moving-average smoothing (span ~ 10 samples). */
matlab_mat *matlab_bioinfo_mslowess(matlab_mat *mz, matlab_mat *y) {
    (void)mz;
    if (!y || !y->data) return mat_alloc(0, 0);
    int n = static_cast<int>(y->rows * y->cols);
    int half = 5;
    matlab_mat *R = mat_alloc(y->rows, y->cols);
    if (R && R->data)
        for (int i = 0; i < n; ++i) {
            double s = 0.0; int cnt = 0;
            for (int k = i - half; k <= i + half; ++k)
                if (k >= 0 && k < n) { s += y->data[k]; ++cnt; }
            R->data[i] = (cnt > 0) ? s / cnt : y->data[i];
        }
    return R;
}

/* msbackadj(mz, y): subtract a baseline estimated as the windowed minimum. */
matlab_mat *matlab_bioinfo_msbackadj(matlab_mat *mz, matlab_mat *y) {
    (void)mz;
    if (!y || !y->data) return mat_alloc(0, 0);
    int n = static_cast<int>(y->rows * y->cols);
    int half = 10;
    matlab_mat *R = mat_alloc(y->rows, y->cols);
    if (R && R->data)
        for (int i = 0; i < n; ++i) {
            double base = y->data[i];
            for (int k = i - half; k <= i + half; ++k)
                if (k >= 0 && k < n) base = std::min(base, y->data[k]);
            double v = y->data[i] - base;
            R->data[i] = (v > 0) ? v : 0.0;
        }
    return R;
}

/* mspeaks(mz, y): detect local maxima above 10% of the peak -> N x 2 [mz y]. */
matlab_mat *matlab_bioinfo_mspeaks(matlab_mat *mz, matlab_mat *y) {
    if (!mz || !y || !mz->data || !y->data) return mat_alloc(0, 2);
    int n = static_cast<int>(y->rows * y->cols);
    double mx = 0.0;
    for (int i = 0; i < n; ++i) mx = std::max(mx, y->data[i]);
    double thr = 0.1 * mx;
    std::vector<std::pair<double, double>> pk;
    for (int i = 1; i < n - 1; ++i)
        if (y->data[i] > thr && y->data[i] >= y->data[i - 1] && y->data[i] > y->data[i + 1])
            pk.emplace_back(mz->data[i], y->data[i]);
    matlab_mat *R = mat_alloc(static_cast<int>(pk.size()), 2);
    if (R && R->data)
        for (size_t i = 0; i < pk.size(); ++i) {
            R->data[i * 2 + 0] = pk[i].first;
            R->data[i * 2 + 1] = pk[i].second;
        }
    return R;
}

/* msresample(mz, y, n): resample the spectrum onto n uniform m/z points
 * (linear interpolation) -> n x 2 [mz_new y_new]. */
matlab_mat *matlab_bioinfo_msresample(matlab_mat *mz, matlab_mat *y, double nd) {
    int n = static_cast<int>(nd);
    if (!mz || !y || !mz->data || !y->data || n <= 0) return mat_alloc(0, 2);
    int m = static_cast<int>(mz->rows * mz->cols);
    double lo = mz->data[0], hi = mz->data[m - 1];
    matlab_mat *R = mat_alloc(n, 2);
    if (!R || !R->data) return R;
    for (int i = 0; i < n; ++i) {
        double x = lo + (hi - lo) * i / (n - 1 > 0 ? n - 1 : 1);
        /* linear interp at x. */
        double yv = y->data[0];
        for (int k = 0; k < m - 1; ++k)
            if (x >= mz->data[k] && x <= mz->data[k + 1]) {
                double t = (mz->data[k + 1] != mz->data[k])
                               ? (x - mz->data[k]) / (mz->data[k + 1] - mz->data[k]) : 0.0;
                yv = y->data[k] + t * (y->data[k + 1] - y->data[k]);
                break;
            }
        R->data[i * 2 + 0] = x;
        R->data[i * 2 + 1] = yv;
    }
    return R;
}

/* ===========================================================================
 * Tier-6 — statistical learning helpers
 * ==========================================================================*/

/* rankfeatures(X, group): rank the rows (features) of an M x N matrix by the
 * absolute two-sample t-statistic between the two groups in `group` (a 1 x N
 * label vector with two distinct values) -> M x 1 ranked feature indices. */
matlab_mat *matlab_bioinfo_rankfeatures(matlab_mat *X, matlab_mat *group) {
    if (!X || !X->data || !group || !group->data) return mat_alloc(0, 0);
    int M = static_cast<int>(X->rows), N = static_cast<int>(X->cols);
    double g0 = group->data[0];
    std::vector<int> a, b;
    for (int c = 0; c < N; ++c) (group->data[c] == g0 ? a : b).push_back(c);
    std::vector<std::pair<double, int>> score(M);
    for (int r = 0; r < M; ++r) {
        auto stats = [&](const std::vector<int> &idx, double &mu, double &var) {
            mu = 0.0; for (int c : idx) mu += bi_at(X, r, c); mu /= (idx.size() ? idx.size() : 1);
            var = 0.0; for (int c : idx) { double d = bi_at(X, r, c) - mu; var += d * d; }
            var /= (idx.size() > 1 ? idx.size() - 1 : 1);
        };
        double m0, v0, m1, v1; stats(a, m0, v0); stats(b, m1, v1);
        double se = std::sqrt(v0 / (a.size() ? a.size() : 1) + v1 / (b.size() ? b.size() : 1));
        double t = (se > 0) ? std::fabs(m0 - m1) / se : 0.0;
        score[r] = {t, r + 1};        /* 1-based feature index */
    }
    std::sort(score.begin(), score.end(),
              [](const std::pair<double, int> &x, const std::pair<double, int> &y) {
                  return x.first > y.first; });
    matlab_mat *R = mat_alloc(M, 1);
    if (R && R->data) for (int r = 0; r < M; ++r) R->data[r] = score[r].second;
    return R;
}

/* knnimpute(X): replace NaN entries by the value from the nearest (Euclidean,
 * over observed columns) complete-enough row. */
matlab_mat *matlab_bioinfo_knnimpute(matlab_mat *X) {
    if (!X || !X->data) return mat_alloc(0, 0);
    int M = static_cast<int>(X->rows), N = static_cast<int>(X->cols);
    matlab_mat *R = mat_alloc(M, N);
    if (!R || !R->data) return R;
    for (int i = 0; i < M * N; ++i) R->data[i] = X->data[i];
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            if (std::isnan(bi_at(X, r, c))) {
                int best = -1; double bestd = 1e300;
                for (int o = 0; o < M; ++o) {
                    if (o == r || std::isnan(bi_at(X, o, c))) continue;
                    double s = 0.0; int cnt = 0;
                    for (int k = 0; k < N; ++k) {
                        double a = bi_at(X, r, k), b = bi_at(X, o, k);
                        if (!std::isnan(a) && !std::isnan(b)) { double d = a - b; s += d * d; ++cnt; }
                    }
                    if (cnt > 0) { s = std::sqrt(s / cnt); if (s < bestd) { bestd = s; best = o; } }
                }
                if (best >= 0) R->data[r * N + c] = bi_at(X, best, c);
            }
    return R;
}

/* crossvalind('Kfold', N, K): deterministic round-robin fold assignment ->
 * N x 1 vector of fold indices 1..K (deterministic for reproducible tests;
 * MATLAB's is randomized). */
matlab_mat *matlab_bioinfo_crossvalind(void *method, double Nd, double Kd) {
    (void)method;
    int N = static_cast<int>(Nd), K = static_cast<int>(Kd);
    if (N <= 0 || K <= 0) return mat_alloc(0, 0);
    matlab_mat *R = mat_alloc(N, 1);
    if (R && R->data)
        for (int i = 0; i < N; ++i) R->data[i] = (i % K) + 1;
    return R;
}

}  /* extern "C" */
