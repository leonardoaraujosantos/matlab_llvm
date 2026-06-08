/* ============================================================================
 * runtime_vision.cpp — Computer Vision Toolbox runtime
 * ----------------------------------------------------------------------------
 * Tier-1: feature detection + description + matching.
 *   detectHarrisFeatures / detectMinEigenFeatures / detectFASTFeatures
 *   (return K x 2 [x y] corner-location matrices, strongest-first);
 *   extractFeatures (normalized intensity-patch descriptors);
 *   matchFeatures (nearest-neighbour + Lowe ratio test);
 *   extractHOGFeatures / extractLBPFeatures (fixed-length descriptor vectors).
 *
 * Representation: images are M x N double matrices (grayscale, row-major);
 * feature points are K x 2 [x y] (1-based, x=column y=row) matrices;
 * descriptors are K x D matrices; everything is the shipped real-matrix lane
 * (no classdef, no complex, no 3-D return) so the surface compiles + runs
 * identically under AOT and the JIT/DAP path.
 *
 * Built over the shipped Image Processing gradient/filter substrate + dense
 * linear algebra; no external dependency (no OpenCV), no Deep Learning.
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

extern "C" matlab_string *matlab_string_from_literal(const char *src, int64_t len);

namespace {

constexpr double kPi = 3.14159265358979323846;

/* matlab_string layout (matches runtime/matlab_runtime.cpp). */
struct cv_string_s { char *data; int64_t len; };
std::string cv_sstr(const void *s) {
    if (!s) return std::string();
    const cv_string_s *p = reinterpret_cast<const cv_string_s *>(s);
    if (!p->data || p->len <= 0 || p->len > 4096) return std::string();
    std::string out(p->data, p->data + p->len);
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

/* Row-major image accessor with edge clamping. */
struct Img {
    const double *d; int R, C;
    Img(const matlab_mat *m) : d(m ? m->data : nullptr),
        R(m ? static_cast<int>(m->rows) : 0), C(m ? static_cast<int>(m->cols) : 0) {}
    double at(int r, int c) const {
        if (r < 0) r = 0; if (r >= R) r = R - 1;
        if (c < 0) c = 0; if (c >= C) c = C - 1;
        return d[r * C + c];
    }
};

matlab_mat *cv_mat(const std::vector<double> &v, int rows, int cols) {
    matlab_mat *r = mat_alloc(rows, cols);
    if (r && r->data)
        for (int i = 0; i < rows * cols; ++i) r->data[i] = (i < static_cast<int>(v.size())) ? v[i] : 0.0;
    return r;
}
matlab_mat *cv_scalar(double v) { return cv_mat({v}, 1, 1); }

/* Sobel gradients. */
void cv_gradients(const Img &I, std::vector<double> &Ix, std::vector<double> &Iy) {
    Ix.assign(I.R * I.C, 0.0); Iy.assign(I.R * I.C, 0.0);
    for (int r = 0; r < I.R; ++r)
        for (int c = 0; c < I.C; ++c) {
            double gx = (I.at(r-1,c+1)+2*I.at(r,c+1)+I.at(r+1,c+1))
                      - (I.at(r-1,c-1)+2*I.at(r,c-1)+I.at(r+1,c-1));
            double gy = (I.at(r+1,c-1)+2*I.at(r+1,c)+I.at(r+1,c+1))
                      - (I.at(r-1,c-1)+2*I.at(r-1,c)+I.at(r-1,c+1));
            Ix[r*I.C+c] = gx / 8.0; Iy[r*I.C+c] = gy / 8.0;
        }
}

/* Non-max suppression over a response map -> strongest-first [x y] list. */
std::vector<std::pair<double,std::pair<int,int>>>
cv_corner_peaks(const std::vector<double> &R, int H, int W, double relThresh, int win) {
    double mx = 0.0;
    for (double v : R) mx = std::max(mx, v);
    double thr = relThresh * mx;
    std::vector<std::pair<double,std::pair<int,int>>> pts;
    for (int r = win; r < H - win; ++r)
        for (int c = win; c < W - win; ++c) {
            double v = R[r*W+c];
            if (v < thr || v <= 0) continue;
            bool peak = true;
            for (int dr = -win; dr <= win && peak; ++dr)
                for (int dc = -win; dc <= win; ++dc)
                    if ((dr||dc) && R[(r+dr)*W+(c+dc)] > v) { peak = false; break; }
            if (peak) pts.push_back({v, {c + 1, r + 1}});   /* [x y] 1-based */
        }
    std::sort(pts.begin(), pts.end(),
              [](auto &a, auto &b){ return a.first > b.first; });
    return pts;
}

matlab_mat *cv_points_mat(const std::vector<std::pair<double,std::pair<int,int>>> &pts) {
    matlab_mat *r = mat_alloc(static_cast<int64_t>(pts.size()), 2);
    if (r && r->data)
        for (size_t i = 0; i < pts.size(); ++i) {
            r->data[i*2+0] = pts[i].second.first;   /* x */
            r->data[i*2+1] = pts[i].second.second;  /* y */
        }
    return r;
}

/* ===== small linear algebra (self-contained) ============================= */

/* Jacobi eigen-decomposition of a symmetric n x n matrix A (row-major,
 * overwritten); eigenvectors in V (columns), eigenvalues in w. */
void cv_jacobi(std::vector<double> &A, int n, std::vector<double> &V,
               std::vector<double> &w) {
    V.assign(n*n, 0.0); for (int i=0;i<n;i++) V[i*n+i]=1.0;
    for (int sweep=0; sweep<100; ++sweep) {
        double off=0; for (int p=0;p<n;p++) for(int q=p+1;q<n;q++) off+=A[p*n+q]*A[p*n+q];
        if (off < 1e-20) break;
        for (int p=0;p<n;p++) for (int q=p+1;q<n;q++){
            if (std::fabs(A[p*n+q])<1e-18) continue;
            double th=(A[q*n+q]-A[p*n+p])/(2*A[p*n+q]);
            double t=(th>=0?1:-1)/(std::fabs(th)+std::sqrt(th*th+1));
            double c=1/std::sqrt(t*t+1), s=t*c;
            for (int i=0;i<n;i++){
                double aip=A[i*n+p], aiq=A[i*n+q];
                A[i*n+p]=c*aip-s*aiq; A[i*n+q]=s*aip+c*aiq;
            }
            for (int i=0;i<n;i++){
                double api=A[p*n+i], aqi=A[q*n+i];
                A[p*n+i]=c*api-s*aqi; A[q*n+i]=s*api+c*aqi;
            }
            for (int i=0;i<n;i++){
                double vip=V[i*n+p], viq=V[i*n+q];
                V[i*n+p]=c*vip-s*viq; V[i*n+q]=s*vip+c*viq;
            }
        }
    }
    w.assign(n,0.0); for (int i=0;i<n;i++) w[i]=A[i*n+i];
}

/* Invert a small n x n matrix (Gauss-Jordan); returns false if singular. */
bool cv_inv(std::vector<double> M, int n, std::vector<double> &inv) {
    inv.assign(n*n,0.0); for(int i=0;i<n;i++) inv[i*n+i]=1.0;
    for (int col=0; col<n; ++col){
        int piv=col; for(int r=col+1;r<n;r++) if(std::fabs(M[r*n+col])>std::fabs(M[piv*n+col])) piv=r;
        if (std::fabs(M[piv*n+col])<1e-12) return false;
        if (piv!=col) for(int k=0;k<n;k++){ std::swap(M[piv*n+k],M[col*n+k]); std::swap(inv[piv*n+k],inv[col*n+k]); }
        double d=M[col*n+col];
        for(int k=0;k<n;k++){ M[col*n+k]/=d; inv[col*n+k]/=d; }
        for(int r=0;r<n;r++) if(r!=col){ double f=M[r*n+col];
            for(int k=0;k<n;k++){ M[r*n+k]-=f*M[col*n+k]; inv[r*n+k]-=f*inv[col*n+k]; } }
    }
    return true;
}

/* Least-squares affine fit: T(3x3, MATLAB post-multiply convention) with
 * [x' y' 1] = [x y 1]*T, last column [0 0 1].  Solves the 3x2 block via
 * normal equations over the given index subset. */
bool cv_fit_affine(const matlab_mat *p1, const matlab_mat *p2,
                   const std::vector<int> &idx, std::vector<double> &T) {
    /* A = [x y 1] (m x 3); solve A*X = [x' y'] (m x 2). */
    double AtA[9]={0}, Atx[3]={0}, Aty[3]={0};
    for (int i : idx) {
        double a0=p1->data[i*2], a1=p1->data[i*2+1], a2=1.0;
        double bx=p2->data[i*2], by=p2->data[i*2+1];
        double row[3]={a0,a1,a2};
        for(int r=0;r<3;r++){ for(int c=0;c<3;c++) AtA[r*3+c]+=row[r]*row[c];
            Atx[r]+=row[r]*bx; Aty[r]+=row[r]*by; }
    }
    std::vector<double> inv;
    if (!cv_inv(std::vector<double>(AtA,AtA+9),3,inv)) return false;
    double cx[3]={0},cy[3]={0};
    for(int r=0;r<3;r++) for(int c=0;c<3;c++){ cx[r]+=inv[r*3+c]*Atx[c]; cy[r]+=inv[r*3+c]*Aty[c]; }
    T.assign(9,0.0);
    T[0]=cx[0]; T[3]=cx[1]; T[6]=cx[2];     /* column for x' */
    T[1]=cy[0]; T[4]=cy[1]; T[7]=cy[2];     /* column for y' */
    T[2]=0; T[5]=0; T[8]=1;
    return true;
}

double cv_affine_resid(const std::vector<double> &T, double x, double y,
                       double xp, double yp) {
    double u = x*T[0]+y*T[3]+T[6];
    double v = x*T[1]+y*T[4]+T[7];
    double dx=u-xp, dy=v-yp; return std::sqrt(dx*dx+dy*dy);
}

}  /* namespace */

extern "C" {

/* detectHarrisFeatures(I) -> K x 2 [x y] corner locations (strongest-first). */
matlab_mat *matlab_vision_harris(matlab_mat *Im) {
    if (!Im || !Im->data) return mat_alloc(0, 2);
    Img I(Im); int H = I.R, W = I.C;
    std::vector<double> Ix, Iy; cv_gradients(I, Ix, Iy);
    /* structure tensor, box-smoothed over a 3x3 window. */
    std::vector<double> R(H*W, 0.0);
    const double k = 0.04;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            double a=0,b=0,d=0;
            for (int dr=-1;dr<=1;dr++) for(int dc=-1;dc<=1;dc++){
                int rr=std::min(std::max(r+dr,0),H-1), cc=std::min(std::max(c+dc,0),W-1);
                double ix=Ix[rr*W+cc], iy=Iy[rr*W+cc];
                a+=ix*ix; b+=iy*iy; d+=ix*iy;
            }
            R[r*W+c] = (a*b - d*d) - k*(a+b)*(a+b);
        }
    return cv_points_mat(cv_corner_peaks(R, H, W, 0.01, 2));
}

/* detectMinEigenFeatures(I) -> Shi-Tomasi corners (min eigenvalue response). */
matlab_mat *matlab_vision_mineigen(matlab_mat *Im) {
    if (!Im || !Im->data) return mat_alloc(0, 2);
    Img I(Im); int H = I.R, W = I.C;
    std::vector<double> Ix, Iy; cv_gradients(I, Ix, Iy);
    std::vector<double> R(H*W, 0.0);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            double a=0,b=0,d=0;
            for (int dr=-1;dr<=1;dr++) for(int dc=-1;dc<=1;dc++){
                int rr=std::min(std::max(r+dr,0),H-1), cc=std::min(std::max(c+dc,0),W-1);
                double ix=Ix[rr*W+cc], iy=Iy[rr*W+cc];
                a+=ix*ix; b+=iy*iy; d+=ix*iy;
            }
            double tr=a+b, det=a*b-d*d;
            R[r*W+c] = tr/2.0 - std::sqrt(std::max(0.0, tr*tr/4.0 - det));  /* min eig */
        }
    return cv_points_mat(cv_corner_peaks(R, H, W, 0.01, 2));
}

/* detectFASTFeatures(I) -> FAST corners (contiguous-arc intensity test). */
matlab_mat *matlab_vision_fast(matlab_mat *Im) {
    if (!Im || !Im->data) return mat_alloc(0, 2);
    Img I(Im); int H = I.R, W = I.C;
    /* Bresenham radius-3 circle offsets (16 pixels). */
    static const int ox[16] = {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
    static const int oy[16] = {-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
    double thr = 0.10;  /* relative to [0,1]-ish intensity range */
    /* scale threshold to data range. */
    double mn=I.d[0],mx=I.d[0];
    for (int i=0;i<H*W;i++){ mn=std::min(mn,I.d[i]); mx=std::max(mx,I.d[i]); }
    thr = 0.10 * std::max(1.0, mx - mn);
    std::vector<double> R(H*W, 0.0);
    for (int r = 3; r < H-3; ++r)
        for (int c = 3; c < W-3; ++c) {
            double p = I.at(r,c); int bright=0, dark=0, run=0, best=0;
            for (int i=0;i<24;i++){          /* wrap to test contiguous arcs */
                double v = I.at(r+oy[i%16], c+ox[i%16]);
                if (v > p + thr) { run = (run>0? run+1: 1); }
                else if (v < p - thr) { run = (run<0? run-1: -1); }
                else run = 0;
                best = std::max(best, std::abs(run));
            }
            if (best >= 9) {
                double s=0; for(int i=0;i<16;i++) s+=std::fabs(I.at(r+oy[i],c+ox[i])-p);
                R[r*W+c]=s;
            }
        }
    return cv_points_mat(cv_corner_peaks(R, H, W, 0.0, 2));
}

/* extractFeatures(I, points) -> K x D normalized intensity-patch descriptors.
 * Each row is a (2*hw+1)^2 patch around the point, mean-subtracted +
 * L2-normalized (so NCC reduces to a dot product). */
matlab_mat *matlab_vision_extract(matlab_mat *Im, matlab_mat *pts) {
    if (!Im || !Im->data || !pts || !pts->data) return mat_alloc(0, 0);
    Img I(Im);
    int K = static_cast<int>(pts->rows);
    const int hw = 5, P = 2*hw+1, D = P*P;       /* 11x11 patch -> 121-D */
    matlab_mat *F = mat_alloc(K, D);
    if (!F || !F->data) return F;
    for (int kk = 0; kk < K; ++kk) {
        int x = static_cast<int>(pts->data[kk*2+0]) - 1;   /* col */
        int y = static_cast<int>(pts->data[kk*2+1]) - 1;   /* row */
        std::vector<double> patch(D); double mean = 0.0;
        int idx = 0;
        for (int dr=-hw; dr<=hw; ++dr) for (int dc=-hw; dc<=hw; ++dc) {
            double v = I.at(y+dr, x+dc); patch[idx++] = v; mean += v;
        }
        mean /= D;
        double nrm = 0.0;
        for (int i=0;i<D;i++){ patch[i] -= mean; nrm += patch[i]*patch[i]; }
        nrm = std::sqrt(nrm); if (nrm < 1e-9) nrm = 1.0;
        for (int i=0;i<D;i++) F->data[kk*D+i] = patch[i]/nrm;
    }
    return F;
}

/* matchFeatures(f1, f2) -> M x 2 [i j] 1-based index pairs (nearest-neighbour
 * by SSD with the Lowe ratio test + a unique-match check). */
matlab_mat *matlab_vision_match(matlab_mat *F1, matlab_mat *F2) {
    if (!F1 || !F2 || !F1->data || !F2->data || F1->cols != F2->cols)
        return mat_alloc(0, 2);
    int K1 = static_cast<int>(F1->rows), K2 = static_cast<int>(F2->rows);
    int D = static_cast<int>(F1->cols);
    const double ratio = 0.6;
    std::vector<std::pair<int,int>> matches;
    for (int i = 0; i < K1; ++i) {
        double d1 = 1e300, d2 = 1e300; int best = -1;
        for (int j = 0; j < K2; ++j) {
            double s = 0.0;
            for (int k = 0; k < D; ++k) { double e = F1->data[i*D+k]-F2->data[j*D+k]; s += e*e; }
            if (s < d1) { d2 = d1; d1 = s; best = j; }
            else if (s < d2) d2 = s;
        }
        if (best >= 0 && d1 < ratio*ratio*d2)
            matches.push_back({i+1, best+1});       /* 1-based */
    }
    matlab_mat *M = mat_alloc(static_cast<int64_t>(matches.size()), 2);
    if (M && M->data)
        for (size_t i=0;i<matches.size();++i){ M->data[i*2]=matches[i].first; M->data[i*2+1]=matches[i].second; }
    return M;
}

/* extractHOGFeatures(I) -> 1 x N HOG descriptor (8x8 cells, 2x2 blocks,
 * 9 unsigned-orientation bins, L2 block normalization). */
matlab_mat *matlab_vision_hog(matlab_mat *Im) {
    if (!Im || !Im->data) return mat_alloc(1, 0);
    Img I(Im); int H = I.R, W = I.C;
    std::vector<double> Ix, Iy; cv_gradients(I, Ix, Iy);
    const int cell = 8, nb = 9;
    int cH = H/cell, cW = W/cell;
    if (cH < 1 || cW < 1) return mat_alloc(1, 0);
    std::vector<double> hist(static_cast<size_t>(cH)*cW*nb, 0.0);
    for (int r=0;r<cH*cell;++r) for (int c=0;c<cW*cell;++c){
        double gx=Ix[r*W+c], gy=Iy[r*W+c];
        double mag=std::sqrt(gx*gx+gy*gy);
        double ang=std::atan2(gy,gx); if (ang<0) ang+=kPi;     /* unsigned 0..pi */
        int bin=static_cast<int>(ang/kPi*nb); if (bin>=nb) bin=nb-1;
        int ci=r/cell, cj=c/cell;
        hist[(ci*cW+cj)*nb+bin]+=mag;
    }
    /* 2x2 block L2 normalization. */
    std::vector<double> feat;
    for (int bi=0; bi<cH-1; ++bi) for (int bj=0; bj<cW-1; ++bj){
        std::vector<double> blk;
        for (int di=0;di<2;di++) for(int dj=0;dj<2;dj++)
            for(int b=0;b<nb;b++) blk.push_back(hist[((bi+di)*cW+(bj+dj))*nb+b]);
        double nrm=0; for(double v:blk) nrm+=v*v; nrm=std::sqrt(nrm+1e-6);
        for(double v:blk) feat.push_back(v/nrm);
    }
    return cv_mat(feat, 1, static_cast<int>(feat.size()));
}

/* extractLBPFeatures(I) -> 1 x 256 local-binary-pattern histogram (normalized). */
matlab_mat *matlab_vision_lbp(matlab_mat *Im) {
    if (!Im || !Im->data) return mat_alloc(1, 256);
    Img I(Im); int H = I.R, W = I.C;
    std::vector<double> hist(256, 0.0); double tot = 0.0;
    static const int dx[8]={-1,0,1,1,1,0,-1,-1}, dy[8]={-1,-1,-1,0,1,1,1,0};
    for (int r=1;r<H-1;++r) for (int c=1;c<W-1;++c){
        double p=I.at(r,c); int code=0;
        for (int n=0;n<8;n++) if (I.at(r+dy[n],c+dx[n])>=p) code|=(1<<n);
        hist[code]+=1.0; tot+=1.0;
    }
    if (tot>0) for (double &v:hist) v/=tot;
    return cv_mat(hist, 1, 256);
}

/* estgeotform2d(matchedPts1, matchedPts2[, type]) -> 3x3 transform T
 * (MATLAB post-multiply convention; feed to affine2d(T) then imwarp).
 * RANSAC over an affine fit with a deterministic internal RNG; returns the
 * consensus-refit transform. (similarity/rigid fall back to affine LS.) */
matlab_mat *matlab_vision_estgeotform(matlab_mat *p1, matlab_mat *p2, void *typeS) {
    (void)typeS;
    if (!p1 || !p2 || !p1->data || !p2->data) return cv_mat({1,0,0, 0,1,0, 0,0,1},3,3);
    int K = static_cast<int>(std::min(p1->rows, p2->rows));
    std::vector<double> best(9, 0.0); best[0]=best[4]=best[8]=1.0;
    if (K < 3) { (void)best; std::vector<int> all; for(int i=0;i<K;i++) all.push_back(i);
        std::vector<double> T; if (K>=1 && cv_fit_affine(p1,p2,all,T)) best=T;
        return cv_mat(best,3,3); }
    uint64_t rng = 0x9E3779B97F4A7C15ULL;       /* deterministic LCG */
    auto rnd = [&](int n){ rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
                           return static_cast<int>((rng>>33) % n); };
    const double thr = 1.5; int bestInliers = -1;
    for (int it = 0; it < 500; ++it) {
        std::vector<int> samp;
        while (static_cast<int>(samp.size()) < 3) {
            int r = rnd(K); bool dup=false; for(int s:samp) if(s==r) dup=true;
            if(!dup) samp.push_back(r);
        }
        std::vector<double> T;
        if (!cv_fit_affine(p1,p2,samp,T)) continue;
        std::vector<int> inl;
        for (int i=0;i<K;i++)
            if (cv_affine_resid(T,p1->data[i*2],p1->data[i*2+1],p2->data[i*2],p2->data[i*2+1])<thr)
                inl.push_back(i);
        if (static_cast<int>(inl.size()) > bestInliers) {
            bestInliers = static_cast<int>(inl.size());
            std::vector<double> Tr; if (cv_fit_affine(p1,p2,inl,Tr)) best=Tr; else best=T;
        }
    }
    return cv_mat(best,3,3);
}

/* 2-arg form estgeotform2d(p1, p2) — defaults to affine. */
matlab_mat *matlab_vision_estgeotform2(matlab_mat *p1, matlab_mat *p2) {
    return matlab_vision_estgeotform(p1, p2, nullptr);
}

/* estimateFundamentalMatrix(p1, p2) -> 3x3 F (normalized 8-point + rank-2). */
matlab_mat *matlab_vision_fundmatrix(matlab_mat *p1, matlab_mat *p2) {
    if (!p1 || !p2 || !p1->data || !p2->data) return cv_mat(std::vector<double>(9,0.0),3,3);
    int K = static_cast<int>(std::min(p1->rows, p2->rows));
    if (K < 8) return cv_mat(std::vector<double>(9,0.0),3,3);
    /* normalize each point set to mean 0 / mean-distance sqrt(2). */
    auto norm = [&](const matlab_mat *p, std::vector<double> &xn, std::vector<double> &yn,
                    double T[9]){
        double mx=0,my=0; for(int i=0;i<K;i++){ mx+=p->data[i*2]; my+=p->data[i*2+1]; }
        mx/=K; my/=K; double md=0;
        for(int i=0;i<K;i++){ double dx=p->data[i*2]-mx, dy=p->data[i*2+1]-my; md+=std::sqrt(dx*dx+dy*dy); }
        md/=K; double s = (md>1e-9)? std::sqrt(2.0)/md : 1.0;
        xn.resize(K); yn.resize(K);
        for(int i=0;i<K;i++){ xn[i]=s*(p->data[i*2]-mx); yn[i]=s*(p->data[i*2+1]-my); }
        T[0]=s; T[1]=0; T[2]=-s*mx; T[3]=0; T[4]=s; T[5]=-s*my; T[6]=0;T[7]=0;T[8]=1;
    };
    std::vector<double> x1,y1,x2,y2; double T1[9],T2[9];
    norm(p1,x1,y1,T1); norm(p2,x2,y2,T2);
    /* AtA (9x9) of the epipolar constraint rows. */
    std::vector<double> AtA(81,0.0);
    for (int i=0;i<K;i++){
        double r[9]={x2[i]*x1[i],x2[i]*y1[i],x2[i],y2[i]*x1[i],y2[i]*y1[i],y2[i],x1[i],y1[i],1};
        for(int a=0;a<9;a++) for(int b=0;b<9;b++) AtA[a*9+b]+=r[a]*r[b];
    }
    std::vector<double> V,w; cv_jacobi(AtA,9,V,w);
    int mn=0; for(int i=1;i<9;i++) if(w[i]<w[mn]) mn=i;
    double Fn[9]; for(int i=0;i<9;i++) Fn[i]=V[i*9+mn];
    /* denormalize: F = T2' * Fn * T1. */
    auto mul=[&](const double*A,const double*B,double*C){ for(int i=0;i<3;i++)for(int j=0;j<3;j++){double s=0;for(int k=0;k<3;k++)s+=A[i*3+k]*B[k*3+j];C[i*3+j]=s;}};
    double T2t[9]={T2[0],T2[3],T2[6],T2[1],T2[4],T2[7],T2[2],T2[5],T2[8]};
    double tmp[9],F[9]; mul(Fn,T1,tmp); mul(T2t,tmp,F);
    double nf=F[8]!=0?F[8]:1.0; for(int i=0;i<9;i++) F[i]/=nf;
    return cv_mat(std::vector<double>(F,F+9),3,3);
}

/* ===========================================================================
 * Tier-3 — bounding boxes + annotation
 * ==========================================================================*/

/* bboxOverlapRatio(A, B) -> Ka x Kb intersection-over-union matrix.
 * Boxes are [x y w h]. */
matlab_mat *matlab_vision_bboxiou(matlab_mat *A, matlab_mat *B) {
    if (!A || !B || !A->data || !B->data) return mat_alloc(0, 0);
    int Ka = static_cast<int>(A->rows), Kb = static_cast<int>(B->rows);
    matlab_mat *R = mat_alloc(Ka, Kb);
    if (!R || !R->data) return R;
    for (int i=0;i<Ka;i++){
        double ax=A->data[i*4],ay=A->data[i*4+1],aw=A->data[i*4+2],ah=A->data[i*4+3];
        for (int j=0;j<Kb;j++){
            double bx=B->data[j*4],by=B->data[j*4+1],bw=B->data[j*4+2],bh=B->data[j*4+3];
            double ix=std::max(ax,bx), iy=std::max(ay,by);
            double ix2=std::min(ax+aw,bx+bw), iy2=std::min(ay+ah,by+bh);
            double iw=std::max(0.0,ix2-ix), ihh=std::max(0.0,iy2-iy);
            double inter=iw*ihh, uni=aw*ah+bw*bh-inter;
            R->data[i*Kb+j] = uni>0 ? inter/uni : 0.0;
        }
    }
    return R;
}

/* selectStrongestBbox(bboxes, scores[, overlapThresh]) -> kept boxes (M x 4),
 * greedy non-max suppression by score. */
matlab_mat *matlab_vision_nms(matlab_mat *bb, matlab_mat *sc, double ovD) {
    if (!bb || !bb->data || !sc || !sc->data) return mat_alloc(0, 4);
    int K = static_cast<int>(bb->rows);
    double ov = (ovD>0 && ovD<=1)? ovD : 0.5;
    std::vector<int> order(K); for(int i=0;i<K;i++) order[i]=i;
    std::sort(order.begin(),order.end(),[&](int a,int b){return sc->data[a]>sc->data[b];});
    std::vector<int> kept; std::vector<bool> dead(K,false);
    for (int oi=0; oi<K; ++oi){ int i=order[oi]; if(dead[i]) continue; kept.push_back(i);
        double ax=bb->data[i*4],ay=bb->data[i*4+1],aw=bb->data[i*4+2],ah=bb->data[i*4+3];
        for (int oj=oi+1; oj<K; ++oj){ int j=order[oj]; if(dead[j]) continue;
            double bx=bb->data[j*4],by=bb->data[j*4+1],bw=bb->data[j*4+2],bh=bb->data[j*4+3];
            double ix=std::max(ax,bx),iy=std::max(ay,by),ix2=std::min(ax+aw,bx+bw),iy2=std::min(ay+ah,by+bh);
            double inter=std::max(0.0,ix2-ix)*std::max(0.0,iy2-iy), uni=aw*ah+bw*bh-inter;
            if (uni>0 && inter/uni > ov) dead[j]=true;
        }
    }
    matlab_mat *R = mat_alloc(static_cast<int64_t>(kept.size()), 4);
    if (R && R->data) for(size_t k=0;k<kept.size();++k) for(int c=0;c<4;c++) R->data[k*4+c]=bb->data[kept[k]*4+c];
    return R;
}

matlab_mat *matlab_vision_nms2(matlab_mat *bb, matlab_mat *sc) {
    return matlab_vision_nms(bb, sc, 0.5);
}

/* bbox2points(bbox) -> 4 x 2 corner points (single [x y w h] box). */
matlab_mat *matlab_vision_bbox2pts(matlab_mat *bb) {
    if (!bb || !bb->data) return mat_alloc(0, 2);
    double x=bb->data[0],y=bb->data[1],w=bb->data[2],h=bb->data[3];
    return cv_mat({x,y, x+w,y, x+w,y+h, x,y+h}, 4, 2);
}

/* annotation helpers: draw into a copy of the image at intensity `val`. */
matlab_mat *matlab_vision_insertshape(matlab_mat *Im, void *shapeS, matlab_mat *pos) {
    if (!Im || !Im->data) return mat_alloc(0, 0);
    int H=static_cast<int>(Im->rows), W=static_cast<int>(Im->cols);
    matlab_mat *J = mat_alloc(H, W);
    if (!J||!J->data) return J;
    for (int i=0;i<H*W;i++) J->data[i]=Im->data[i];
    double mx=0; for(int i=0;i<H*W;i++) mx=std::max(mx,Im->data[i]);
    double val = mx>0? mx : 1.0;
    auto setpx=[&](int r,int c){ if(r>=0&&r<H&&c>=0&&c<W) J->data[r*W+c]=val; };
    std::string shape = cv_sstr(shapeS);
    int K = static_cast<int>(pos->rows);
    for (int i=0;i<K;i++){
        if (shape=="rectangle"){
            int x=static_cast<int>(pos->data[i*pos->cols])-1, y=static_cast<int>(pos->data[i*pos->cols+1])-1;
            int w=static_cast<int>(pos->data[i*pos->cols+2]), h=static_cast<int>(pos->data[i*pos->cols+3]);
            for(int c=x;c<=x+w;c++){ setpx(y,c); setpx(y+h,c); }
            for(int r=y;r<=y+h;r++){ setpx(r,x); setpx(r,x+w); }
        } else if (shape=="line"){
            int x1=static_cast<int>(pos->data[i*pos->cols])-1, y1=static_cast<int>(pos->data[i*pos->cols+1])-1;
            int x2=static_cast<int>(pos->data[i*pos->cols+2])-1, y2=static_cast<int>(pos->data[i*pos->cols+3])-1;
            int steps=std::max(std::abs(x2-x1),std::abs(y2-y1)); if(steps<1) steps=1;
            for(int s=0;s<=steps;s++){ setpx(y1+(y2-y1)*s/steps, x1+(x2-x1)*s/steps); }
        } else if (shape=="circle"){
            int cx=static_cast<int>(pos->data[i*pos->cols])-1, cy=static_cast<int>(pos->data[i*pos->cols+1])-1;
            int rad=static_cast<int>(pos->data[i*pos->cols+2]);
            for(int a=0;a<360;a+=4){ double t=a*kPi/180.0; setpx(cy+static_cast<int>(rad*std::sin(t)), cx+static_cast<int>(rad*std::cos(t))); }
        }
    }
    return J;
}

matlab_mat *matlab_vision_insertmarker(matlab_mat *Im, matlab_mat *pts) {
    if (!Im || !Im->data) return mat_alloc(0, 0);
    int H=static_cast<int>(Im->rows), W=static_cast<int>(Im->cols);
    matlab_mat *J = mat_alloc(H, W);
    if(!J||!J->data) return J;
    for(int i=0;i<H*W;i++) J->data[i]=Im->data[i];
    double mx=0; for(int i=0;i<H*W;i++) mx=std::max(mx,Im->data[i]);
    double val=mx>0?mx:1.0;
    int K=static_cast<int>(pts->rows);
    for(int i=0;i<K;i++){ int x=static_cast<int>(pts->data[i*2])-1, y=static_cast<int>(pts->data[i*2+1])-1;
        for(int d=-2;d<=2;d++){ if(y>=0&&y<H&&x+d>=0&&x+d<W) J->data[y*W+x+d]=val;
                                if(x>=0&&x<W&&y+d>=0&&y+d<H) J->data[(y+d)*W+x]=val; } }
    return J;
}

/* ===========================================================================
 * Tier-4 — optical flow (returns [Vx; Vy] stacked 2M x N)
 * ==========================================================================*/

matlab_mat *matlab_vision_oflk(matlab_mat *Ia, matlab_mat *Ib) {
    if (!Ia||!Ib||!Ia->data||!Ib->data) return mat_alloc(0,0);
    Img A(Ia), B(Ib); int H=A.R, W=A.C;
    std::vector<double> Ax,Ay; cv_gradients(A,Ax,Ay);
    matlab_mat *F = mat_alloc(2*H, W);
    if(!F||!F->data) return F;
    const int win=3;
    for(int r=0;r<H;r++) for(int c=0;c<W;c++){
        double sxx=0,syy=0,sxy=0,sxt=0,syt=0;
        for(int dr=-win;dr<=win;dr++) for(int dc=-win;dc<=win;dc++){
            int rr=std::min(std::max(r+dr,0),H-1), cc=std::min(std::max(c+dc,0),W-1);
            double ix=Ax[rr*W+cc], iy=Ay[rr*W+cc], it=B.at(rr,cc)-A.at(rr,cc);
            sxx+=ix*ix; syy+=iy*iy; sxy+=ix*iy; sxt+=ix*it; syt+=iy*it;
        }
        double det=sxx*syy-sxy*sxy, u=0,v=0;
        if(std::fabs(det)>1e-6){ u=-( syy*sxt - sxy*syt)/det; v=-(-sxy*sxt + sxx*syt)/det; }
        F->data[r*W+c]=u; F->data[(H+r)*W+c]=v;
    }
    return F;
}

matlab_mat *matlab_vision_ofhs(matlab_mat *Ia, matlab_mat *Ib) {
    if (!Ia||!Ib||!Ia->data||!Ib->data) return mat_alloc(0,0);
    Img A(Ia), B(Ib); int H=A.R, W=A.C;
    std::vector<double> Ax,Ay; cv_gradients(A,Ax,Ay);
    std::vector<double> u(H*W,0.0), v(H*W,0.0);
    const double alpha=1.0; const int iters=50;
    for(int it=0;it<iters;it++){
        std::vector<double> un=u, vn=v;
        for(int r=0;r<H;r++) for(int c=0;c<W;c++){
            auto avg=[&](const std::vector<double>&q){
                double s=0; int n=0;
                int rs[4]={r-1,r+1,r,r}, cs[4]={c,c,c-1,c+1};
                for(int k=0;k<4;k++){int rr=rs[k],cc=cs[k]; if(rr>=0&&rr<H&&cc>=0&&cc<W){s+=q[rr*W+cc];n++;}}
                return n? s/n : 0.0; };
            double ub=avg(u), vb=avg(v);
            double ix=Ax[r*W+c], iy=Ay[r*W+c], itt=B.at(r,c)-A.at(r,c);
            double den=alpha*alpha+ix*ix+iy*iy;
            double t=(ix*ub+iy*vb+itt)/den;
            un[r*W+c]=ub-ix*t; vn[r*W+c]=vb-iy*t;
        }
        u.swap(un); v.swap(vn);
    }
    matlab_mat *F=mat_alloc(2*H,W);
    if(F&&F->data){ for(int i=0;i<H*W;i++){F->data[i]=u[i]; F->data[H*W+i]=v[i];} }
    return F;
}

/* ===========================================================================
 * Tier-5 — camera geometry + stereo
 * ==========================================================================*/

/* triangulate(pts1, pts2, camMtx1, camMtx2) -> N x 3 world points (linear
 * DLT).  Camera matrices are 4 x 3 (MATLAB convention: [X Y Z 1]*camMtx =
 * [x*w y*w w]); image points are N x 2. */
matlab_mat *matlab_vision_triangulate(matlab_mat *p1, matlab_mat *p2,
                                      matlab_mat *C1, matlab_mat *C2) {
    if (!p1||!p2||!C1||!C2||!p1->data||!p2->data||!C1->data||!C2->data) return mat_alloc(0,3);
    int N = static_cast<int>(std::min(p1->rows, p2->rows));
    /* column k of the 4x3 camera matrix: C[r*3+k], r=0..3. */
    auto col=[&](const matlab_mat *C,int k,double v[4]){ for(int r=0;r<4;r++) v[r]=C->data[r*3+k]; };
    double c1x[4],c1y[4],c1w[4],c2x[4],c2y[4],c2w[4];
    col(C1,0,c1x); col(C1,1,c1y); col(C1,2,c1w);
    col(C2,0,c2x); col(C2,1,c2y); col(C2,2,c2w);
    matlab_mat *R = mat_alloc(N, 3);
    if(!R||!R->data) return R;
    for (int i=0;i<N;i++){
        double x1=p1->data[i*2],y1=p1->data[i*2+1],x2=p2->data[i*2],y2=p2->data[i*2+1];
        double A[16];
        for(int k=0;k<4;k++){
            A[0*4+k]=x1*c1w[k]-c1x[k];
            A[1*4+k]=y1*c1w[k]-c1y[k];
            A[2*4+k]=x2*c2w[k]-c2x[k];
            A[3*4+k]=y2*c2w[k]-c2y[k];
        }
        /* smallest right singular vector of A = eigvec of A'A (4x4). */
        std::vector<double> AtA(16,0.0);
        for(int a=0;a<4;a++) for(int b=0;b<4;b++){ double s=0; for(int r=0;r<4;r++) s+=A[r*4+a]*A[r*4+b]; AtA[a*4+b]=s; }
        std::vector<double> V,w; cv_jacobi(AtA,4,V,w);
        int mn=0; for(int k=1;k<4;k++) if(w[k]<w[mn]) mn=k;
        double X=V[0*4+mn],Y=V[1*4+mn],Z=V[2*4+mn],Wt=V[3*4+mn];
        if(std::fabs(Wt)<1e-12) Wt=1e-12;
        R->data[i*3+0]=X/Wt; R->data[i*3+1]=Y/Wt; R->data[i*3+2]=Z/Wt;
    }
    return R;
}

/* disparityBM(IL, IR[, maxDisparity]) -> M x N disparity map (block matching,
 * left-to-right horizontal search by SSD). */
matlab_mat *matlab_vision_disparity(matlab_mat *Lm, matlab_mat *Rm, double maxD) {
    if(!Lm||!Rm||!Lm->data||!Rm->data) return mat_alloc(0,0);
    Img L(Lm), Rr(Rm); int H=L.R, W=L.C;
    int maxDisp = (maxD>0)? static_cast<int>(maxD) : 16;
    const int win=3;
    matlab_mat *D=mat_alloc(H,W);
    if(!D||!D->data) return D;
    for(int r=0;r<H;r++) for(int c=0;c<W;c++){
        double best=1e300; int bestd=0;
        for(int d=0; d<=maxDisp && c-d>=0; ++d){
            double ssd=0;
            for(int dr=-win;dr<=win;dr++) for(int dc=-win;dc<=win;dc++){
                double e=L.at(r+dr,c+dc)-Rr.at(r+dr,c-d+dc); ssd+=e*e;
            }
            if(ssd<best){ best=ssd; bestd=d; }
        }
        D->data[r*W+c]=bestd;
    }
    return D;
}

matlab_mat *matlab_vision_disparity2(matlab_mat *Lm, matlab_mat *Rm) {
    return matlab_vision_disparity(Lm, Rm, 16);
}

/* ===========================================================================
 * Tier-6 — point cloud (N x 3 matrices)
 * ==========================================================================*/

/* pcwrite(filename, pts) — write an ASCII PLY point cloud. */
matlab_mat *matlab_vision_pcwrite(void *path_s, matlab_mat *pts) {
    std::string path = cv_sstr(path_s);
    /* cv_sstr lowercases — re-read raw for the path. */
    const cv_string_s *ps = reinterpret_cast<const cv_string_s *>(path_s);
    std::string raw = (ps && ps->data && ps->len>0) ? std::string(ps->data, ps->data+ps->len) : path;
    if (!pts || !pts->data) return cv_scalar(0);
    int N = static_cast<int>(pts->rows);
    FILE *f = std::fopen(raw.c_str(), "w");
    if (!f) return cv_scalar(0);
    std::fprintf(f, "ply\nformat ascii 1.0\nelement vertex %d\n", N);
    std::fprintf(f, "property float x\nproperty float y\nproperty float z\nend_header\n");
    for (int i=0;i<N;i++)
        std::fprintf(f, "%g %g %g\n", pts->data[i*3], pts->data[i*3+1], pts->data[i*3+2]);
    std::fclose(f);
    return cv_scalar(1);
}

/* pcread(filename) -> N x 3 point matrix (ASCII PLY). */
matlab_mat *matlab_vision_pcread(void *path_s) {
    const cv_string_s *ps = reinterpret_cast<const cv_string_s *>(path_s);
    std::string raw = (ps && ps->data && ps->len>0) ? std::string(ps->data, ps->data+ps->len) : std::string();
    FILE *f = std::fopen(raw.c_str(), "r");
    if (!f) return mat_alloc(0,3);
    char line[512]; int N=0; bool inData=false;
    std::vector<double> pts;
    while (std::fgets(line, sizeof(line), f)) {
        if (!inData) {
            if (std::strncmp(line,"element vertex",14)==0) N=std::atoi(line+14);
            if (std::strncmp(line,"end_header",10)==0) inData=true;
        } else {
            double x,y,z; if (std::sscanf(line,"%lf %lf %lf",&x,&y,&z)==3){ pts.push_back(x);pts.push_back(y);pts.push_back(z); }
        }
    }
    std::fclose(f);
    int M = static_cast<int>(pts.size()/3);
    matlab_mat *R = mat_alloc(M, 3);
    if (R && R->data) for (int i=0;i<M*3;i++) R->data[i]=pts[i];
    return R;
}

/* pcdownsample(pts, gridSize) -> voxel-grid-averaged point cloud. */
matlab_mat *matlab_vision_pcdownsample(matlab_mat *pts, double grid) {
    if (!pts || !pts->data || grid<=0) return mat_alloc(0,3);
    int N = static_cast<int>(pts->rows);
    /* hash voxel index -> accumulate. */
    std::vector<long long> keys; std::vector<std::vector<double>> acc;
    auto findk=[&](long long k)->int{ for(size_t i=0;i<keys.size();i++) if(keys[i]==k) return static_cast<int>(i); return -1; };
    for (int i=0;i<N;i++){
        long long ix=static_cast<long long>(std::floor(pts->data[i*3]/grid));
        long long iy=static_cast<long long>(std::floor(pts->data[i*3+1]/grid));
        long long iz=static_cast<long long>(std::floor(pts->data[i*3+2]/grid));
        long long key=(ix*73856093LL)^(iy*19349663LL)^(iz*83492791LL);
        int idx=findk(key);
        if(idx<0){ keys.push_back(key); acc.push_back({pts->data[i*3],pts->data[i*3+1],pts->data[i*3+2],1}); }
        else { acc[idx][0]+=pts->data[i*3]; acc[idx][1]+=pts->data[i*3+1]; acc[idx][2]+=pts->data[i*3+2]; acc[idx][3]+=1; }
    }
    matlab_mat *R = mat_alloc(static_cast<int64_t>(acc.size()), 3);
    if (R && R->data) for(size_t i=0;i<acc.size();i++){ double n=acc[i][3]; R->data[i*3]=acc[i][0]/n; R->data[i*3+1]=acc[i][1]/n; R->data[i*3+2]=acc[i][2]/n; }
    return R;
}

/* pcfitplane(pts, threshold) -> 1 x 4 plane [a b c d] (unit normal), RANSAC. */
matlab_mat *matlab_vision_pcfitplane(matlab_mat *pts, double thr) {
    if (!pts || !pts->data) return cv_mat({0,0,1,0},1,4);
    int N = static_cast<int>(pts->rows);
    if (N < 3) return cv_mat({0,0,1,0},1,4);
    if (thr<=0) thr=0.05;
    uint64_t rng=0xDEADBEEF12345678ULL;
    auto rnd=[&](int n){ rng=rng*6364136223846793005ULL+1442695040888963407ULL; return static_cast<int>((rng>>33)%n); };
    double bestPlane[4]={0,0,1,0}; int bestInl=-1;
    for (int it=0;it<300;it++){
        int a=rnd(N),b=rnd(N),c=rnd(N); if(a==b||b==c||a==c) continue;
        double ax=pts->data[a*3],ay=pts->data[a*3+1],az=pts->data[a*3+2];
        double v1x=pts->data[b*3]-ax,v1y=pts->data[b*3+1]-ay,v1z=pts->data[b*3+2]-az;
        double v2x=pts->data[c*3]-ax,v2y=pts->data[c*3+1]-ay,v2z=pts->data[c*3+2]-az;
        double nx=v1y*v2z-v1z*v2y, ny=v1z*v2x-v1x*v2z, nz=v1x*v2y-v1y*v2x;
        double nn=std::sqrt(nx*nx+ny*ny+nz*nz); if(nn<1e-9) continue;
        nx/=nn; ny/=nn; nz/=nn; double d=-(nx*ax+ny*ay+nz*az);
        int inl=0; for(int i=0;i<N;i++){ double dist=std::fabs(nx*pts->data[i*3]+ny*pts->data[i*3+1]+nz*pts->data[i*3+2]+d); if(dist<thr) inl++; }
        if(inl>bestInl){ bestInl=inl; bestPlane[0]=nx;bestPlane[1]=ny;bestPlane[2]=nz;bestPlane[3]=d; }
    }
    return cv_mat({bestPlane[0],bestPlane[1],bestPlane[2],bestPlane[3]},1,4);
}

/* pcregistericp(moving, fixed) -> 4 x 4 rigid transform aligning moving->fixed.
 * Point-to-point ICP with nearest-neighbour + Kabsch (SVD via 3x3 Jacobi). */
matlab_mat *matlab_vision_pcicp(matlab_mat *mov, matlab_mat *fix) {
    if (!mov||!fix||!mov->data||!fix->data) { std::vector<double> I(16,0); for(int i=0;i<4;i++) I[i*4+i]=1; return cv_mat(I,4,4); }
    int Nm=static_cast<int>(mov->rows), Nf=static_cast<int>(fix->rows);
    /* accumulated transform. */
    double Rt[9]={1,0,0,0,1,0,0,0,1}, tt[3]={0,0,0};
    std::vector<double> P(Nm*3); for(int i=0;i<Nm*3;i++) P[i]=mov->data[i];
    for (int iter=0; iter<20; ++iter){
        /* nearest neighbours fixed for each current moving point. */
        std::vector<int> nn(Nm);
        for(int i=0;i<Nm;i++){ double best=1e300;int bj=0;
            for(int j=0;j<Nf;j++){ double dx=P[i*3]-fix->data[j*3],dy=P[i*3+1]-fix->data[j*3+1],dz=P[i*3+2]-fix->data[j*3+2];
                double dd=dx*dx+dy*dy+dz*dz; if(dd<best){best=dd;bj=j;} } nn[i]=bj; }
        double mc[3]={0,0,0}, fc[3]={0,0,0};
        for(int i=0;i<Nm;i++){ for(int k=0;k<3;k++){ mc[k]+=P[i*3+k]; fc[k]+=fix->data[nn[i]*3+k]; } }
        for(int k=0;k<3;k++){ mc[k]/=Nm; fc[k]/=Nm; }
        /* cross-covariance Hc (3x3). */
        std::vector<double> Hc(9,0.0);
        for(int i=0;i<Nm;i++){ double a[3]={P[i*3]-mc[0],P[i*3+1]-mc[1],P[i*3+2]-mc[2]};
            double b[3]={fix->data[nn[i]*3]-fc[0],fix->data[nn[i]*3+1]-fc[1],fix->data[nn[i]*3+2]-fc[2]};
            for(int r=0;r<3;r++) for(int c=0;c<3;c++) Hc[r*3+c]+=a[r]*b[c]; }
        /* R = V*U' via eig of H'H and HH' (approx SVD). Use polar-ish: for small
         * problems, R = (H'H)^{-1/2} H'... simpler: closed-form via Jacobi on
         * the symmetric 6x6 is heavy; use the analytic 3x3 SVD through HtH. */
        std::vector<double> HtH(9,0.0);
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){double s=0;for(int k=0;k<3;k++)s+=Hc[k*3+r]*Hc[k*3+c];HtH[r*3+c]=s;}
        std::vector<double> Vv,w; cv_jacobi(HtH,3,Vv,w);
        /* singular values + V columns; U = H*V*S^-1. */
        double Rstep[9]={1,0,0,0,1,0,0,0,1};
        double sv[3]={std::sqrt(std::max(0.0,w[0])),std::sqrt(std::max(0.0,w[1])),std::sqrt(std::max(0.0,w[2]))};
        if(sv[0]>1e-9&&sv[1]>1e-9&&sv[2]>1e-9){
            double U[9];
            for(int c=0;c<3;c++){ for(int r=0;r<3;r++){ double s=0; for(int k=0;k<3;k++) s+=Hc[r*3+k]*Vv[k*3+c]; U[r*3+c]=s/sv[c]; } }
            /* R = U * V' */
            for(int r=0;r<3;r++)for(int c=0;c<3;c++){double s=0;for(int k=0;k<3;k++)s+=U[r*3+k]*Vv[c*3+k];Rstep[r*3+c]=s;}
        }
        double tstep[3]; for(int k=0;k<3;k++){ tstep[k]=fc[k]-(Rstep[k*3]*mc[0]+Rstep[k*3+1]*mc[1]+Rstep[k*3+2]*mc[2]); }
        /* apply step to P. */
        for(int i=0;i<Nm;i++){ double x=P[i*3],y=P[i*3+1],z=P[i*3+2];
            P[i*3]=Rstep[0]*x+Rstep[1]*y+Rstep[2]*z+tstep[0];
            P[i*3+1]=Rstep[3]*x+Rstep[4]*y+Rstep[5]*z+tstep[1];
            P[i*3+2]=Rstep[6]*x+Rstep[7]*y+Rstep[8]*z+tstep[2]; }
        /* compose: Rt = Rstep*Rt, tt = Rstep*tt + tstep. */
        double Rn[9]; for(int r=0;r<3;r++)for(int c=0;c<3;c++){double s=0;for(int k=0;k<3;k++)s+=Rstep[r*3+k]*Rt[k*3+c];Rn[r*3+c]=s;}
        double tn[3]; for(int k=0;k<3;k++) tn[k]=Rstep[k*3]*tt[0]+Rstep[k*3+1]*tt[1]+Rstep[k*3+2]*tt[2]+tstep[k];
        for(int i=0;i<9;i++) Rt[i]=Rn[i]; for(int k=0;k<3;k++) tt[k]=tn[k];
    }
    std::vector<double> Tm(16,0.0);
    for(int r=0;r<3;r++){ for(int c=0;c<3;c++) Tm[r*4+c]=Rt[r*3+c]; Tm[r*4+3]=tt[r]; }
    Tm[15]=1;
    return cv_mat(Tm,4,4);
}

}  /* extern "C" */
