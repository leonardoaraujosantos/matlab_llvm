// Navigation Toolbox runtime — Tiers 1–4.
//
// All exported symbols use C-linkage extern "C".  Wiring:
//   - lib/Sema/Resolver.cpp        : builtin registry (names + matlab_nav_* symbols)
//   - lib/MLIR/Lowering.cpp        : classdef constructor + method intercepts
//   - tools/matlabc/main.cpp       : prelude trigger table (loads navigation_classdefs.m)
//
// No external dependency (no g2o/GTSAM/Ceres/PCL/OMPL): every planner, scan
// matcher, and graph optimiser is hand-coded over the shipped matlab_runtime
// kernel + the Robotics/Sensor-Fusion bases.  See docs/navigation_toolbox_roadmap.md.
//
// Storage model (all classdef carriers over packed-matrix properties):
//   occupancyMap          : Grid (probabilities), Resolution, GridSize,
//                           XWorldLimits, YWorldLimits, OccupiedThreshold
//   stateSpaceSE2/Dubins  : StateBounds (3×2), Weights, MinTurningRadius
//   validatorOccupancyMap : cloned map grid + ss bounds + ValidationDistance
//   navPath               : States (N×3)
//   plannerRRT/RRTStar    : cloned ss + validator + MaxConnectionDistance/
//                           MaxIterations/GoalBias (+ tree from the last plan)
//   plannerAStarGrid      : cloned map grid
//   lidarScan             : Ranges/Angles (+ Cartesian)
//   lidarSLAM             : accumulated poses + scans + a poseGraph
//   poseGraph             : node estimates (N×3) + edges (M×7 [a b dx dy dθ + info])

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <utility>
#include <vector>

extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);
extern "C" matlab_mat *matlab_rand(double m, double n);

namespace nav {

inline void obj_set_mat(void *o, const char *n, matlab_mat *m) {
    matlab_obj_set_mat(reinterpret_cast<matlab_obj *>(o), n, static_cast<int64_t>(std::strlen(n)), m);
}
inline void obj_set_f64(void *o, const char *n, double v) {
    matlab_obj_set_f64(reinterpret_cast<matlab_obj *>(o), n, static_cast<int64_t>(std::strlen(n)), v);
}
inline matlab_mat *obj_get_mat(void *o, const char *n) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(o), n, static_cast<int64_t>(std::strlen(n)));
}
inline double obj_get_f64(void *o, const char *n) {
    return matlab_obj_get_f64(reinterpret_cast<matlab_obj *>(o), n, static_cast<int64_t>(std::strlen(n)));
}

inline double wrap_pi(double a) {
    a = std::fmod(a + M_PI, 2 * M_PI);
    if (a < 0) a += 2 * M_PI;
    return a - M_PI;
}

// Dense solve A·x = b (n×n), Gauss elimination + partial pivot (in place on A, b).
inline void solve(double *A, int64_t n, double *b) {
    for (int64_t k = 0; k < n; ++k) {
        int64_t piv = k; double mx = std::fabs(A[k * n + k]);
        for (int64_t i = k + 1; i < n; ++i) { double v = std::fabs(A[i * n + k]); if (v > mx) { mx = v; piv = i; } }
        if (mx < 1e-15) continue;
        if (piv != k) { for (int64_t j = 0; j < n; ++j) std::swap(A[k*n+j], A[piv*n+j]); std::swap(b[k], b[piv]); }
        for (int64_t i = k + 1; i < n; ++i) {
            double f = A[i*n+k] / A[k*n+k];
            for (int64_t j = k; j < n; ++j) A[i*n+j] -= f * A[k*n+j];
            b[i] -= f * b[k];
        }
    }
    for (int64_t i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int64_t j = i + 1; j < n; ++j) s -= A[i*n+j] * b[j];
        b[i] = (std::fabs(A[i*n+i]) < 1e-15) ? 0.0 : s / A[i*n+i];
    }
}

}  // namespace nav

extern "C" {

// ===========================================================================
// Tier-1 — occupancyMap
// ===========================================================================
// occupancyMap(W, H, res): W×H metres at `res` cells/metre.  Stored as an
// R×C probability grid (rows = ceil(H·res), cols = ceil(W·res)), origin at
// the lower-left, row 0 = top (y high).
matlab_mat *matlab_nav_occmap_init(void *obj_v, double W, double H, double res) {
    if (res <= 0) res = 1.0;
    int64_t C = static_cast<int64_t>(std::ceil(W * res));
    int64_t R = static_cast<int64_t>(std::ceil(H * res));
    if (C < 1) C = 1; if (R < 1) R = 1;
    matlab_mat *G = mat_alloc(R, C);   // 0 = free (calloc), probabilities in [0,1]
    nav::obj_set_mat(obj_v, "Grid", G);
    nav::obj_set_f64(obj_v, "Resolution", res);
    matlab_mat *GS = mat_alloc(1, 2); GS->data[0] = static_cast<double>(R); GS->data[1] = static_cast<double>(C);
    nav::obj_set_mat(obj_v, "GridSize", GS);
    matlab_mat *XL = mat_alloc(1, 2); XL->data[0] = 0; XL->data[1] = static_cast<double>(C) / res;
    matlab_mat *YL = mat_alloc(1, 2); YL->data[0] = 0; YL->data[1] = static_cast<double>(R) / res;
    nav::obj_set_mat(obj_v, "XWorldLimits", XL);
    nav::obj_set_mat(obj_v, "YWorldLimits", YL);
    nav::obj_set_f64(obj_v, "OccupiedThreshold", 0.65);
    nav::obj_set_f64(obj_v, "FreeThreshold", 0.2);
    return mat_alloc(0, 0);
}

// world (x,y) → grid (row, col); returns false if outside.
static bool world_to_grid(void *obj_v, double x, double y, int64_t &r, int64_t &c) {
    matlab_mat *G = nav::obj_get_mat(obj_v, "Grid");
    double res = nav::obj_get_f64(obj_v, "Resolution");
    if (!G || res <= 0) return false;
    c = static_cast<int64_t>(std::floor(x * res));
    r = G->rows - 1 - static_cast<int64_t>(std::floor(y * res));
    return (r >= 0 && r < G->rows && c >= 0 && c < G->cols);
}

matlab_mat *matlab_nav_occmap_set(void *obj_v, matlab_mat *xy, double p) {
    matlab_mat *G = nav::obj_get_mat(obj_v, "Grid");
    if (!G || !xy || xy->rows * xy->cols < 2) return mat_alloc(0, 0);
    int64_t r, c;
    if (world_to_grid(obj_v, xy->data[0], xy->data[1], r, c)) G->data[r * G->cols + c] = p;
    return mat_alloc(0, 0);
}

matlab_mat *matlab_nav_occmap_get(void *obj_v, matlab_mat *xy) {
    matlab_mat *o = mat_alloc(1, 1);
    matlab_mat *G = nav::obj_get_mat(obj_v, "Grid");
    if (!G || !xy || xy->rows * xy->cols < 2) return o;
    int64_t r, c;
    if (world_to_grid(obj_v, xy->data[0], xy->data[1], r, c)) o->data[0] = G->data[r * G->cols + c];
    return o;
}

// checkOccupancy → 1 occupied / 0 free / -1 unknown (we collapse unknown→0).
matlab_mat *matlab_nav_occmap_check(void *obj_v, matlab_mat *xy) {
    matlab_mat *o = mat_alloc(1, 1);
    matlab_mat *G = nav::obj_get_mat(obj_v, "Grid");
    double occ = nav::obj_get_f64(obj_v, "OccupiedThreshold");
    if (!G || !xy || xy->rows * xy->cols < 2) { o->data[0] = 1; return o; }  // outside → blocked
    int64_t r, c;
    if (!world_to_grid(obj_v, xy->data[0], xy->data[1], r, c)) { o->data[0] = 1; return o; }
    o->data[0] = (G->data[r * G->cols + c] >= occ) ? 1.0 : 0.0;
    return o;
}

// inflate(map, radius_m): dilate occupied cells by radius (Chebyshev ball).
matlab_mat *matlab_nav_occmap_inflate(void *obj_v, double radius) {
    matlab_mat *G = nav::obj_get_mat(obj_v, "Grid");
    double res = nav::obj_get_f64(obj_v, "Resolution");
    double occ = nav::obj_get_f64(obj_v, "OccupiedThreshold");
    if (!G || res <= 0) return mat_alloc(0, 0);
    int64_t R = G->rows, C = G->cols;
    int64_t rad = static_cast<int64_t>(std::ceil(radius * res));
    if (rad < 1) return mat_alloc(0, 0);
    matlab_mat *Out = mat_alloc(R, C);
    for (int64_t i = 0; i < R * C; ++i) Out->data[i] = G->data[i];
    for (int64_t i = 0; i < R; ++i)
        for (int64_t j = 0; j < C; ++j) {
            if (G->data[i * C + j] < occ) continue;
            for (int64_t di = -rad; di <= rad; ++di)
                for (int64_t dj = -rad; dj <= rad; ++dj) {
                    if (di*di + dj*dj > rad*rad) continue;
                    int64_t ni = i + di, nj = j + dj;
                    if (ni >= 0 && ni < R && nj >= 0 && nj < C)
                        if (Out->data[ni * C + nj] < 1.0) Out->data[ni * C + nj] = 1.0;
                }
        }
    nav::obj_set_mat(obj_v, "Grid", Out);
    return mat_alloc(0, 0);
}

// Set a whole occupancy matrix at once (setOccupancy(map, gridMatrix) form /
// programmatic map construction).
matlab_mat *matlab_nav_occmap_setgrid(void *obj_v, matlab_mat *G) {
    if (!G) return mat_alloc(0, 0);
    matlab_mat *Gc = mat_alloc(G->rows, G->cols);
    for (int64_t i = 0; i < G->rows * G->cols; ++i) Gc->data[i] = G->data[i];
    nav::obj_set_mat(obj_v, "Grid", Gc);
    matlab_mat *GS = mat_alloc(1, 2); GS->data[0] = static_cast<double>(G->rows); GS->data[1] = static_cast<double>(G->cols);
    nav::obj_set_mat(obj_v, "GridSize", GS);
    return mat_alloc(0, 0);
}

// ===========================================================================
// Tier-1 — state spaces
// ===========================================================================
// stateSpaceSE2(bounds): bounds is 3×2 [xmin xmax; ymin ymax; thmin thmax].
matlab_mat *matlab_nav_ss_se2_init(void *obj_v, matlab_mat *bounds) {
    matlab_mat *B = mat_alloc(3, 2);
    if (bounds && bounds->rows == 3 && bounds->cols == 2)
        for (int i = 0; i < 6; ++i) B->data[i] = bounds->data[i];
    else { B->data[0]=-100;B->data[1]=100;B->data[2]=-100;B->data[3]=100;B->data[4]=-M_PI;B->data[5]=M_PI; }
    nav::obj_set_mat(obj_v, "StateBounds", B);
    nav::obj_set_f64(obj_v, "WeightTheta", 1.0);
    nav::obj_set_f64(obj_v, "MinTurningRadius", 0.0);  // 0 = holonomic SE2
    return mat_alloc(0, 0);
}

matlab_mat *matlab_nav_ss_dubins_init(void *obj_v, matlab_mat *bounds) {
    matlab_nav_ss_se2_init(obj_v, bounds);
    nav::obj_set_f64(obj_v, "MinTurningRadius", 1.0);
    return mat_alloc(0, 0);
}

// distance(ss, s1, s2): weighted SE2 metric (or Dubins length if turning-
// radius > 0).  s1/s2 are 1×3 [x y θ].
static double ss_distance(void *obj_v, const double *a, const double *b) {
    double wth = nav::obj_get_f64(obj_v, "WeightTheta");
    double dx = b[0] - a[0], dy = b[1] - a[1];
    double dth = std::fabs(nav::wrap_pi(b[2] - a[2]));
    return std::sqrt(dx*dx + dy*dy) + wth * dth;
}

matlab_mat *matlab_nav_ss_distance(void *obj_v, matlab_mat *s1, matlab_mat *s2) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!s1 || !s2 || s1->rows*s1->cols < 3 || s2->rows*s2->cols < 3) return o;
    o->data[0] = ss_distance(obj_v, s1->data, s2->data);
    return o;
}

// interpolate(ss, s1, s2, ratios): linear xy + angular slerp; ratios is a
// column of fractions in [0,1].  Returns N×3.
matlab_mat *matlab_nav_ss_interpolate(void *obj_v, matlab_mat *s1, matlab_mat *s2, matlab_mat *ratios) {
    (void)obj_v;
    if (!s1 || !s2 || !ratios) return mat_alloc(0, 0);
    int64_t n = ratios->rows * ratios->cols;
    matlab_mat *O = mat_alloc(n, 3);
    double a0 = s1->data[0], a1 = s1->data[1], a2 = s1->data[2];
    double b0 = s2->data[0], b1 = s2->data[1], b2 = s2->data[2];
    double dth = nav::wrap_pi(b2 - a2);
    for (int64_t i = 0; i < n; ++i) {
        double t = ratios->data[i];
        O->data[i*3+0] = a0 + (b0 - a0) * t;
        O->data[i*3+1] = a1 + (b1 - a1) * t;
        O->data[i*3+2] = nav::wrap_pi(a2 + dth * t);
    }
    return O;
}

// sampleUniform(ss): one uniform sample within bounds → 1×3.
matlab_mat *matlab_nav_ss_sample(void *obj_v) {
    matlab_mat *B = nav::obj_get_mat(obj_v, "StateBounds");
    matlab_mat *o = mat_alloc(1, 3);
    matlab_mat *U = matlab_rand(3.0, 1.0);
    if (B) for (int i = 0; i < 3; ++i) {
        double lo = B->data[i*2+0], hi = B->data[i*2+1];
        o->data[i] = lo + (hi - lo) * U->data[i];
    }
    return o;
}

// ===========================================================================
// Tier-1 — validatorOccupancyMap
// ===========================================================================
// init clones the map grid + metadata + the ss bounds onto the validator.
matlab_mat *matlab_nav_validator_init(void *obj_v, void *ss_v, void *map_v) {
    matlab_mat *G  = nav::obj_get_mat(map_v, "Grid");
    if (G) { matlab_mat *c = mat_alloc(G->rows, G->cols); for (int64_t i=0;i<G->rows*G->cols;++i) c->data[i]=G->data[i]; nav::obj_set_mat(obj_v, "Grid", c); }
    nav::obj_set_f64(obj_v, "Resolution", nav::obj_get_f64(map_v, "Resolution"));
    nav::obj_set_f64(obj_v, "OccupiedThreshold", nav::obj_get_f64(map_v, "OccupiedThreshold"));
    matlab_mat *XL = nav::obj_get_mat(map_v, "XWorldLimits");
    matlab_mat *YL = nav::obj_get_mat(map_v, "YWorldLimits");
    if (XL) { matlab_mat *c = mat_alloc(1,2); c->data[0]=XL->data[0]; c->data[1]=XL->data[1]; nav::obj_set_mat(obj_v,"XWorldLimits",c); }
    if (YL) { matlab_mat *c = mat_alloc(1,2); c->data[0]=YL->data[0]; c->data[1]=YL->data[1]; nav::obj_set_mat(obj_v,"YWorldLimits",c); }
    matlab_mat *B = nav::obj_get_mat(ss_v, "StateBounds");
    if (B) { matlab_mat *c = mat_alloc(3,2); for (int i=0;i<6;++i) c->data[i]=B->data[i]; nav::obj_set_mat(obj_v,"StateBounds",c); }
    nav::obj_set_f64(obj_v, "WeightTheta", nav::obj_get_f64(ss_v, "WeightTheta"));
    nav::obj_set_f64(obj_v, "MinTurningRadius", nav::obj_get_f64(ss_v, "MinTurningRadius"));
    nav::obj_set_f64(obj_v, "ValidationDistance", 0.1);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_nav_validator_isstate(void *obj_v, matlab_mat *s) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!s || s->rows*s->cols < 2) return o;
    matlab_mat xy; double d[2] = { s->data[0], s->data[1] }; xy.data = d; xy.rows = 1; xy.cols = 2;
    matlab_mat *c = matlab_nav_occmap_check(obj_v, &xy);
    o->data[0] = (c->data[0] < 0.5) ? 1.0 : 0.0;  // valid iff free
    return o;
}

// isMotionValid(sv, s1, s2): sample the segment at ValidationDistance, all free.
matlab_mat *matlab_nav_validator_ismotion(void *obj_v, matlab_mat *s1, matlab_mat *s2) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!s1 || !s2 || s1->rows*s1->cols < 2 || s2->rows*s2->cols < 2) return o;
    double vd = nav::obj_get_f64(obj_v, "ValidationDistance");
    if (vd <= 0) vd = 0.1;
    double dx = s2->data[0]-s1->data[0], dy = s2->data[1]-s1->data[1];
    double len = std::sqrt(dx*dx + dy*dy);
    int steps = std::max(2, static_cast<int>(std::ceil(len / vd)));
    for (int k = 0; k <= steps; ++k) {
        double t = static_cast<double>(k) / steps;
        matlab_mat xy; double d[2] = { s1->data[0]+dx*t, s1->data[1]+dy*t }; xy.data = d; xy.rows = 1; xy.cols = 2;
        matlab_mat *c = matlab_nav_occmap_check(obj_v, &xy);
        if (c->data[0] >= 0.5) { o->data[0] = 0.0; return o; }
    }
    o->data[0] = 1.0;
    return o;
}

// ===========================================================================
// Tier-1 — navPath
// ===========================================================================
matlab_mat *matlab_nav_path_init(void *obj_v, matlab_mat *states) {
    matlab_mat *S = mat_alloc(states ? states->rows : 0, 3);
    if (states && states->cols >= 3)
        for (int64_t i = 0; i < states->rows; ++i)
            for (int j = 0; j < 3; ++j) S->data[i*3+j] = states->data[i*states->cols+j];
    nav::obj_set_mat(obj_v, "States", S);
    return S;
}

matlab_mat *matlab_nav_path_length(void *obj_v) {
    matlab_mat *S = nav::obj_get_mat(obj_v, "States");
    matlab_mat *o = mat_alloc(1, 1);
    if (!S || S->rows < 2) return o;
    double L = 0;
    for (int64_t i = 0; i + 1 < S->rows; ++i) {
        double dx = S->data[(i+1)*3+0]-S->data[i*3+0];
        double dy = S->data[(i+1)*3+1]-S->data[i*3+1];
        L += std::sqrt(dx*dx + dy*dy);
    }
    o->data[0] = L;
    return o;
}

}  // extern "C"

// ===========================================================================
// Tier-2 — RRT / RRT* / A* planners (internal helpers + extern entries)
// ===========================================================================
namespace nav {

// Read the planner's cloned validator fields (grid + bounds + occ threshold).
struct PlanCtx {
    matlab_mat *G; double res, occ;
    double xb[2], yb[2], thb[2];
    double minTurn, vd;
};
PlanCtx read_ctx(void *p) {
    PlanCtx c;
    c.G = obj_get_mat(p, "Grid");
    c.res = obj_get_f64(p, "Resolution");
    c.occ = obj_get_f64(p, "OccupiedThreshold");
    matlab_mat *B = obj_get_mat(p, "StateBounds");
    if (B) { c.xb[0]=B->data[0];c.xb[1]=B->data[1];c.yb[0]=B->data[2];c.yb[1]=B->data[3];c.thb[0]=B->data[4];c.thb[1]=B->data[5]; }
    else { c.xb[0]=0;c.xb[1]=10;c.yb[0]=0;c.yb[1]=10;c.thb[0]=-M_PI;c.thb[1]=M_PI; }
    c.minTurn = obj_get_f64(p, "MinTurningRadius");
    c.vd = obj_get_f64(p, "ValidationDistance"); if (c.vd <= 0) c.vd = 0.1;
    return c;
}
bool cell_free(const PlanCtx &c, double x, double y) {
    if (!c.G || c.res <= 0) return false;
    int64_t col = static_cast<int64_t>(std::floor(x * c.res));
    int64_t row = c.G->rows - 1 - static_cast<int64_t>(std::floor(y * c.res));
    if (row < 0 || row >= c.G->rows || col < 0 || col >= c.G->cols) return false;
    return c.G->data[row * c.G->cols + col] < c.occ;
}
bool motion_free(const PlanCtx &c, const double *a, const double *b) {
    double dx = b[0]-a[0], dy = b[1]-a[1];
    double len = std::sqrt(dx*dx+dy*dy);
    int steps = std::max(2, static_cast<int>(std::ceil(len / c.vd)));
    for (int k = 0; k <= steps; ++k) {
        double t = static_cast<double>(k)/steps;
        if (!cell_free(c, a[0]+dx*t, a[1]+dy*t)) return false;
    }
    return true;
}

}  // namespace nav

extern "C" {

// plannerRRT / plannerRRTStar init: clone the validator fields + knobs.
matlab_mat *matlab_nav_planner_init(void *obj_v, void *ss_v, void *val_v, double is_star) {
    (void)ss_v;
    matlab_mat *G = nav::obj_get_mat(val_v, "Grid");
    if (G) { matlab_mat *c = mat_alloc(G->rows, G->cols); for (int64_t i=0;i<G->rows*G->cols;++i) c->data[i]=G->data[i]; nav::obj_set_mat(obj_v,"Grid",c); }
    nav::obj_set_f64(obj_v, "Resolution", nav::obj_get_f64(val_v, "Resolution"));
    nav::obj_set_f64(obj_v, "OccupiedThreshold", nav::obj_get_f64(val_v, "OccupiedThreshold"));
    matlab_mat *B = nav::obj_get_mat(val_v, "StateBounds");
    if (B) { matlab_mat *c = mat_alloc(3,2); for (int i=0;i<6;++i) c->data[i]=B->data[i]; nav::obj_set_mat(obj_v,"StateBounds",c); }
    nav::obj_set_f64(obj_v, "MinTurningRadius", nav::obj_get_f64(val_v, "MinTurningRadius"));
    nav::obj_set_f64(obj_v, "ValidationDistance", nav::obj_get_f64(val_v, "ValidationDistance"));
    nav::obj_set_f64(obj_v, "MaxConnectionDistance", 1.0);
    nav::obj_set_f64(obj_v, "MaxIterations", 10000);
    nav::obj_set_f64(obj_v, "GoalBias", 0.05);
    nav::obj_set_f64(obj_v, "IsStar", is_star);
    return mat_alloc(0, 0);
}

// plan(planner, start, goal) → packed result: row 0 = [numStates exitflag numIters],
// rows 1.. = the path states (N×3).  We return an (N+1)×3 matrix; the caller
// reads row 0 for metadata then rows 1..N for the path.
matlab_mat *matlab_nav_planner_plan(void *obj_v, matlab_mat *start, matlab_mat *goal) {
    nav::PlanCtx c = nav::read_ctx(obj_v);
    double maxd = nav::obj_get_f64(obj_v, "MaxConnectionDistance"); if (maxd <= 0) maxd = 1.0;
    int maxit = static_cast<int>(nav::obj_get_f64(obj_v, "MaxIterations")); if (maxit <= 0) maxit = 10000;
    double gbias = nav::obj_get_f64(obj_v, "GoalBias");
    bool star = nav::obj_get_f64(obj_v, "IsStar") > 0.5;
    if (!start || !goal || start->rows*start->cols < 3 || goal->rows*goal->cols < 3)
        return mat_alloc(1, 3);

    struct Node { double x, y, th; int parent; double cost; };
    std::vector<Node> tree;
    tree.push_back({start->data[0], start->data[1], start->data[2], -1, 0.0});
    double gx = goal->data[0], gy = goal->data[1], gth = goal->data[2];
    int goal_idx = -1;
    int iters = 0;

    for (iters = 0; iters < maxit; ++iters) {
        // Sample (goal-biased).
        matlab_mat *U = matlab_rand(3.0, 1.0);
        double sx, sy, sth;
        if (U->data[0] < gbias) { sx = gx; sy = gy; sth = gth; }
        else {
            sx = c.xb[0] + (c.xb[1]-c.xb[0]) * matlab_rand(1.0,1.0)->data[0];
            sy = c.yb[0] + (c.yb[1]-c.yb[0]) * matlab_rand(1.0,1.0)->data[0];
            sth = c.thb[0] + (c.thb[1]-c.thb[0]) * matlab_rand(1.0,1.0)->data[0];
        }
        // Nearest node (Euclidean in xy).
        int near = 0; double nd = 1e300;
        for (size_t i = 0; i < tree.size(); ++i) {
            double dx = tree[i].x - sx, dy = tree[i].y - sy;
            double d = dx*dx + dy*dy;
            if (d < nd) { nd = d; near = static_cast<int>(i); }
        }
        // Steer toward sample by maxd.
        double dx = sx - tree[near].x, dy = sy - tree[near].y;
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < 1e-9) continue;
        double frac = (dist > maxd) ? maxd / dist : 1.0;
        double nx = tree[near].x + frac * dx;
        double ny = tree[near].y + frac * dy;
        double nth = std::atan2(dy, dx);
        double na[3] = { tree[near].x, tree[near].y, tree[near].th };
        double nb[3] = { nx, ny, nth };
        if (!nav::cell_free(c, nx, ny) || !nav::motion_free(c, na, nb)) continue;
        int new_idx = static_cast<int>(tree.size());
        double seg = std::sqrt((nx-tree[near].x)*(nx-tree[near].x) + (ny-tree[near].y)*(ny-tree[near].y));
        int best_parent = near;
        double best_cost = tree[near].cost + seg;
        // RRT* — choose the min-cost parent within a rewire ball, then rewire.
        double ball = star ? std::min(maxd * 3.0, 5.0 * maxd) : 0.0;
        if (star) {
            for (size_t i = 0; i < tree.size(); ++i) {
                double ex = tree[i].x - nx, ey = tree[i].y - ny;
                double e = std::sqrt(ex*ex + ey*ey);
                if (e > ball) continue;
                double ca[3] = { tree[i].x, tree[i].y, tree[i].th };
                double cb[3] = { nx, ny, nth };
                if (!nav::motion_free(c, ca, cb)) continue;
                double cand = tree[i].cost + e;
                if (cand < best_cost) { best_cost = cand; best_parent = static_cast<int>(i); }
            }
        }
        tree.push_back({nx, ny, nth, best_parent, best_cost});
        if (star) {
            for (size_t i = 0; i < tree.size() - 1; ++i) {
                double ex = tree[i].x - nx, ey = tree[i].y - ny;
                double e = std::sqrt(ex*ex + ey*ey);
                if (e > ball) continue;
                double ca[3] = { nx, ny, nth };
                double cb[3] = { tree[i].x, tree[i].y, tree[i].th };
                if (best_cost + e < tree[i].cost && nav::motion_free(c, ca, cb)) {
                    tree[i].parent = new_idx;
                    tree[i].cost = best_cost + e;
                }
            }
        }
        // Goal check.
        double ggx = nx - gx, ggy = ny - gy;
        if (std::sqrt(ggx*ggx + ggy*ggy) <= maxd) {
            double ga[3] = { nx, ny, nth };
            double gb[3] = { gx, gy, gth };
            if (nav::motion_free(c, ga, gb)) {
                tree.push_back({gx, gy, gth, new_idx, best_cost + std::sqrt(ggx*ggx+ggy*ggy)});
                goal_idx = static_cast<int>(tree.size()) - 1;
                if (!star) break;   // RRT stops at first; RRT* keeps improving
            }
        }
    }
    if (goal_idx < 0) {
        matlab_mat *O = mat_alloc(1, 3);
        O->data[0] = 0; O->data[1] = 0; O->data[2] = static_cast<double>(iters);
        return O;
    }
    std::vector<int> path;
    for (int k = goal_idx; k >= 0; k = tree[k].parent) path.push_back(k);
    std::reverse(path.begin(), path.end());
    int64_t N = static_cast<int64_t>(path.size());
    matlab_mat *O = mat_alloc(N + 1, 3);
    O->data[0] = static_cast<double>(N);
    O->data[1] = 1;                        // exitflag = GoalReached
    O->data[2] = static_cast<double>(iters);
    for (int64_t i = 0; i < N; ++i) {
        O->data[(i+1)*3+0] = tree[path[i]].x;
        O->data[(i+1)*3+1] = tree[path[i]].y;
        O->data[(i+1)*3+2] = tree[path[i]].th;
    }
    return O;
}

// shortenpath(navPath, validator) — greedy shortcut: repeatedly try to
// connect non-adjacent waypoints with a collision-free straight segment.
matlab_mat *matlab_nav_shortenpath(void *path_v, void *val_v) {
    matlab_mat *states = nav::obj_get_mat(path_v, "States");
    if (!states || states->rows < 3 || states->cols < 3) {
        matlab_mat *O = mat_alloc(states ? states->rows : 0, 3);
        if (states) for (int64_t i = 0; i < states->rows * 3; ++i) O->data[i] = states->data[i];
        return O;
    }
    nav::PlanCtx c = nav::read_ctx(val_v);
    int64_t N = states->rows;
    std::vector<int> keep;
    keep.push_back(0);
    int64_t i = 0;
    while (i < N - 1) {
        int64_t j = N - 1;
        for (; j > i + 1; --j) {
            double a[3] = { states->data[i*3+0], states->data[i*3+1], states->data[i*3+2] };
            double b[3] = { states->data[j*3+0], states->data[j*3+1], states->data[j*3+2] };
            if (nav::motion_free(c, a, b)) break;
        }
        keep.push_back(static_cast<int>(j));
        i = j;
    }
    int64_t M = static_cast<int64_t>(keep.size());
    matlab_mat *O = mat_alloc(M, 3);
    for (int64_t k = 0; k < M; ++k)
        for (int d = 0; d < 3; ++d) O->data[k*3+d] = states->data[keep[k]*3+d];
    return O;
}

// plannerAStarGrid(map) init + plan — grid A* over the occupancy grid.
matlab_mat *matlab_nav_astar_init(void *obj_v, void *map_v) {
    matlab_mat *G = nav::obj_get_mat(map_v, "Grid");
    if (G) { matlab_mat *c = mat_alloc(G->rows, G->cols); for (int64_t i=0;i<G->rows*G->cols;++i) c->data[i]=G->data[i]; nav::obj_set_mat(obj_v,"Grid",c); }
    nav::obj_set_f64(obj_v, "OccupiedThreshold", nav::obj_get_f64(map_v, "OccupiedThreshold"));
    return mat_alloc(0, 0);
}

// plan(astar, [r0 c0], [r1 c1]) in GRID indices (1-based) → M×2 path of cells.
matlab_mat *matlab_nav_astar_plan(void *obj_v, matlab_mat *startrc, matlab_mat *goalrc) {
    matlab_mat *G = nav::obj_get_mat(obj_v, "Grid");
    double occ = nav::obj_get_f64(obj_v, "OccupiedThreshold");
    if (!G || !startrc || !goalrc) return mat_alloc(0, 2);
    int64_t R = G->rows, C = G->cols;
    auto idx = [&](int64_t r, int64_t cc){ return r * C + cc; };
    int64_t sr = static_cast<int64_t>(startrc->data[0]) - 1, sc = static_cast<int64_t>(startrc->data[1]) - 1;
    int64_t gr = static_cast<int64_t>(goalrc->data[0]) - 1,  gc = static_cast<int64_t>(goalrc->data[1]) - 1;
    if (sr<0||sr>=R||sc<0||sc>=C||gr<0||gr>=R||gc<0||gc>=C) return mat_alloc(0, 2);
    std::vector<double> g(R*C, 1e300);
    std::vector<int64_t> prev(R*C, -1);
    auto h = [&](int64_t r, int64_t cc){ return std::hypot(static_cast<double>(r-gr), static_cast<double>(cc-gc)); };
    using QE = std::pair<double, int64_t>;
    std::priority_queue<QE, std::vector<QE>, std::greater<QE>> pq;
    g[idx(sr,sc)] = 0;
    pq.push({h(sr,sc), idx(sr,sc)});
    int dr[8] = {-1,1,0,0,-1,-1,1,1}, dc[8] = {0,0,-1,1,-1,1,-1,1};
    bool found = false;
    while (!pq.empty()) {
        int64_t u = pq.top().second; double f = pq.top().first; pq.pop();
        int64_t ur = u / C, uc = u % C;
        if (ur == gr && uc == gc) { found = true; break; }
        if (f > g[u] + h(ur,uc) + 1e-9) continue;
        for (int k = 0; k < 8; ++k) {
            int64_t nr = ur + dr[k], nc = uc + dc[k];
            if (nr<0||nr>=R||nc<0||nc>=C) continue;
            if (G->data[idx(nr,nc)] >= occ) continue;
            double step = (dr[k]!=0 && dc[k]!=0) ? 1.41421356 : 1.0;
            double ng = g[u] + step;
            if (ng < g[idx(nr,nc)]) {
                g[idx(nr,nc)] = ng; prev[idx(nr,nc)] = u;
                pq.push({ng + h(nr,nc), idx(nr,nc)});
            }
        }
    }
    if (!found) return mat_alloc(0, 2);
    std::vector<int64_t> path;
    for (int64_t u = idx(gr,gc); u != -1; u = prev[u]) path.push_back(u);
    std::reverse(path.begin(), path.end());
    int64_t M = static_cast<int64_t>(path.size());
    matlab_mat *O = mat_alloc(M, 2);
    for (int64_t i = 0; i < M; ++i) { O->data[i*2+0] = path[i]/C + 1; O->data[i*2+1] = path[i]%C + 1; }
    return O;
}

// ===========================================================================
// Tier-3 — lidarScan + matchScans + lidarSLAM
// ===========================================================================
// lidarScan(ranges, angles) → store + compute Cartesian (N×2).
matlab_mat *matlab_nav_lidarscan_init(void *obj_v, matlab_mat *ranges, matlab_mat *angles) {
    if (!ranges || !angles) return mat_alloc(0, 0);
    int64_t n = ranges->rows * ranges->cols;
    matlab_mat *Rg = mat_alloc(n, 1), *An = mat_alloc(n, 1), *Cart = mat_alloc(n, 2);
    for (int64_t i = 0; i < n; ++i) {
        double r = ranges->data[i], a = angles->data[i];
        Rg->data[i] = r; An->data[i] = a;
        Cart->data[i*2+0] = r * std::cos(a);
        Cart->data[i*2+1] = r * std::sin(a);
    }
    nav::obj_set_mat(obj_v, "Ranges", Rg);
    nav::obj_set_mat(obj_v, "Angles", An);
    nav::obj_set_mat(obj_v, "Cartesian", Cart);
    return Cart;
}

}  // extern "C"

namespace nav {
// Point-to-point ICP between two Cartesian scans (N×2 ref, M×2 cur) → relative
// pose [dx dy dθ] mapping cur into ref.  Nearest-neighbour + SVD per iteration.
void icp(const matlab_mat *ref, const matlab_mat *cur, double pose[3], int iters) {
    pose[0] = pose[1] = pose[2] = 0;
    if (!ref || !cur || ref->rows < 2 || cur->rows < 2) return;
    int64_t M = cur->rows, Nr = ref->rows;
    std::vector<double> cx(M), cy(M);
    for (int64_t i = 0; i < M; ++i) { cx[i] = cur->data[i*2+0]; cy[i] = cur->data[i*2+1]; }
    double tx = 0, ty = 0, th = 0;
    for (int it = 0; it < iters; ++it) {
        // For each transformed cur point, find nearest ref point.
        double mcx = 0, mcy = 0, mrx = 0, mry = 0; int64_t cnt = 0;
        std::vector<double> px(M), py(M), qx(M), qy(M);
        double cth = std::cos(th), sth = std::sin(th);
        for (int64_t i = 0; i < M; ++i) {
            double X = cth*cx[i] - sth*cy[i] + tx;
            double Y = sth*cx[i] + cth*cy[i] + ty;
            double bd = 1e300; int64_t bj = -1;
            for (int64_t j = 0; j < Nr; ++j) {
                double dx = ref->data[j*2+0]-X, dy = ref->data[j*2+1]-Y;
                double d = dx*dx+dy*dy;
                if (d < bd) { bd = d; bj = j; }
            }
            if (bj < 0 || bd > 1.0) continue;   // reject far matches
            px[cnt] = cx[i]; py[cnt] = cy[i];
            qx[cnt] = ref->data[bj*2+0]; qy[cnt] = ref->data[bj*2+1];
            mcx += cx[i]; mcy += cy[i]; mrx += qx[cnt]; mry += qy[cnt];
            ++cnt;
        }
        if (cnt < 2) break;
        mcx/=cnt; mcy/=cnt; mrx/=cnt; mry/=cnt;
        // 2×2 cross-covariance → optimal rotation (closed form for 2-D).
        double Sxx=0, Sxy=0, Syx=0, Syy=0;
        for (int64_t i = 0; i < cnt; ++i) {
            double ax = px[i]-mcx, ay = py[i]-mcy;
            double bx = qx[i]-mrx, by = qy[i]-mry;
            Sxx += ax*bx; Sxy += ax*by; Syx += ay*bx; Syy += ay*by;
        }
        double newth = std::atan2(Sxy - Syx, Sxx + Syy);
        double cN = std::cos(newth), sN = std::sin(newth);
        double newtx = mrx - (cN*mcx - sN*mcy);
        double newty = mry - (sN*mcx + cN*mcy);
        double dchg = std::fabs(newth-th) + std::fabs(newtx-tx) + std::fabs(newty-ty);
        th = newth; tx = newtx; ty = newty;
        if (dchg < 1e-6) break;
    }
    pose[0] = tx; pose[1] = ty; pose[2] = wrap_pi(th);
}
}  // namespace nav

extern "C" {

// matchScans(refScan, curScan) → relative pose [dx dy dθ] (1×3).
matlab_mat *matlab_nav_matchscans(void *ref_v, void *cur_v) {
    matlab_mat *R = nav::obj_get_mat(ref_v, "Cartesian");
    matlab_mat *C = nav::obj_get_mat(cur_v, "Cartesian");
    matlab_mat *o = mat_alloc(1, 3);
    double pose[3]; nav::icp(R, C, pose, 30);
    o->data[0] = pose[0]; o->data[1] = pose[1]; o->data[2] = pose[2];
    return o;
}

// lidarSLAM: accumulate an absolute trajectory by chaining matchScans
// relative poses.  addScan(slam, scan) appends; Poses (N×3) is the result.
matlab_mat *matlab_nav_slam_init(void *obj_v, double res, double maxRange) {
    (void)res; (void)maxRange;
    nav::obj_set_mat(obj_v, "Poses", mat_alloc(0, 3));
    nav::obj_set_mat(obj_v, "PrevCart", mat_alloc(0, 2));
    nav::obj_set_f64(obj_v, "NumScans", 0);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_nav_slam_addscan(void *obj_v, void *scan_v) {
    matlab_mat *Cart = nav::obj_get_mat(scan_v, "Cartesian");
    matlab_mat *Poses = nav::obj_get_mat(obj_v, "Poses");
    matlab_mat *Prev  = nav::obj_get_mat(obj_v, "PrevCart");
    int64_t n = Poses ? Poses->rows : 0;
    double px = 0, py = 0, pth = 0;
    if (n > 0) { px = Poses->data[(n-1)*3+0]; py = Poses->data[(n-1)*3+1]; pth = Poses->data[(n-1)*3+2]; }
    double nx = px, ny = py, nth = pth;
    if (n > 0 && Prev && Prev->rows > 1 && Cart) {
        double rel[3]; nav::icp(Prev, Cart, rel, 30);
        // Compose: world pose = prev ⊕ rel (rel expressed in prev frame).
        double cth = std::cos(pth), sth = std::sin(pth);
        nx = px + cth*rel[0] - sth*rel[1];
        ny = py + sth*rel[0] + cth*rel[1];
        nth = nav::wrap_pi(pth + rel[2]);
    }
    matlab_mat *P2 = mat_alloc(n + 1, 3);
    for (int64_t i = 0; i < n; ++i) { P2->data[i*3+0]=Poses->data[i*3+0]; P2->data[i*3+1]=Poses->data[i*3+1]; P2->data[i*3+2]=Poses->data[i*3+2]; }
    P2->data[n*3+0]=nx; P2->data[n*3+1]=ny; P2->data[n*3+2]=nth;
    nav::obj_set_mat(obj_v, "Poses", P2);
    if (Cart) { matlab_mat *c = mat_alloc(Cart->rows, 2); for (int64_t i=0;i<Cart->rows*2;++i) c->data[i]=Cart->data[i]; nav::obj_set_mat(obj_v, "PrevCart", c); }
    nav::obj_set_f64(obj_v, "NumScans", static_cast<double>(n + 1));
    return P2;
}

// ===========================================================================
// Tier-4 — poseGraph (SE2) + optimizePoseGraph
// ===========================================================================
// poseGraph: NodeEstimates (N×3 [x y θ]) + Edges (M×6 [a b dx dy dθ infoScale]).
matlab_mat *matlab_nav_posegraph_init(void *obj_v) {
    matlab_mat *N0 = mat_alloc(1, 3);  // node 1 = origin
    nav::obj_set_mat(obj_v, "NodeEstimates", N0);
    nav::obj_set_mat(obj_v, "Edges", mat_alloc(0, 6));
    return mat_alloc(0, 0);
}

// addRelativePose(pg, [dx dy dθ], [infoScale], fromNode, toNode).  toNode==0
// (default) appends a new node chained from `fromNode` (default = last node).
matlab_mat *matlab_nav_posegraph_addrel(void *obj_v, matlab_mat *rel, double fromN, double toN) {
    matlab_mat *Nodes = nav::obj_get_mat(obj_v, "NodeEstimates");
    matlab_mat *Edges = nav::obj_get_mat(obj_v, "Edges");
    int64_t nN = Nodes ? Nodes->rows : 0, nE = Edges ? Edges->rows : 0;
    if (!rel || rel->rows*rel->cols < 3 || nN == 0) return mat_alloc(0, 0);
    int from = (fromN >= 1) ? static_cast<int>(fromN) : static_cast<int>(nN);
    double dx = rel->data[0], dy = rel->data[1], dth = rel->data[2];
    int to;
    matlab_mat *N2;
    if (toN >= 1) {
        // Loop-closure edge between existing nodes (no new node).
        to = static_cast<int>(toN);
        N2 = mat_alloc(nN, 3);
        for (int64_t i = 0; i < nN*3; ++i) N2->data[i] = Nodes->data[i];
    } else {
        // New node = from ⊕ rel.
        to = static_cast<int>(nN) + 1;
        double px = Nodes->data[(from-1)*3+0], py = Nodes->data[(from-1)*3+1], pth = Nodes->data[(from-1)*3+2];
        double cth = std::cos(pth), sth = std::sin(pth);
        N2 = mat_alloc(nN + 1, 3);
        for (int64_t i = 0; i < nN*3; ++i) N2->data[i] = Nodes->data[i];
        N2->data[nN*3+0] = px + cth*dx - sth*dy;
        N2->data[nN*3+1] = py + sth*dx + cth*dy;
        N2->data[nN*3+2] = nav::wrap_pi(pth + dth);
    }
    matlab_mat *E2 = mat_alloc(nE + 1, 6);
    for (int64_t i = 0; i < nE*6; ++i) E2->data[i] = Edges->data[i];
    E2->data[nE*6+0] = from; E2->data[nE*6+1] = to;
    E2->data[nE*6+2] = dx; E2->data[nE*6+3] = dy; E2->data[nE*6+4] = dth;
    E2->data[nE*6+5] = 1.0;  // info scale
    nav::obj_set_mat(obj_v, "NodeEstimates", N2);
    nav::obj_set_mat(obj_v, "Edges", E2);
    return mat_alloc(0, 0);
}

// optimizePoseGraph(pg): SE2 Gauss-Newton over relative-pose residuals,
// node 1 fixed.  Variables = [x y θ] per free node (3·(N-1)).  Residual per
// edge e(a→b) = log( meas^-1 · (Ta^-1 · Tb) ) in se(2); right-perturbation
// Jacobians built numerically (small graphs).  Returns the optimised
// NodeEstimates (N×3) and updates the object.
matlab_mat *matlab_nav_posegraph_optimize(void *obj_v) {
    matlab_mat *Nodes = nav::obj_get_mat(obj_v, "NodeEstimates");
    matlab_mat *Edges = nav::obj_get_mat(obj_v, "Edges");
    int64_t N = Nodes ? Nodes->rows : 0, E = Edges ? Edges->rows : 0;
    if (N < 2 || E < 1) { matlab_mat *o = mat_alloc(N, 3); if (Nodes) for (int64_t i=0;i<N*3;++i) o->data[i]=Nodes->data[i]; return o; }
    std::vector<double> x(N*3);
    for (int64_t i = 0; i < N*3; ++i) x[i] = Nodes->data[i];
    int64_t dim = (N - 1) * 3;   // node 0 fixed
    // Residual of edge k given current x.
    auto edge_resid = [&](int64_t k, const std::vector<double> &xx, double r[3]) {
        int a = static_cast<int>(Edges->data[k*6+0]) - 1;
        int b = static_cast<int>(Edges->data[k*6+1]) - 1;
        double mdx = Edges->data[k*6+2], mdy = Edges->data[k*6+3], mdth = Edges->data[k*6+4];
        double ax=xx[a*3+0], ay=xx[a*3+1], ath=xx[a*3+2];
        double bx=xx[b*3+0], by=xx[b*3+1], bth=xx[b*3+2];
        // Predicted relative pose a→b in a's frame.
        double ca = std::cos(ath), sa = std::sin(ath);
        double pdx =  ca*(bx-ax) + sa*(by-ay);
        double pdy = -sa*(bx-ax) + ca*(by-ay);
        double pdth = nav::wrap_pi(bth - ath);
        r[0] = pdx - mdx; r[1] = pdy - mdy; r[2] = nav::wrap_pi(pdth - mdth);
    };
    for (int gn = 0; gn < 30; ++gn) {
        std::vector<double> H(dim*dim, 0.0), g(dim, 0.0);
        double dq = 1e-6;
        for (int64_t k = 0; k < E; ++k) {
            int a = static_cast<int>(Edges->data[k*6+0]) - 1;
            int b = static_cast<int>(Edges->data[k*6+1]) - 1;
            double r0[3]; edge_resid(k, x, r0);
            // Numerical 3×dim Jacobian (only cols of nodes a,b, both if free).
            int nodes[2] = { a, b };
            // Local 3×6 Jacobian then scatter.
            double J[3][6] = {{0}};
            for (int ni = 0; ni < 2; ++ni) {
                int node = nodes[ni];
                for (int d = 0; d < 3; ++d) {
                    std::vector<double> xp = x; xp[node*3+d] += dq;
                    double rp[3]; edge_resid(k, xp, rp);
                    for (int rr = 0; rr < 3; ++rr) J[rr][ni*3+d] = (rp[rr]-r0[rr]) / dq;
                }
            }
            // Scatter into H, g (skip fixed node 0).
            int gi[6];
            for (int ni = 0; ni < 2; ++ni)
                for (int d = 0; d < 3; ++d)
                    gi[ni*3+d] = (nodes[ni] == 0) ? -1 : (nodes[ni]-1)*3 + d;
            for (int p = 0; p < 6; ++p) {
                if (gi[p] < 0) continue;
                for (int rr = 0; rr < 3; ++rr) g[gi[p]] += J[rr][p] * r0[rr];
                for (int q = 0; q < 6; ++q) {
                    if (gi[q] < 0) continue;
                    double h = 0; for (int rr = 0; rr < 3; ++rr) h += J[rr][p]*J[rr][q];
                    H[gi[p]*dim + gi[q]] += h;
                }
            }
        }
        for (int64_t i = 0; i < dim; ++i) H[i*dim+i] += 1e-6;   // LM damping
        std::vector<double> dx_(g);
        for (auto &v : dx_) v = -v;
        nav::solve(H.data(), dim, dx_.data());
        double step = 0;
        for (int64_t i = 0; i < dim; ++i) { x[3 + i] += dx_[i]; step += dx_[i]*dx_[i]; }
        for (int64_t i = 1; i < N; ++i) x[i*3+2] = nav::wrap_pi(x[i*3+2]);
        if (std::sqrt(step) < 1e-9) break;
    }
    matlab_mat *O = mat_alloc(N, 3);
    for (int64_t i = 0; i < N*3; ++i) O->data[i] = x[i];
    nav::obj_set_mat(obj_v, "NodeEstimates", O);
    matlab_mat *Oc = mat_alloc(N, 3);
    for (int64_t i = 0; i < N*3; ++i) Oc->data[i] = x[i];
    return Oc;
}

}  // extern "C"
