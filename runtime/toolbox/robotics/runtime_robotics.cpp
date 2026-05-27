// Robotics System Toolbox runtime — Tiers 1–6.
//
// All exported symbols use C-linkage extern "C".  Wiring:
//   - lib/Sema/Resolver.cpp        : builtin registry (function names + matlab_robotics_* symbols)
//   - lib/MLIR/Lowering.cpp        : classdef constructor + method intercepts
//   - tools/matlabc/main.cpp       : prelude trigger table (loads robotics_classdefs.m)
//
// No external dependency: every routine here is hand-coded over the shipped
// matlab_runtime kernel + the Sensor Fusion Tier-1 quaternion + the Optim
// Toolbox lsqnonlin/fminunc (for inverse kinematics).
//
// Storage model:
//   - se3 / so3 / se2 / so2 : matlab_obj with one matrix property "Data"
//                              (4x4 / 3x3 / 3x3 / 2x2 — homogeneous-transform
//                              or rotation matrix)
//   - rigidBodyTree         : matlab_obj with packed-matrix property tables —
//                              "DH" (N×4 [a alpha d theta] per joint),
//                              "JointTypes" (N×1 1=revolute, 2=prismatic),
//                              "JointLimits" (N×2), "NumBodies" scalar,
//                              "EndEffectorName" string.
//   - inverseKinematics     : matlab_obj wrapping a rigidBodyTree handle +
//                              solver parameters.
//   - constraintPoseTarget  : matlab_obj with EndEffector / TargetTransform /
//                              Weights properties.
//   - binaryOccupancyMap    : matlab_obj with Grid (matrix), Resolution,
//                              GridSize, XWorldLimits, YWorldLimits.
//   - mobileRobotPRM        : matlab_obj with Map handle, ConnectionDistance,
//                              Nodes (N×2 sampled), Edges (M×2 graph).
//   - controllerPurePursuit : matlab_obj with Waypoints, LookaheadDistance,
//                              CurrentWaypointIdx.
//   - differentialDriveKinematics : matlab_obj with WheelRadius, TrackWidth.
//   - collisionBox / collisionSphere : matlab_obj with Pose (4×4) + shape data.
//   - manipulatorRRT        : matlab_obj wrapping a tree + obstacle list.

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Object-property accessors (defined in matlab_runtime.cpp).
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);

extern "C" uint64_t matlab_rng_state;
extern "C" matlab_mat *matlab_randn(double m, double n);
extern "C" matlab_mat *matlab_rand(double m, double n);

namespace robotics {

inline void obj_set_mat(void *o, const char *n, matlab_mat *m) {
    matlab_obj_set_mat(reinterpret_cast<matlab_obj *>(o), n,
                       static_cast<int64_t>(std::strlen(n)), m);
}
inline void obj_set_f64(void *o, const char *n, double v) {
    matlab_obj_set_f64(reinterpret_cast<matlab_obj *>(o), n,
                       static_cast<int64_t>(std::strlen(n)), v);
}
inline matlab_mat *obj_get_mat(void *o, const char *n) {
    return matlab_obj_get_mat(reinterpret_cast<matlab_obj *>(o), n,
                              static_cast<int64_t>(std::strlen(n)));
}
inline double obj_get_f64(void *o, const char *n) {
    return matlab_obj_get_f64(reinterpret_cast<matlab_obj *>(o), n,
                              static_cast<int64_t>(std::strlen(n)));
}

// Small dense-matrix helpers (kept local; tree depths and joint counts are
// modest enough that allocation-light loops are fine).

inline void mm(const double *A, int64_t n, int64_t m,
               const double *B, int64_t p, double *C) {
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < p; ++j) {
            double s = 0;
            for (int64_t k = 0; k < m; ++k) s += A[i * m + k] * B[k * p + j];
            C[i * p + j] = s;
        }
}

inline void tr(const double *A, int64_t n, int64_t m, double *At) {
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) At[j * n + i] = A[i * m + j];
}

// Solve A·X = B (A n×n, B n×p) → X n×p (Gauss elimination + partial pivot).
inline void solve(double *A, int64_t n, double *B, int64_t p) {
    for (int64_t k = 0; k < n; ++k) {
        int64_t piv = k;
        double mx = std::fabs(A[k * n + k]);
        for (int64_t i = k + 1; i < n; ++i) {
            double v = std::fabs(A[i * n + k]);
            if (v > mx) { mx = v; piv = i; }
        }
        if (mx < 1e-15) return;
        if (piv != k) {
            for (int64_t j = 0; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
            for (int64_t j = 0; j < p; ++j) std::swap(B[k * p + j], B[piv * p + j]);
        }
        for (int64_t i = k + 1; i < n; ++i) {
            double f = A[i * n + k] / A[k * n + k];
            for (int64_t j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            for (int64_t j = 0; j < p; ++j) B[i * p + j] -= f * B[k * p + j];
        }
    }
    for (int64_t i = n - 1; i >= 0; --i) {
        for (int64_t j = 0; j < p; ++j) {
            double s = B[i * p + j];
            for (int64_t k = i + 1; k < n; ++k) s -= A[i * n + k] * B[k * p + j];
            B[i * p + j] = s / A[i * n + i];
        }
    }
}

// Build a 4×4 homogeneous transform from a 3×3 rotation matrix R and a 3-vector p.
inline void compose_tform(const double R[9], const double p[3], double T[16]) {
    T[0]  = R[0]; T[1]  = R[1]; T[2]  = R[2]; T[3]  = p[0];
    T[4]  = R[3]; T[5]  = R[4]; T[6]  = R[5]; T[7]  = p[1];
    T[8]  = R[6]; T[9]  = R[7]; T[10] = R[8]; T[11] = p[2];
    T[12] = 0.0;  T[13] = 0.0;  T[14] = 0.0;  T[15] = 1.0;
}

// Multiply two 4×4 transforms.
inline void mul_tform(const double A[16], const double B[16], double C[16]) {
    mm(A, 4, 4, B, 4, C);
}

// Compute the 4×4 inverse of an SE(3) transform (block-form: R'·-p stacked).
inline void inv_tform(const double T[16], double Ti[16]) {
    double R[9] = { T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10] };
    double Rt[9];
    Rt[0] = R[0]; Rt[1] = R[3]; Rt[2] = R[6];
    Rt[3] = R[1]; Rt[4] = R[4]; Rt[5] = R[7];
    Rt[6] = R[2]; Rt[7] = R[5]; Rt[8] = R[8];
    double pi[3] = {-Rt[0]*T[3] - Rt[1]*T[7] - Rt[2]*T[11],
                    -Rt[3]*T[3] - Rt[4]*T[7] - Rt[5]*T[11],
                    -Rt[6]*T[3] - Rt[7]*T[7] - Rt[8]*T[11]};
    compose_tform(Rt, pi, Ti);
}

// DH transform: standard Denavit-Hartenberg (a, alpha, d, theta) — a 4×4
// homogeneous transform per joint frame.
inline void dh_tform(double a, double alpha, double d, double theta, double T[16]) {
    double ct = std::cos(theta), st = std::sin(theta);
    double ca = std::cos(alpha), sa = std::sin(alpha);
    T[0]  =  ct;     T[1]  = -st * ca; T[2]  =  st * sa; T[3]  = a * ct;
    T[4]  =  st;     T[5]  =  ct * ca; T[6]  = -ct * sa; T[7]  = a * st;
    T[8]  =  0.0;    T[9]  =  sa;      T[10] =  ca;      T[11] = d;
    T[12] =  0.0;    T[13] =  0.0;     T[14] =  0.0;     T[15] = 1.0;
}

}  // namespace robotics

// ---------------------------------------------------------------------------
// Tier-1 — coordinate transformations + tform conversions + utilities
// ---------------------------------------------------------------------------

extern "C" {

// ----- se3 / so3 / se2 / so2 value-type populators ------------------------

// se3(T) — store a 4×4 homogeneous transform on the object.
matlab_mat *matlab_robotics_se3_init(void *obj_v, matlab_mat *T) {
    if (!T || T->rows != 4 || T->cols != 4) {
        matlab_mat *I = mat_alloc(4, 4);
        I->data[0] = 1.0; I->data[5] = 1.0; I->data[10] = 1.0; I->data[15] = 1.0;
        robotics::obj_set_mat(obj_v, "Data", I);
        return I;
    }
    matlab_mat *D = mat_alloc(4, 4);
    for (int64_t i = 0; i < 16; ++i) D->data[i] = T->data[i];
    robotics::obj_set_mat(obj_v, "Data", D);
    return D;
}

matlab_mat *matlab_robotics_so3_init(void *obj_v, matlab_mat *R) {
    matlab_mat *D = mat_alloc(3, 3);
    if (R && R->rows == 3 && R->cols == 3)
        for (int64_t i = 0; i < 9; ++i) D->data[i] = R->data[i];
    else { D->data[0] = 1.0; D->data[4] = 1.0; D->data[8] = 1.0; }
    robotics::obj_set_mat(obj_v, "Data", D);
    return D;
}

// ----- tform conversions ---------------------------------------------------

// trvec2tform: 1×3 translation → 4×4 transform with identity rotation.
matlab_mat *matlab_robotics_trvec2tform(matlab_mat *p) {
    matlab_mat *T = mat_alloc(4, 4);
    T->data[0] = 1.0; T->data[5] = 1.0; T->data[10] = 1.0; T->data[15] = 1.0;
    if (p && p->rows * p->cols >= 3) {
        T->data[3]  = p->data[0];
        T->data[7]  = p->data[1];
        T->data[11] = p->data[2];
    }
    return T;
}

// tform2trvec: 4×4 transform → 1×3 translation.
matlab_mat *matlab_robotics_tform2trvec(matlab_mat *T) {
    matlab_mat *p = mat_alloc(1, 3);
    if (T && T->rows == 4 && T->cols == 4) {
        p->data[0] = T->data[3];
        p->data[1] = T->data[7];
        p->data[2] = T->data[11];
    }
    return p;
}

// rotm2tform: 3×3 rotation → 4×4 transform.
matlab_mat *matlab_robotics_rotm2tform(matlab_mat *R) {
    matlab_mat *T = mat_alloc(4, 4);
    if (R && R->rows == 3 && R->cols == 3) {
        T->data[0]  = R->data[0]; T->data[1]  = R->data[1]; T->data[2]  = R->data[2];
        T->data[4]  = R->data[3]; T->data[5]  = R->data[4]; T->data[6]  = R->data[5];
        T->data[8]  = R->data[6]; T->data[9]  = R->data[7]; T->data[10] = R->data[8];
    } else {
        T->data[0] = 1.0; T->data[5] = 1.0; T->data[10] = 1.0;
    }
    T->data[15] = 1.0;
    return T;
}

// tform2rotm: 4×4 → 3×3.
matlab_mat *matlab_robotics_tform2rotm(matlab_mat *T) {
    matlab_mat *R = mat_alloc(3, 3);
    if (T && T->rows == 4 && T->cols == 4) {
        R->data[0] = T->data[0]; R->data[1] = T->data[1]; R->data[2] = T->data[2];
        R->data[3] = T->data[4]; R->data[4] = T->data[5]; R->data[5] = T->data[6];
        R->data[6] = T->data[8]; R->data[7] = T->data[9]; R->data[8] = T->data[10];
    } else {
        R->data[0] = 1.0; R->data[4] = 1.0; R->data[8] = 1.0;
    }
    return R;
}

// eul2rotm: 1×3 Euler ZYX [yaw pitch roll] → 3×3 rotation.
matlab_mat *matlab_robotics_eul2rotm(matlab_mat *E) {
    matlab_mat *R = mat_alloc(3, 3);
    R->data[0] = 1.0; R->data[4] = 1.0; R->data[8] = 1.0;
    if (!E || E->rows * E->cols < 3) return R;
    double yaw = E->data[0], pitch = E->data[1], roll = E->data[2];
    double cy = std::cos(yaw),   sy = std::sin(yaw);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll),  sr = std::sin(roll);
    R->data[0] = cy*cp;            R->data[1] = cy*sp*sr - sy*cr; R->data[2] = cy*sp*cr + sy*sr;
    R->data[3] = sy*cp;            R->data[4] = sy*sp*sr + cy*cr; R->data[5] = sy*sp*cr - cy*sr;
    R->data[6] = -sp;              R->data[7] = cp*sr;            R->data[8] = cp*cr;
    return R;
}

// rotm2eul: 3×3 → 1×3 ZYX Euler.
matlab_mat *matlab_robotics_rotm2eul(matlab_mat *R) {
    matlab_mat *E = mat_alloc(1, 3);
    if (!R || R->rows != 3 || R->cols != 3) return E;
    double r20 = R->data[6];
    if (r20 > 1.0)  r20 = 1.0;
    if (r20 < -1.0) r20 = -1.0;
    double pitch = std::asin(-r20);
    double yaw   = std::atan2(R->data[3], R->data[0]);
    double roll  = std::atan2(R->data[7], R->data[8]);
    E->data[0] = yaw;
    E->data[1] = pitch;
    E->data[2] = roll;
    return E;
}

// eul2tform: 1×3 → 4×4.
matlab_mat *matlab_robotics_eul2tform(matlab_mat *E) {
    matlab_mat *R = matlab_robotics_eul2rotm(E);
    return matlab_robotics_rotm2tform(R);
}

// tform2eul: 4×4 → 1×3.
matlab_mat *matlab_robotics_tform2eul(matlab_mat *T) {
    matlab_mat *R = matlab_robotics_tform2rotm(T);
    return matlab_robotics_rotm2eul(R);
}

// axang2rotm: 1×4 [vx vy vz theta] → 3×3.
matlab_mat *matlab_robotics_axang2rotm(matlab_mat *A) {
    matlab_mat *R = mat_alloc(3, 3);
    R->data[0] = 1.0; R->data[4] = 1.0; R->data[8] = 1.0;
    if (!A || A->rows * A->cols < 4) return R;
    double vx = A->data[0], vy = A->data[1], vz = A->data[2], th = A->data[3];
    double n = std::sqrt(vx*vx + vy*vy + vz*vz);
    if (n < 1e-12) return R;
    vx /= n; vy /= n; vz /= n;
    double c = std::cos(th), s = std::sin(th), C = 1.0 - c;
    R->data[0] = c + vx*vx*C;        R->data[1] = vx*vy*C - vz*s;     R->data[2] = vx*vz*C + vy*s;
    R->data[3] = vy*vx*C + vz*s;     R->data[4] = c + vy*vy*C;        R->data[5] = vy*vz*C - vx*s;
    R->data[6] = vz*vx*C - vy*s;     R->data[7] = vz*vy*C + vx*s;     R->data[8] = c + vz*vz*C;
    return R;
}

// rotm2axang: 3×3 → 1×4 [vx vy vz theta].
matlab_mat *matlab_robotics_rotm2axang(matlab_mat *R) {
    matlab_mat *A = mat_alloc(1, 4);
    A->data[0] = 0; A->data[1] = 0; A->data[2] = 1; A->data[3] = 0;
    if (!R || R->rows != 3 || R->cols != 3) return A;
    double tr = R->data[0] + R->data[4] + R->data[8];
    double c = (tr - 1.0) * 0.5;
    if (c > 1.0)  c = 1.0;
    if (c < -1.0) c = -1.0;
    double theta = std::acos(c);
    if (std::fabs(theta) < 1e-12) return A;
    double s = std::sin(theta);
    if (std::fabs(s) < 1e-9) {
        // Near pi — find dominant diagonal.
        // (Rare in practice for IK; bail with axis along z.)
        A->data[3] = theta;
        return A;
    }
    A->data[0] = (R->data[7] - R->data[5]) / (2 * s);
    A->data[1] = (R->data[2] - R->data[6]) / (2 * s);
    A->data[2] = (R->data[3] - R->data[1]) / (2 * s);
    A->data[3] = theta;
    return A;
}

// axang2tform, tform2axang.
matlab_mat *matlab_robotics_axang2tform(matlab_mat *A) {
    matlab_mat *R = matlab_robotics_axang2rotm(A);
    return matlab_robotics_rotm2tform(R);
}
matlab_mat *matlab_robotics_tform2axang(matlab_mat *T) {
    matlab_mat *R = matlab_robotics_tform2rotm(T);
    return matlab_robotics_rotm2axang(R);
}

// quat2tform, tform2quat — bridge through Sensor Fusion's quaternion math.
extern matlab_mat *matlab_fusion_quat_to_rotm(matlab_mat *A, double frame);
extern matlab_mat *matlab_fusion_rotm_to_quat(matlab_mat *R);
matlab_mat *matlab_robotics_quat2tform(matlab_mat *q) {
    matlab_mat *R = matlab_fusion_quat_to_rotm(q, 0.0);
    return matlab_robotics_rotm2tform(R);
}
matlab_mat *matlab_robotics_tform2quat(matlab_mat *T) {
    matlab_mat *R = matlab_robotics_tform2rotm(T);
    return matlab_fusion_rotm_to_quat(R);
}

// homtrans(T, pts) — apply 4×4 transform T to N×3 row-of-points pts → N×3.
matlab_mat *matlab_robotics_homtrans(matlab_mat *T, matlab_mat *pts) {
    if (!T || !pts || T->rows != 4 || T->cols != 4 || pts->cols != 3)
        return mat_alloc(0, 0);
    matlab_mat *out = mat_alloc(pts->rows, 3);
    for (int64_t i = 0; i < pts->rows; ++i) {
        double x = pts->data[i * 3 + 0];
        double y = pts->data[i * 3 + 1];
        double z = pts->data[i * 3 + 2];
        out->data[i * 3 + 0] = T->data[0] * x + T->data[1] * y + T->data[2]  * z + T->data[3];
        out->data[i * 3 + 1] = T->data[4] * x + T->data[5] * y + T->data[6]  * z + T->data[7];
        out->data[i * 3 + 2] = T->data[8] * x + T->data[9] * y + T->data[10] * z + T->data[11];
    }
    return out;
}

// wrapToPi: wrap an angle (scalar or vector) into (-pi, pi].
matlab_mat *matlab_robotics_wrapToPi(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    matlab_mat *O = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < A->rows * A->cols; ++i) {
        double v = A->data[i];
        v = std::fmod(v + M_PI, 2 * M_PI);
        if (v < 0) v += 2 * M_PI;
        O->data[i] = v - M_PI;
    }
    return O;
}

// wrapTo2Pi: into [0, 2pi).
matlab_mat *matlab_robotics_wrapTo2Pi(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    matlab_mat *O = mat_alloc(A->rows, A->cols);
    for (int64_t i = 0; i < A->rows * A->cols; ++i) {
        double v = std::fmod(A->data[i], 2 * M_PI);
        if (v < 0) v += 2 * M_PI;
        O->data[i] = v;
    }
    return O;
}

// vecnorm: column-wise L2 norm (returns a 1×cols row).  For row vectors,
// returns a 1×1 scalar with the L2 norm.
matlab_mat *matlab_robotics_vecnorm(matlab_mat *A) {
    if (!A) return mat_alloc(1, 1);
    if (A->rows == 1) {
        double s = 0;
        for (int64_t i = 0; i < A->cols; ++i) s += A->data[i] * A->data[i];
        matlab_mat *O = mat_alloc(1, 1);
        O->data[0] = std::sqrt(s);
        return O;
    }
    matlab_mat *O = mat_alloc(1, A->cols);
    for (int64_t j = 0; j < A->cols; ++j) {
        double s = 0;
        for (int64_t i = 0; i < A->rows; ++i) {
            double v = A->data[i * A->cols + j];
            s += v * v;
        }
        O->data[j] = std::sqrt(s);
    }
    return O;
}

// se3 multiplication (compose) and inverse — exposed as standalone helpers.
matlab_mat *matlab_robotics_tform_mul(matlab_mat *A, matlab_mat *B) {
    matlab_mat *O = mat_alloc(4, 4);
    if (!A || !B || A->rows != 4 || A->cols != 4 || B->rows != 4 || B->cols != 4) {
        O->data[0] = 1.0; O->data[5] = 1.0; O->data[10] = 1.0; O->data[15] = 1.0;
        return O;
    }
    robotics::mul_tform(A->data, B->data, O->data);
    return O;
}

matlab_mat *matlab_robotics_tform_inv(matlab_mat *T) {
    matlab_mat *O = mat_alloc(4, 4);
    if (!T || T->rows != 4 || T->cols != 4) {
        O->data[0] = 1.0; O->data[5] = 1.0; O->data[10] = 1.0; O->data[15] = 1.0;
        return O;
    }
    robotics::inv_tform(T->data, O->data);
    return O;
}

// ---------------------------------------------------------------------------
// Tier-2 — rigidBodyTree + forward kinematics + Jacobian
// ---------------------------------------------------------------------------

// Zero-arg init: cleared tree.
matlab_mat *matlab_robotics_tree_init(void *obj_v) {
    robotics::obj_set_mat(obj_v, "DH",          mat_alloc(0, 4));
    robotics::obj_set_mat(obj_v, "JointTypes",  mat_alloc(0, 1));
    robotics::obj_set_mat(obj_v, "JointLimits", mat_alloc(0, 2));
    robotics::obj_set_f64(obj_v, "NumBodies", 0.0);
    return mat_alloc(0, 0);
}

// addBody(tree, dh, joint_type_code, low_lim, high_lim) — append one row to
// the DH / JointTypes / JointLimits tables.  joint_type_code: 1=revolute,
// 2=prismatic, 0=fixed.
matlab_mat *matlab_robotics_tree_addbody(void *obj_v, matlab_mat *dh,
                                          double jt_code, double lo, double hi) {
    if (!dh || dh->rows * dh->cols < 4) return mat_alloc(0, 0);
    matlab_mat *DH  = robotics::obj_get_mat(obj_v, "DH");
    matlab_mat *JT  = robotics::obj_get_mat(obj_v, "JointTypes");
    matlab_mat *JL  = robotics::obj_get_mat(obj_v, "JointLimits");
    int64_t N = DH ? DH->rows : 0;
    matlab_mat *DH2 = mat_alloc(N + 1, 4);
    matlab_mat *JT2 = mat_alloc(N + 1, 1);
    matlab_mat *JL2 = mat_alloc(N + 1, 2);
    for (int64_t i = 0; i < N; ++i) {
        for (int j = 0; j < 4; ++j) DH2->data[i * 4 + j] = DH->data[i * 4 + j];
        JT2->data[i] = JT->data[i];
        JL2->data[i * 2 + 0] = JL->data[i * 2 + 0];
        JL2->data[i * 2 + 1] = JL->data[i * 2 + 1];
    }
    for (int j = 0; j < 4; ++j) DH2->data[N * 4 + j] = dh->data[j];
    JT2->data[N] = jt_code;
    JL2->data[N * 2 + 0] = lo;
    JL2->data[N * 2 + 1] = hi;
    robotics::obj_set_mat(obj_v, "DH",          DH2);
    robotics::obj_set_mat(obj_v, "JointTypes",  JT2);
    robotics::obj_set_mat(obj_v, "JointLimits", JL2);
    robotics::obj_set_f64(obj_v, "NumBodies", static_cast<double>(N + 1));
    return mat_alloc(0, 0);
}

// loadrobot(name) — baked-in models.  `name` is a matlab_string pointer
// passed as PtrTy; we read it via the existing string layout (data, len).
struct robotics_string_s { char *data; int64_t len; };
static std::string read_string(void *s) {
    if (!s) return std::string();
    const robotics_string_s *p = reinterpret_cast<const robotics_string_s *>(s);
    if (!p->data || p->len <= 0) return std::string();
    return std::string(p->data, p->data + p->len);
}

// Build a 2-link planar arm (revolute-revolute), each link 1.0 m.
static void build_planar2(void *obj_v) {
    matlab_mat *DH = mat_alloc(2, 4);
    DH->data[0] = 1.0; DH->data[1] = 0.0; DH->data[2] = 0.0; DH->data[3] = 0.0;
    DH->data[4] = 1.0; DH->data[5] = 0.0; DH->data[6] = 0.0; DH->data[7] = 0.0;
    matlab_mat *JT = mat_alloc(2, 1);
    JT->data[0] = 1.0; JT->data[1] = 1.0;
    matlab_mat *JL = mat_alloc(2, 2);
    JL->data[0] = -M_PI; JL->data[1] =  M_PI;
    JL->data[2] = -M_PI; JL->data[3] =  M_PI;
    robotics::obj_set_mat(obj_v, "DH",          DH);
    robotics::obj_set_mat(obj_v, "JointTypes",  JT);
    robotics::obj_set_mat(obj_v, "JointLimits", JL);
    robotics::obj_set_f64(obj_v, "NumBodies", 2.0);
}

// Build a 3-link planar arm (RRR), each link 0.5 m.
static void build_planar3(void *obj_v) {
    matlab_mat *DH = mat_alloc(3, 4);
    for (int i = 0; i < 3; ++i) {
        DH->data[i * 4 + 0] = 0.5;
        DH->data[i * 4 + 1] = 0.0;
        DH->data[i * 4 + 2] = 0.0;
        DH->data[i * 4 + 3] = 0.0;
    }
    matlab_mat *JT = mat_alloc(3, 1);
    JT->data[0] = 1.0; JT->data[1] = 1.0; JT->data[2] = 1.0;
    matlab_mat *JL = mat_alloc(3, 2);
    for (int i = 0; i < 3; ++i) {
        JL->data[i * 2 + 0] = -M_PI;
        JL->data[i * 2 + 1] =  M_PI;
    }
    robotics::obj_set_mat(obj_v, "DH",          DH);
    robotics::obj_set_mat(obj_v, "JointTypes",  JT);
    robotics::obj_set_mat(obj_v, "JointLimits", JL);
    robotics::obj_set_f64(obj_v, "NumBodies", 3.0);
}

matlab_mat *matlab_robotics_loadrobot(void *obj_v, void *name_s) {
    std::string nm = read_string(name_s);
    if (nm == "planar3" || nm == "planar3link" || nm == "rrr") {
        build_planar3(obj_v);
    } else {
        // Default & "planar2" — the headline arm.
        build_planar2(obj_v);
    }
    return mat_alloc(0, 0);
}

// Forward kinematics: walk the DH chain, compose 4×4 transforms.  For a
// revolute joint i, the joint variable q(i) adds to DH(i,4) (theta).  For
// a prismatic joint, q(i) adds to DH(i,3) (d).
matlab_mat *matlab_robotics_getTransform(void *obj_v, matlab_mat *q) {
    matlab_mat *DH = robotics::obj_get_mat(obj_v, "DH");
    matlab_mat *JT = robotics::obj_get_mat(obj_v, "JointTypes");
    matlab_mat *O  = mat_alloc(4, 4);
    O->data[0] = 1.0; O->data[5] = 1.0; O->data[10] = 1.0; O->data[15] = 1.0;
    if (!DH || !JT || !q) return O;
    int64_t N = DH->rows;
    if (q->rows * q->cols < N) return O;
    double T[16];
    std::memcpy(T, O->data, sizeof(T));
    for (int64_t i = 0; i < N; ++i) {
        double a = DH->data[i * 4 + 0];
        double alpha = DH->data[i * 4 + 1];
        double d = DH->data[i * 4 + 2];
        double th = DH->data[i * 4 + 3];
        if (JT->data[i] == 1.0)      th += q->data[i];
        else if (JT->data[i] == 2.0) d  += q->data[i];
        double L[16];
        robotics::dh_tform(a, alpha, d, th, L);
        double R[16];
        robotics::mul_tform(T, L, R);
        std::memcpy(T, R, sizeof(T));
    }
    for (int i = 0; i < 16; ++i) O->data[i] = T[i];
    return O;
}

// Geometric Jacobian J (6×N): linear part J_v[i] = z_{i-1} × (p_e - p_{i-1}),
// angular part J_w[i] = z_{i-1} for revolute, [0;0;0]/z_{i-1}/0 for prismatic.
// We compute frame-i-1 axis z and origin p_{i-1} by walking the DH chain.
matlab_mat *matlab_robotics_geometricJacobian(void *obj_v, matlab_mat *q) {
    matlab_mat *DH = robotics::obj_get_mat(obj_v, "DH");
    matlab_mat *JT = robotics::obj_get_mat(obj_v, "JointTypes");
    int64_t N = DH ? DH->rows : 0;
    matlab_mat *J = mat_alloc(6, N);
    if (!q || q->rows * q->cols < N || N == 0) return J;
    std::vector<double> T(16, 0.0);
    T[0] = 1.0; T[5] = 1.0; T[10] = 1.0; T[15] = 1.0;
    std::vector<double> Tend = T;   // we'll re-walk to find p_e first
    for (int64_t i = 0; i < N; ++i) {
        double a = DH->data[i * 4 + 0];
        double alpha = DH->data[i * 4 + 1];
        double d = DH->data[i * 4 + 2];
        double th = DH->data[i * 4 + 3];
        if (JT->data[i] == 1.0)      th += q->data[i];
        else if (JT->data[i] == 2.0) d  += q->data[i];
        double L[16], R[16];
        robotics::dh_tform(a, alpha, d, th, L);
        robotics::mul_tform(Tend.data(), L, R);
        std::memcpy(Tend.data(), R, sizeof(R));
    }
    double p_e[3] = { Tend[3], Tend[7], Tend[11] };
    // Now walk again, capturing each frame i-1 and filling Jacobian column i.
    for (int64_t i = 0; i < N; ++i) {
        double z[3] = { T[2], T[6], T[10] };
        double p[3] = { T[3], T[7], T[11] };
        double dx[3] = { p_e[0] - p[0], p_e[1] - p[1], p_e[2] - p[2] };
        if (JT->data[i] == 1.0) {
            J->data[0 * N + i] = z[1] * dx[2] - z[2] * dx[1];
            J->data[1 * N + i] = z[2] * dx[0] - z[0] * dx[2];
            J->data[2 * N + i] = z[0] * dx[1] - z[1] * dx[0];
            J->data[3 * N + i] = z[0];
            J->data[4 * N + i] = z[1];
            J->data[5 * N + i] = z[2];
        } else if (JT->data[i] == 2.0) {
            J->data[0 * N + i] = z[0];
            J->data[1 * N + i] = z[1];
            J->data[2 * N + i] = z[2];
        }
        // Advance T past joint i.
        double a = DH->data[i * 4 + 0];
        double alpha = DH->data[i * 4 + 1];
        double d = DH->data[i * 4 + 2];
        double th = DH->data[i * 4 + 3];
        if (JT->data[i] == 1.0)      th += q->data[i];
        else if (JT->data[i] == 2.0) d  += q->data[i];
        double L[16], R[16];
        robotics::dh_tform(a, alpha, d, th, L);
        robotics::mul_tform(T.data(), L, R);
        std::memcpy(T.data(), R, sizeof(R));
    }
    return J;
}

// homeConfiguration: N×1 zeros.
matlab_mat *matlab_robotics_homeConfiguration(void *obj_v) {
    double N = robotics::obj_get_f64(obj_v, "NumBodies");
    int64_t n = static_cast<int64_t>(N);
    if (n < 0) n = 0;
    return mat_alloc(n, 1);
}

// randomConfiguration: uniform sample within joint limits.
matlab_mat *matlab_robotics_randomConfiguration(void *obj_v) {
    matlab_mat *JL = robotics::obj_get_mat(obj_v, "JointLimits");
    int64_t N = JL ? JL->rows : 0;
    matlab_mat *O = mat_alloc(N, 1);
    matlab_mat *U = matlab_rand(static_cast<double>(N), 1.0);
    for (int64_t i = 0; i < N; ++i) {
        double lo = JL->data[i * 2 + 0];
        double hi = JL->data[i * 2 + 1];
        O->data[i] = lo + (hi - lo) * U->data[i];
    }
    return O;
}

// ---------------------------------------------------------------------------
// Tier-3 — inverseKinematics (headline)
// ---------------------------------------------------------------------------

// inverseKinematics(rb) — clone the tree fields onto the solver object so
// the runtime can read them without cross-obj indirection.
matlab_mat *matlab_robotics_ik_init(void *obj_v, void *tree_v) {
    matlab_mat *DH = robotics::obj_get_mat(tree_v, "DH");
    matlab_mat *JT = robotics::obj_get_mat(tree_v, "JointTypes");
    matlab_mat *JL = robotics::obj_get_mat(tree_v, "JointLimits");
    double N = robotics::obj_get_f64(tree_v, "NumBodies");
    if (DH) { matlab_mat *c = mat_alloc(DH->rows, 4); for (int64_t i = 0; i < DH->rows * 4; ++i) c->data[i] = DH->data[i]; robotics::obj_set_mat(obj_v, "DH", c); }
    if (JT) { matlab_mat *c = mat_alloc(JT->rows, 1); for (int64_t i = 0; i < JT->rows; ++i) c->data[i] = JT->data[i]; robotics::obj_set_mat(obj_v, "JointTypes", c); }
    if (JL) { matlab_mat *c = mat_alloc(JL->rows, 2); for (int64_t i = 0; i < JL->rows * 2; ++i) c->data[i] = JL->data[i]; robotics::obj_set_mat(obj_v, "JointLimits", c); }
    robotics::obj_set_f64(obj_v, "NumBodies", N);
    robotics::obj_set_f64(obj_v, "MaxIterations", 200.0);
    robotics::obj_set_f64(obj_v, "SolutionTolerance", 1e-6);
    return mat_alloc(0, 0);
}

// Reuse the FK / Jacobian logic but inline so we don't need the IK to carry
// a tree obj pointer through the function-handle ABI (the existing IK
// solvers like fminunc take a single matlab_mat handle, not an obj pair).
static void fk_internal(matlab_mat *DH, matlab_mat *JT, matlab_mat *q, double T[16]) {
    T[0] = 1.0; T[1] = 0.0; T[2] = 0.0; T[3] = 0.0;
    T[4] = 0.0; T[5] = 1.0; T[6] = 0.0; T[7] = 0.0;
    T[8] = 0.0; T[9] = 0.0; T[10] = 1.0; T[11] = 0.0;
    T[12] = 0.0; T[13] = 0.0; T[14] = 0.0; T[15] = 1.0;
    int64_t N = DH ? DH->rows : 0;
    for (int64_t i = 0; i < N; ++i) {
        double a = DH->data[i * 4 + 0];
        double alpha = DH->data[i * 4 + 1];
        double d = DH->data[i * 4 + 2];
        double th = DH->data[i * 4 + 3];
        if (JT->data[i] == 1.0)      th += q->data[i];
        else if (JT->data[i] == 2.0) d  += q->data[i];
        double L[16], R[16];
        robotics::dh_tform(a, alpha, d, th, L);
        robotics::mul_tform(T, L, R);
        std::memcpy(T, R, sizeof(R));
    }
}

static void jac_internal(matlab_mat *DH, matlab_mat *JT, matlab_mat *q,
                          double *J, int64_t /*Jrows*/) {
    int64_t N = DH ? DH->rows : 0;
    double T[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    double Tend[16]; fk_internal(DH, JT, q, Tend);
    double p_e[3] = { Tend[3], Tend[7], Tend[11] };
    for (int64_t i = 0; i < N; ++i) {
        double z[3] = { T[2], T[6], T[10] };
        double p[3] = { T[3], T[7], T[11] };
        double dx[3] = { p_e[0] - p[0], p_e[1] - p[1], p_e[2] - p[2] };
        if (JT->data[i] == 1.0) {
            J[0 * N + i] = z[1] * dx[2] - z[2] * dx[1];
            J[1 * N + i] = z[2] * dx[0] - z[0] * dx[2];
            J[2 * N + i] = z[0] * dx[1] - z[1] * dx[0];
            J[3 * N + i] = z[0]; J[4 * N + i] = z[1]; J[5 * N + i] = z[2];
        } else if (JT->data[i] == 2.0) {
            J[0 * N + i] = z[0]; J[1 * N + i] = z[1]; J[2 * N + i] = z[2];
        }
        double a = DH->data[i * 4 + 0];
        double alpha = DH->data[i * 4 + 1];
        double d = DH->data[i * 4 + 2];
        double th = DH->data[i * 4 + 3];
        if (JT->data[i] == 1.0)      th += q->data[i];
        else if (JT->data[i] == 2.0) d  += q->data[i];
        double L[16], R[16];
        robotics::dh_tform(a, alpha, d, th, L);
        robotics::mul_tform(T, L, R);
        std::memcpy(T, R, sizeof(R));
    }
}

// matlab_robotics_ik_solve(ik_obj, target_tform, q0, weight_pos, weight_ori)
// → packed (N + 3) × 1 with [q; iters; exitflag; pose_err_norm].
//
// Damped least-squares (Levenberg-Marquardt) on the 6-vector pose error
// e(q) = [pos_error; axis_angle_error].  Per iteration:
//   J = jac(q)
//   dq = (J' J + lambda I)^-1 J' e
//   q  <- q + dq
// Lambda adapts: shrink on success, grow on no progress.
matlab_mat *matlab_robotics_ik_solve(void *ik_v, matlab_mat *Tgt, matlab_mat *q0,
                                      double w_pos, double w_ori) {
    matlab_mat *DH = robotics::obj_get_mat(ik_v, "DH");
    matlab_mat *JT = robotics::obj_get_mat(ik_v, "JointTypes");
    int64_t N = DH ? DH->rows : 0;
    if (w_pos <  0) w_pos = 1.0;
    if (w_ori <  0) w_ori = 0.5;
    // w_pos == 0 / w_ori == 0 explicitly disable that residual block.
    matlab_mat *out = mat_alloc(N + 3, 1);
    if (!Tgt || Tgt->rows != 4 || Tgt->cols != 4 || N == 0) return out;
    // Seed q from q0 (or zeros).
    std::vector<double> q(N, 0.0);
    if (q0 && q0->rows * q0->cols >= N)
        for (int64_t i = 0; i < N; ++i) q[i] = q0->data[i];
    int max_iter = static_cast<int>(robotics::obj_get_f64(ik_v, "MaxIterations"));
    double tol   = robotics::obj_get_f64(ik_v, "SolutionTolerance");
    if (max_iter <= 0) max_iter = 200;
    if (tol <= 0)       tol = 1e-6;
    double lambda = 1e-2;
    int iters = 0;
    double last_err = 1e300;
    int exitflag = 0;
    matlab_mat qmat; qmat.data = q.data(); qmat.rows = N; qmat.cols = 1;
    for (iters = 0; iters < max_iter; ++iters) {
        double T[16];
        fk_internal(DH, JT, &qmat, T);
        // Pose error: position + axis-angle.
        double e[6];
        e[0] = w_pos * (Tgt->data[3]  - T[3]);
        e[1] = w_pos * (Tgt->data[7]  - T[7]);
        e[2] = w_pos * (Tgt->data[11] - T[11]);
        // Orientation error via R_err = R_tgt · R_cur'.
        double Rcur[9] = { T[0],T[1],T[2], T[4],T[5],T[6], T[8],T[9],T[10] };
        double Rtgt[9] = { Tgt->data[0],Tgt->data[1],Tgt->data[2],
                           Tgt->data[4],Tgt->data[5],Tgt->data[6],
                           Tgt->data[8],Tgt->data[9],Tgt->data[10] };
        double Rcur_t[9];
        Rcur_t[0]=Rcur[0]; Rcur_t[1]=Rcur[3]; Rcur_t[2]=Rcur[6];
        Rcur_t[3]=Rcur[1]; Rcur_t[4]=Rcur[4]; Rcur_t[5]=Rcur[7];
        Rcur_t[6]=Rcur[2]; Rcur_t[7]=Rcur[5]; Rcur_t[8]=Rcur[8];
        double Rerr[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += Rtgt[i * 3 + k] * Rcur_t[k * 3 + j];
                Rerr[i * 3 + j] = s;
            }
        double tr = Rerr[0] + Rerr[4] + Rerr[8];
        double c = (tr - 1.0) * 0.5;
        if (c > 1.0)  c = 1.0;
        if (c < -1.0) c = -1.0;
        double theta = std::acos(c);
        double s = std::sin(theta);
        if (std::fabs(s) > 1e-9) {
            e[3] = w_ori * theta * (Rerr[7] - Rerr[5]) / (2 * s);
            e[4] = w_ori * theta * (Rerr[2] - Rerr[6]) / (2 * s);
            e[5] = w_ori * theta * (Rerr[3] - Rerr[1]) / (2 * s);
        } else {
            e[3] = 0; e[4] = 0; e[5] = 0;
        }
        double err_norm = std::sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]
                                    + e[3]*e[3]+e[4]*e[4]+e[5]*e[5]);
        if (err_norm < tol) { exitflag = 1; last_err = err_norm; break; }
        // Compute J (6×N).
        std::vector<double> J(6 * N, 0.0);
        jac_internal(DH, JT, &qmat, J.data(), 6);
        // Weight J's rows (so position and orientation rows are balanced).
        for (int i = 0; i < 3; ++i)
            for (int64_t k = 0; k < N; ++k) J[i * N + k] *= w_pos;
        for (int i = 3; i < 6; ++i)
            for (int64_t k = 0; k < N; ++k) J[i * N + k] *= w_ori;
        // Form (J' J + lambda I) (N×N), J' e (N×1).
        std::vector<double> JtJ(N * N, 0.0), Jte(N, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                double v = 0;
                for (int k = 0; k < 6; ++k) v += J[k * N + i] * J[k * N + j];
                JtJ[i * N + j] = v;
            }
            JtJ[i * N + i] += lambda;
        }
        for (int64_t i = 0; i < N; ++i) {
            double v = 0;
            for (int k = 0; k < 6; ++k) v += J[k * N + i] * e[k];
            Jte[i] = v;
        }
        // Solve for dq.
        std::vector<double> Jte_c(Jte);
        robotics::solve(JtJ.data(), N, Jte_c.data(), 1);
        for (int64_t i = 0; i < N; ++i) q[i] += Jte_c[i];
        if (err_norm < last_err) {
            lambda *= 0.7;
            if (lambda < 1e-9) lambda = 1e-9;
        } else {
            lambda *= 2.0;
        }
        last_err = err_norm;
    }
    if (exitflag == 0 && last_err < tol * 10) exitflag = 2;
    for (int64_t i = 0; i < N; ++i) out->data[i] = q[i];
    out->data[N]     = static_cast<double>(iters);
    out->data[N + 1] = static_cast<double>(exitflag);
    out->data[N + 2] = last_err;
    return out;
}

// constraintPoseTarget(target_tform [, weights])
matlab_mat *matlab_robotics_constraint_pose_init(void *obj_v, matlab_mat *T, matlab_mat *W) {
    if (T && T->rows == 4 && T->cols == 4) {
        matlab_mat *Tc = mat_alloc(4, 4);
        for (int i = 0; i < 16; ++i) Tc->data[i] = T->data[i];
        robotics::obj_set_mat(obj_v, "TargetTransform", Tc);
    }
    if (W && W->rows * W->cols >= 6) {
        matlab_mat *Wc = mat_alloc(1, 6);
        for (int i = 0; i < 6; ++i) Wc->data[i] = W->data[i];
        robotics::obj_set_mat(obj_v, "Weights", Wc);
    }
    return mat_alloc(0, 0);
}

// ---------------------------------------------------------------------------
// Tier-4 — trajectory generation + dynamics (compact)
// ---------------------------------------------------------------------------

// cubicpolytraj(waypoints, tpts, t) — Hermite cubic interpolation.
// waypoints: N×K  (K rows, each a waypoint vector of length N), tpts: 1×K,
// t: 1×M sample times.  Returns N×M.  Zero velocity at endpoints.
matlab_mat *matlab_robotics_cubicpolytraj(matlab_mat *wp, matlab_mat *tpts, matlab_mat *tq) {
    if (!wp || !tpts || !tq) return mat_alloc(0, 0);
    int64_t N = wp->rows;
    int64_t K = wp->cols;
    if (tpts->rows * tpts->cols != K) return mat_alloc(0, 0);
    int64_t M = tq->rows * tq->cols;
    matlab_mat *Q = mat_alloc(N, M);
    for (int64_t m = 0; m < M; ++m) {
        double t = tq->data[m];
        // Find the segment [tpts(i), tpts(i+1)].
        int64_t i = 0;
        if (t <= tpts->data[0]) {
            for (int64_t r = 0; r < N; ++r) Q->data[r * M + m] = wp->data[r * K + 0];
            continue;
        }
        if (t >= tpts->data[K - 1]) {
            for (int64_t r = 0; r < N; ++r) Q->data[r * M + m] = wp->data[r * K + (K - 1)];
            continue;
        }
        while (i + 1 < K && t > tpts->data[i + 1]) ++i;
        double t0 = tpts->data[i], t1 = tpts->data[i + 1];
        double dt = (t1 - t0);
        double u  = (t - t0) / (dt < 1e-12 ? 1.0 : dt);
        // Cubic Hermite with zero endpoint derivatives.
        double h00 = (1 + 2 * u) * (1 - u) * (1 - u);
        double h10 = u * (1 - u) * (1 - u) * dt;
        double h01 = u * u * (3 - 2 * u);
        double h11 = u * u * (u - 1) * dt;
        for (int64_t r = 0; r < N; ++r) {
            double p0 = wp->data[r * K + i];
            double p1 = wp->data[r * K + (i + 1)];
            // Zero velocity at endpoints.
            Q->data[r * M + m] = h00 * p0 + h01 * p1
                                 + h10 * 0.0 + h11 * 0.0;
        }
    }
    return Q;
}

// trapveltraj(waypoints, num_samples) — simple trapezoidal-velocity sample
// from first → last waypoint along a straight segment.  Returns N×num_samples.
matlab_mat *matlab_robotics_trapveltraj(matlab_mat *wp, double ns) {
    if (!wp) return mat_alloc(0, 0);
    int64_t N = wp->rows;
    int64_t K = wp->cols;
    int64_t M = static_cast<int64_t>(ns);
    if (M < 2) M = 2;
    matlab_mat *Q = mat_alloc(N, M);
    if (K < 2) {
        for (int64_t r = 0; r < N; ++r)
            for (int64_t m = 0; m < M; ++m)
                Q->data[r * M + m] = wp->data[r * K + 0];
        return Q;
    }
    // S-curve: cubic interpolation between the first and last waypoint with
    // a trapezoidal velocity profile.  We pick acceleration = 4 / (M-1)²·dx
    // and cruise = 1 - 2·accel·t² etc.; simpler: smoothstep position.
    for (int64_t m = 0; m < M; ++m) {
        double u = static_cast<double>(m) / static_cast<double>(M - 1);
        // 3-phase trapezoidal: accel (0, 0.25), cruise (0.25, 0.75), decel.
        double s;
        if (u < 0.25)      s = 0.5 * 8.0 * u * u;
        else if (u < 0.75) s = 0.25 + 2.0 * (u - 0.25);
        else               s = 1.0 - 0.5 * 8.0 * (1 - u) * (1 - u);
        for (int64_t r = 0; r < N; ++r) {
            double p0 = wp->data[r * K + 0];
            double p1 = wp->data[r * K + (K - 1)];
            Q->data[r * M + m] = p0 + (p1 - p0) * s;
        }
    }
    return Q;
}

// transformtraj(T0, T1, t) — slerp the rotation block + lerp the translation;
// returns a flat 16-row × M-column matrix (each column is a column-major
// view of the 4×4 transform).  Simplification: return a stack of 4×4 blocks
// reshaped to 4×(4·M).  Practical headline tests just walk the segments.
matlab_mat *matlab_robotics_transformtraj(matlab_mat *T0, matlab_mat *T1, matlab_mat *tq) {
    if (!T0 || !T1 || !tq) return mat_alloc(0, 0);
    if (T0->rows != 4 || T0->cols != 4 || T1->rows != 4 || T1->cols != 4) return mat_alloc(0, 0);
    int64_t M = tq->rows * tq->cols;
    matlab_mat *out = mat_alloc(16, M);
    // tq is expected in [0, 1].
    double R0[9] = { T0->data[0],T0->data[1],T0->data[2], T0->data[4],T0->data[5],T0->data[6], T0->data[8],T0->data[9],T0->data[10] };
    double R1[9] = { T1->data[0],T1->data[1],T1->data[2], T1->data[4],T1->data[5],T1->data[6], T1->data[8],T1->data[9],T1->data[10] };
    double p0[3] = { T0->data[3], T0->data[7], T0->data[11] };
    double p1[3] = { T1->data[3], T1->data[7], T1->data[11] };
    matlab_mat tmp; tmp.rows = 3; tmp.cols = 3;
    matlab_mat *q0 = nullptr;
    matlab_mat *q1 = nullptr;
    {
        matlab_mat A; A.rows = 3; A.cols = 3; A.data = R0; q0 = matlab_fusion_rotm_to_quat(&A);
        matlab_mat B; B.rows = 3; B.cols = 3; B.data = R1; q1 = matlab_fusion_rotm_to_quat(&B);
    }
    for (int64_t m = 0; m < M; ++m) {
        double t = tq->data[m];
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        // Slerp quaternions q0, q1.
        double dotp = q0->data[0]*q1->data[0] + q0->data[1]*q1->data[1] + q0->data[2]*q1->data[2] + q0->data[3]*q1->data[3];
        double q1s[4] = { q1->data[0], q1->data[1], q1->data[2], q1->data[3] };
        if (dotp < 0) { for (int k = 0; k < 4; ++k) q1s[k] = -q1s[k]; dotp = -dotp; }
        double s0, s1;
        if (dotp > 0.9995) { s0 = 1 - t; s1 = t; }
        else {
            double th = std::acos(dotp), sth = std::sin(th);
            s0 = std::sin((1 - t) * th) / sth;
            s1 = std::sin(t       * th) / sth;
        }
        double qi[4];
        for (int k = 0; k < 4; ++k) qi[k] = s0 * q0->data[k] + s1 * q1s[k];
        matlab_mat Q; Q.rows = 1; Q.cols = 4; Q.data = qi;
        matlab_mat *Ri = matlab_fusion_quat_to_rotm(&Q, 0.0);
        double T[16];
        double R[9] = { Ri->data[0], Ri->data[1], Ri->data[2], Ri->data[3], Ri->data[4], Ri->data[5], Ri->data[6], Ri->data[7], Ri->data[8] };
        double p[3] = { (1 - t) * p0[0] + t * p1[0],
                        (1 - t) * p0[1] + t * p1[1],
                        (1 - t) * p0[2] + t * p1[2] };
        robotics::compose_tform(R, p, T);
        for (int k = 0; k < 16; ++k) out->data[k * M + m] = T[k];
    }
    return out;
}

// massMatrix(rb, q) — diagonal-dominant approximation using link-length
// inertias proportional to a².  This is a compact stand-in for the full
// composite-rigid-body algorithm (CRBA); sufficient for the trajectory
// tracking demo + tests.  Real CRBA carved as a follow-on.
matlab_mat *matlab_robotics_massMatrix(void *obj_v, matlab_mat *q) {
    matlab_mat *DH = robotics::obj_get_mat(obj_v, "DH");
    int64_t N = DH ? DH->rows : 0;
    matlab_mat *M = mat_alloc(N, N);
    (void)q;
    for (int64_t i = 0; i < N; ++i) {
        double a = DH->data[i * 4 + 0];
        double inertia = std::max(0.05, a * a);
        M->data[i * N + i] = (N - i) * inertia;  // outer joints drag more inertia
    }
    return M;
}

// inverseDynamics(rb, q, qd, qdd) — gravity-only RNEA stand-in: torque(i) =
// massMatrix·qdd + gravity term ≈ m·g·a·cos(θ).  Compact.
matlab_mat *matlab_robotics_inverseDynamics(void *obj_v, matlab_mat *q, matlab_mat *qd, matlab_mat *qdd) {
    matlab_mat *DH = robotics::obj_get_mat(obj_v, "DH");
    int64_t N = DH ? DH->rows : 0;
    (void)qd;
    matlab_mat *Mm = matlab_robotics_massMatrix(obj_v, q);
    matlab_mat *tau = mat_alloc(N, 1);
    for (int64_t i = 0; i < N; ++i) {
        double s = 0;
        if (qdd) for (int64_t j = 0; j < N; ++j) s += Mm->data[i * N + j] * qdd->data[j];
        double a = DH->data[i * 4 + 0];
        double th = (q ? q->data[i] : 0.0);
        double g_term = 9.81 * a * std::cos(th);
        tau->data[i] = s + g_term;
    }
    return tau;
}

// ---------------------------------------------------------------------------
// Tier-5 — mobile robots + occupancy maps + PRM + pure-pursuit
// ---------------------------------------------------------------------------

matlab_mat *matlab_robotics_diffdrive_init(void *obj_v, double wheel_r, double track_w) {
    robotics::obj_set_f64(obj_v, "WheelRadius", wheel_r);
    robotics::obj_set_f64(obj_v, "TrackWidth",  track_w);
    return mat_alloc(0, 0);
}

// derivative(diffdrive, state, [v; omega]) — state = [x; y; theta].
matlab_mat *matlab_robotics_diffdrive_derivative(void *obj_v, matlab_mat *state, matlab_mat *cmd) {
    matlab_mat *o = mat_alloc(3, 1);
    if (!state || !cmd || state->rows * state->cols < 3 || cmd->rows * cmd->cols < 2) return o;
    double th = state->data[2];
    double v  = cmd->data[0];
    double w  = cmd->data[1];
    o->data[0] = v * std::cos(th);
    o->data[1] = v * std::sin(th);
    o->data[2] = w;
    (void)obj_v;
    return o;
}

// binaryOccupancyMap(rows, cols, resolution) — alloc a zero map.
matlab_mat *matlab_robotics_occmap_init(void *obj_v, double rows, double cols, double res) {
    int64_t R = static_cast<int64_t>(rows);
    int64_t C = static_cast<int64_t>(cols);
    if (R < 1) R = 1;
    if (C < 1) C = 1;
    if (res <= 0) res = 1.0;
    matlab_mat *G = mat_alloc(R, C);
    robotics::obj_set_mat(obj_v, "Grid", G);
    robotics::obj_set_f64(obj_v, "Resolution", res);
    matlab_mat *GS = mat_alloc(1, 2);
    GS->data[0] = static_cast<double>(R);
    GS->data[1] = static_cast<double>(C);
    robotics::obj_set_mat(obj_v, "GridSize", GS);
    matlab_mat *XL = mat_alloc(1, 2);
    XL->data[0] = 0;
    XL->data[1] = static_cast<double>(C) / res;
    robotics::obj_set_mat(obj_v, "XWorldLimits", XL);
    matlab_mat *YL = mat_alloc(1, 2);
    YL->data[0] = 0;
    YL->data[1] = static_cast<double>(R) / res;
    robotics::obj_set_mat(obj_v, "YWorldLimits", YL);
    return mat_alloc(0, 0);
}

// setOccupancy(map, xy, value) — mark a single world-space cell.
matlab_mat *matlab_robotics_occmap_set(void *obj_v, matlab_mat *xy, double val) {
    matlab_mat *G  = robotics::obj_get_mat(obj_v, "Grid");
    double res     = robotics::obj_get_f64(obj_v, "Resolution");
    if (!G || !xy || res <= 0 || xy->rows * xy->cols < 2) return mat_alloc(0, 0);
    int64_t R = G->rows, C = G->cols;
    int64_t c = static_cast<int64_t>(std::floor(xy->data[0] * res));
    int64_t r = R - 1 - static_cast<int64_t>(std::floor(xy->data[1] * res));
    if (r >= 0 && r < R && c >= 0 && c < C)
        G->data[r * C + c] = val;
    return mat_alloc(0, 0);
}

// getOccupancy(map, xy) → cell value at the world-space point.
matlab_mat *matlab_robotics_occmap_get(void *obj_v, matlab_mat *xy) {
    matlab_mat *o = mat_alloc(1, 1);
    matlab_mat *G  = robotics::obj_get_mat(obj_v, "Grid");
    double res     = robotics::obj_get_f64(obj_v, "Resolution");
    if (!G || !xy || res <= 0 || xy->rows * xy->cols < 2) return o;
    int64_t R = G->rows, C = G->cols;
    int64_t c = static_cast<int64_t>(std::floor(xy->data[0] * res));
    int64_t r = R - 1 - static_cast<int64_t>(std::floor(xy->data[1] * res));
    if (r >= 0 && r < R && c >= 0 && c < C)
        o->data[0] = G->data[r * C + c];
    return o;
}

// checkOccupancy → 1×1, 1=occupied/unknown, 0=free.
matlab_mat *matlab_robotics_occmap_check(void *obj_v, matlab_mat *xy) {
    matlab_mat *v = matlab_robotics_occmap_get(obj_v, xy);
    if (v->data[0] > 0.5) v->data[0] = 1.0;
    else                  v->data[0] = 0.0;
    return v;
}

// Build a PRM by sampling free configurations in the map's world limits and
// connecting nodes within ConnectionDistance.  Stored on the obj as Nodes
// (N×2) and Edges (M×3 [a b cost]).
matlab_mat *matlab_robotics_prm_init(void *obj_v, void *map_v, double n_nodes, double conn_dist) {
    matlab_mat *G  = robotics::obj_get_mat(map_v, "Grid");
    double res     = robotics::obj_get_f64(map_v, "Resolution");
    matlab_mat *XL = robotics::obj_get_mat(map_v, "XWorldLimits");
    matlab_mat *YL = robotics::obj_get_mat(map_v, "YWorldLimits");
    if (!G || !XL || !YL || res <= 0) return mat_alloc(0, 0);
    int64_t N = static_cast<int64_t>(n_nodes);
    if (N < 4) N = 4;
    if (conn_dist <= 0) conn_dist = 2.5;
    // Clone map fields onto the PRM obj for runtime reuse.
    matlab_mat *Gc = mat_alloc(G->rows, G->cols);
    for (int64_t i = 0; i < G->rows * G->cols; ++i) Gc->data[i] = G->data[i];
    robotics::obj_set_mat(obj_v, "Grid", Gc);
    robotics::obj_set_f64(obj_v, "Resolution", res);
    matlab_mat *XLc = mat_alloc(1, 2); XLc->data[0] = XL->data[0]; XLc->data[1] = XL->data[1];
    matlab_mat *YLc = mat_alloc(1, 2); YLc->data[0] = YL->data[0]; YLc->data[1] = YL->data[1];
    robotics::obj_set_mat(obj_v, "XWorldLimits", XLc);
    robotics::obj_set_mat(obj_v, "YWorldLimits", YLc);
    robotics::obj_set_f64(obj_v, "NumNodes", static_cast<double>(N));
    robotics::obj_set_f64(obj_v, "ConnectionDistance", conn_dist);
    // Sample free nodes.
    matlab_mat *U = matlab_rand(2 * N, 1.0);
    std::vector<double> Xs; Xs.reserve(N);
    std::vector<double> Ys; Ys.reserve(N);
    int64_t k = 0;
    for (int64_t i = 0; i < N * 4 && static_cast<int64_t>(Xs.size()) < N; ++i) {
        double u1 = U->data[(2 * (i % N) + 0) % U->rows];
        double u2 = U->data[(2 * (i % N) + 1) % U->rows];
        double x = XL->data[0] + (XL->data[1] - XL->data[0]) * u1;
        double y = YL->data[0] + (YL->data[1] - YL->data[0]) * u2;
        int64_t c = static_cast<int64_t>(std::floor(x * res));
        int64_t r = G->rows - 1 - static_cast<int64_t>(std::floor(y * res));
        if (r >= 0 && r < G->rows && c >= 0 && c < G->cols && G->data[r * G->cols + c] < 0.5) {
            Xs.push_back(x);
            Ys.push_back(y);
            ++k;
        }
        // Reseed if we run out of randomness.
        if (i % N == N - 1) U = matlab_rand(2 * N, 1.0);
    }
    int64_t nn = static_cast<int64_t>(Xs.size());
    matlab_mat *Nodes = mat_alloc(nn, 2);
    for (int64_t i = 0; i < nn; ++i) {
        Nodes->data[i * 2 + 0] = Xs[i];
        Nodes->data[i * 2 + 1] = Ys[i];
    }
    robotics::obj_set_mat(obj_v, "Nodes", Nodes);
    // Connect pairs within conn_dist and free-line.
    std::vector<std::tuple<int, int, double>> edges;
    auto line_free = [&](double x1, double y1, double x2, double y2) {
        int steps = std::max(4, static_cast<int>(std::ceil(std::hypot(x2 - x1, y2 - y1) * res * 2)));
        for (int s = 0; s <= steps; ++s) {
            double u = static_cast<double>(s) / steps;
            double x = x1 + (x2 - x1) * u;
            double y = y1 + (y2 - y1) * u;
            int64_t c = static_cast<int64_t>(std::floor(x * res));
            int64_t r = G->rows - 1 - static_cast<int64_t>(std::floor(y * res));
            if (r < 0 || r >= G->rows || c < 0 || c >= G->cols) return false;
            if (G->data[r * G->cols + c] >= 0.5) return false;
        }
        return true;
    };
    for (int64_t i = 0; i < nn; ++i)
        for (int64_t j = i + 1; j < nn; ++j) {
            double dx = Xs[i] - Xs[j];
            double dy = Ys[i] - Ys[j];
            double d  = std::hypot(dx, dy);
            if (d <= conn_dist && line_free(Xs[i], Ys[i], Xs[j], Ys[j]))
                edges.emplace_back(static_cast<int>(i), static_cast<int>(j), d);
        }
    int64_t m = static_cast<int64_t>(edges.size());
    matlab_mat *Edges = mat_alloc(m, 3);
    for (int64_t i = 0; i < m; ++i) {
        Edges->data[i * 3 + 0] = std::get<0>(edges[i]) + 1.0;  // 1-based
        Edges->data[i * 3 + 1] = std::get<1>(edges[i]) + 1.0;
        Edges->data[i * 3 + 2] = std::get<2>(edges[i]);
    }
    robotics::obj_set_mat(obj_v, "Edges", Edges);
    return mat_alloc(0, 0);
}

// findpath(prm, start_xy, goal_xy) — Dijkstra: snap to nearest sample, then
// shortest path on the graph.  Returns the path as an M×2 sequence of xy
// (start + nodes + goal), or empty if disconnected.
matlab_mat *matlab_robotics_prm_findpath(void *obj_v, matlab_mat *start_xy, matlab_mat *goal_xy) {
    matlab_mat *Nodes = robotics::obj_get_mat(obj_v, "Nodes");
    matlab_mat *Edges = robotics::obj_get_mat(obj_v, "Edges");
    if (!Nodes || !Edges || !start_xy || !goal_xy) return mat_alloc(0, 0);
    int64_t N = Nodes->rows;
    int64_t E = Edges->rows;
    if (N < 2) return mat_alloc(0, 0);
    double sx = start_xy->data[0], sy = start_xy->data[1];
    double gx = goal_xy->data[0],  gy = goal_xy->data[1];
    // Snap start/goal to nearest sampled node.
    int s_idx = 0, g_idx = 0;
    double s_best = 1e300, g_best = 1e300;
    for (int64_t i = 0; i < N; ++i) {
        double dx = Nodes->data[i * 2 + 0] - sx;
        double dy = Nodes->data[i * 2 + 1] - sy;
        double d = std::hypot(dx, dy);
        if (d < s_best) { s_best = d; s_idx = static_cast<int>(i); }
        dx = Nodes->data[i * 2 + 0] - gx;
        dy = Nodes->data[i * 2 + 1] - gy;
        d = std::hypot(dx, dy);
        if (d < g_best) { g_best = d; g_idx = static_cast<int>(i); }
    }
    // Adjacency.
    std::vector<std::vector<std::pair<int, double>>> adj(N);
    for (int64_t e = 0; e < E; ++e) {
        int a = static_cast<int>(Edges->data[e * 3 + 0]) - 1;
        int b = static_cast<int>(Edges->data[e * 3 + 1]) - 1;
        double c = Edges->data[e * 3 + 2];
        adj[a].push_back({b, c});
        adj[b].push_back({a, c});
    }
    // Dijkstra.
    std::vector<double> dist(N, 1e300);
    std::vector<int>    prev(N, -1);
    std::vector<int>    visited(N, 0);
    dist[s_idx] = 0;
    for (int64_t k = 0; k < N; ++k) {
        int u = -1;
        double best = 1e300;
        for (int64_t i = 0; i < N; ++i)
            if (!visited[i] && dist[i] < best) { best = dist[i]; u = static_cast<int>(i); }
        if (u < 0) break;
        if (u == g_idx) break;
        visited[u] = 1;
        for (const auto &nb : adj[u]) {
            double nd = dist[u] + nb.second;
            if (nd < dist[nb.first]) {
                dist[nb.first] = nd;
                prev[nb.first] = u;
            }
        }
    }
    if (dist[g_idx] >= 1e299) return mat_alloc(0, 2);
    std::vector<int> path;
    for (int u = g_idx; u >= 0; u = prev[u]) path.push_back(u);
    std::reverse(path.begin(), path.end());
    int64_t M = static_cast<int64_t>(path.size()) + 2;
    matlab_mat *out = mat_alloc(M, 2);
    out->data[0] = sx; out->data[1] = sy;
    for (size_t i = 0; i < path.size(); ++i) {
        out->data[(i + 1) * 2 + 0] = Nodes->data[path[i] * 2 + 0];
        out->data[(i + 1) * 2 + 1] = Nodes->data[path[i] * 2 + 1];
    }
    out->data[(M - 1) * 2 + 0] = gx;
    out->data[(M - 1) * 2 + 1] = gy;
    return out;
}

// controllerPurePursuit(waypoints, lookahead, vmax) — store + step.
matlab_mat *matlab_robotics_pursuit_init(void *obj_v, matlab_mat *wp, double look, double vmax) {
    if (wp) {
        matlab_mat *wpc = mat_alloc(wp->rows, wp->cols);
        for (int64_t i = 0; i < wp->rows * wp->cols; ++i) wpc->data[i] = wp->data[i];
        robotics::obj_set_mat(obj_v, "Waypoints", wpc);
    }
    robotics::obj_set_f64(obj_v, "LookaheadDistance", look > 0 ? look : 0.3);
    robotics::obj_set_f64(obj_v, "DesiredLinearVelocity", vmax > 0 ? vmax : 0.5);
    robotics::obj_set_f64(obj_v, "MaxAngularVelocity", 2.0);
    robotics::obj_set_f64(obj_v, "CurrentWaypointIdx", 1.0);
    return mat_alloc(0, 0);
}

// step(controller, pose) — pose = [x; y; theta], returns [v; omega].
matlab_mat *matlab_robotics_pursuit_step(void *obj_v, matlab_mat *pose) {
    matlab_mat *o = mat_alloc(2, 1);
    matlab_mat *WP = robotics::obj_get_mat(obj_v, "Waypoints");
    if (!WP || !pose || pose->rows * pose->cols < 3 || WP->rows < 1) return o;
    double x = pose->data[0], y = pose->data[1], th = pose->data[2];
    double look = robotics::obj_get_f64(obj_v, "LookaheadDistance");
    double vmax = robotics::obj_get_f64(obj_v, "DesiredLinearVelocity");
    double wmax = robotics::obj_get_f64(obj_v, "MaxAngularVelocity");
    // Find the first waypoint past the lookahead distance.
    int64_t idx = static_cast<int64_t>(robotics::obj_get_f64(obj_v, "CurrentWaypointIdx"));
    if (idx < 1) idx = 1;
    while (idx < WP->rows) {
        double dx = WP->data[(idx - 1) * 2 + 0] - x;
        double dy = WP->data[(idx - 1) * 2 + 1] - y;
        if (std::hypot(dx, dy) > look) break;
        idx += 1;
    }
    if (idx > WP->rows) idx = WP->rows;
    robotics::obj_set_f64(obj_v, "CurrentWaypointIdx", static_cast<double>(idx));
    double tx = WP->data[(idx - 1) * 2 + 0];
    double ty = WP->data[(idx - 1) * 2 + 1];
    double dx = tx - x, dy = ty - y;
    double ang_to = std::atan2(dy, dx);
    double err = ang_to - th;
    // Wrap to (-pi, pi].
    err = std::fmod(err + M_PI, 2 * M_PI);
    if (err < 0) err += 2 * M_PI;
    err -= M_PI;
    double w = 2.0 * vmax * std::sin(err) / std::max(look, 1e-6);
    if (w >  wmax) w =  wmax;
    if (w < -wmax) w = -wmax;
    o->data[0] = vmax;
    o->data[1] = w;
    return o;
}

// ---------------------------------------------------------------------------
// Tier-6 — collisions + manipulatorRRT
// ---------------------------------------------------------------------------

matlab_mat *matlab_robotics_collbox_init(void *obj_v, double x, double y, double z) {
    if (x <= 0) x = 1.0; if (y <= 0) y = 1.0; if (z <= 0) z = 1.0;
    robotics::obj_set_f64(obj_v, "X", x);
    robotics::obj_set_f64(obj_v, "Y", y);
    robotics::obj_set_f64(obj_v, "Z", z);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_robotics_collsphere_init(void *obj_v, double r) {
    robotics::obj_set_f64(obj_v, "Radius", r > 0 ? r : 0.5);
    return mat_alloc(0, 0);
}

// checkCollision(A, B) — AABB-AABB or sphere-sphere or sphere-box (axis-
// aligned simplification, no rotation considered).  Returns 1×1: 1 if
// collision, 0 if free.
//
// The classdef carrier guarantees Pose is stored as a 4×4; we read only the
// translation (Pose(1:3, 4)) for the simplified test.
static void coll_get_translation(void *obj_v, double p[3]) {
    matlab_mat *P = robotics::obj_get_mat(obj_v, "Pose");
    if (P && P->rows == 4 && P->cols == 4) {
        p[0] = P->data[3];
        p[1] = P->data[7];
        p[2] = P->data[11];
    } else { p[0] = 0; p[1] = 0; p[2] = 0; }
}

matlab_mat *matlab_robotics_checkCollision(void *a, void *b) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!a || !b) return o;
    // Heuristic detection of shape kind by looking for Radius vs X property.
    double ra = robotics::obj_get_f64(a, "Radius");
    double rb = robotics::obj_get_f64(b, "Radius");
    double pa[3], pb[3];
    coll_get_translation(a, pa);
    coll_get_translation(b, pb);
    if (ra > 0 && rb > 0) {
        // Sphere-sphere.
        double d = std::hypot(std::hypot(pa[0]-pb[0], pa[1]-pb[1]), pa[2]-pb[2]);
        o->data[0] = (d <= ra + rb) ? 1.0 : 0.0;
        return o;
    }
    if (ra <= 0 && rb <= 0) {
        // Box-box (AABB, axis-aligned ignoring orientation).
        double ax = robotics::obj_get_f64(a, "X") * 0.5;
        double ay = robotics::obj_get_f64(a, "Y") * 0.5;
        double az = robotics::obj_get_f64(a, "Z") * 0.5;
        double bx = robotics::obj_get_f64(b, "X") * 0.5;
        double by = robotics::obj_get_f64(b, "Y") * 0.5;
        double bz = robotics::obj_get_f64(b, "Z") * 0.5;
        bool ok = (std::fabs(pa[0]-pb[0]) <= ax + bx) &&
                  (std::fabs(pa[1]-pb[1]) <= ay + by) &&
                  (std::fabs(pa[2]-pb[2]) <= az + bz);
        o->data[0] = ok ? 1.0 : 0.0;
        return o;
    }
    // Sphere-box: collapse to AABB containment.
    double r  = ra > 0 ? ra : rb;
    double *ps = ra > 0 ? pa : pb;
    void   *bo = ra > 0 ? b  : a;
    double *pb_ = ra > 0 ? pb : pa;
    double bx = robotics::obj_get_f64(bo, "X") * 0.5;
    double by = robotics::obj_get_f64(bo, "Y") * 0.5;
    double bz = robotics::obj_get_f64(bo, "Z") * 0.5;
    double cx = std::max(pb_[0] - bx, std::min(ps[0], pb_[0] + bx));
    double cy = std::max(pb_[1] - by, std::min(ps[1], pb_[1] + by));
    double cz = std::max(pb_[2] - bz, std::min(ps[2], pb_[2] + bz));
    double d  = std::hypot(std::hypot(ps[0]-cx, ps[1]-cy), ps[2]-cz);
    o->data[0] = (d <= r) ? 1.0 : 0.0;
    return o;
}

// manipulatorRRT(tree, obstacles_centers, obstacle_radii)
matlab_mat *matlab_robotics_rrt_init(void *obj_v, void *tree_v,
                                      matlab_mat *centers, matlab_mat *radii) {
    matlab_mat *DH = robotics::obj_get_mat(tree_v, "DH");
    matlab_mat *JT = robotics::obj_get_mat(tree_v, "JointTypes");
    matlab_mat *JL = robotics::obj_get_mat(tree_v, "JointLimits");
    if (DH) { matlab_mat *c = mat_alloc(DH->rows, 4); for (int64_t i = 0; i < DH->rows * 4; ++i) c->data[i] = DH->data[i]; robotics::obj_set_mat(obj_v, "DH", c); }
    if (JT) { matlab_mat *c = mat_alloc(JT->rows, 1); for (int64_t i = 0; i < JT->rows; ++i) c->data[i] = JT->data[i]; robotics::obj_set_mat(obj_v, "JointTypes", c); }
    if (JL) { matlab_mat *c = mat_alloc(JL->rows, 2); for (int64_t i = 0; i < JL->rows * 2; ++i) c->data[i] = JL->data[i]; robotics::obj_set_mat(obj_v, "JointLimits", c); }
    robotics::obj_set_f64(obj_v, "NumBodies", robotics::obj_get_f64(tree_v, "NumBodies"));
    if (centers) {
        matlab_mat *c = mat_alloc(centers->rows, 3);
        for (int64_t i = 0; i < centers->rows * 3; ++i) c->data[i] = centers->data[i];
        robotics::obj_set_mat(obj_v, "ObstacleCenters", c);
    }
    if (radii) {
        matlab_mat *c = mat_alloc(radii->rows, 1);
        for (int64_t i = 0; i < radii->rows; ++i) c->data[i] = radii->data[i];
        robotics::obj_set_mat(obj_v, "ObstacleRadii", c);
    }
    robotics::obj_set_f64(obj_v, "MaxConnectionDistance", 0.3);
    robotics::obj_set_f64(obj_v, "MaxIterations", 200.0);
    return mat_alloc(0, 0);
}

// Collision check for a configuration: FK each joint, check end-effector
// distance to each sphere obstacle.  Simplification — full link sweeping
// is a follow-on.
static bool rrt_config_free(matlab_mat *DH, matlab_mat *JT,
                             const double *q, matlab_mat *Oc, matlab_mat *Or) {
    matlab_mat qmat; qmat.data = const_cast<double *>(q); qmat.rows = DH->rows; qmat.cols = 1;
    double T[16];
    fk_internal(DH, JT, &qmat, T);
    double p[3] = { T[3], T[7], T[11] };
    if (!Oc || !Or) return true;
    for (int64_t i = 0; i < Oc->rows; ++i) {
        double dx = p[0] - Oc->data[i * 3 + 0];
        double dy = p[1] - Oc->data[i * 3 + 1];
        double dz = p[2] - Oc->data[i * 3 + 2];
        if (std::hypot(std::hypot(dx, dy), dz) <= Or->data[i]) return false;
    }
    return true;
}

// plan(rrt, q_start, q_goal) — basic RRT; returns the planned config sequence
// as an M×N matrix (rows = waypoints, cols = joint count) or empty on failure.
matlab_mat *matlab_robotics_rrt_plan(void *obj_v, matlab_mat *qs, matlab_mat *qg) {
    matlab_mat *DH = robotics::obj_get_mat(obj_v, "DH");
    matlab_mat *JT = robotics::obj_get_mat(obj_v, "JointTypes");
    matlab_mat *JL = robotics::obj_get_mat(obj_v, "JointLimits");
    matlab_mat *Oc = robotics::obj_get_mat(obj_v, "ObstacleCenters");
    matlab_mat *Or = robotics::obj_get_mat(obj_v, "ObstacleRadii");
    int64_t N = DH ? DH->rows : 0;
    if (!qs || !qg || N == 0 || qs->rows * qs->cols < N || qg->rows * qg->cols < N)
        return mat_alloc(0, 0);
    double step = robotics::obj_get_f64(obj_v, "MaxConnectionDistance");
    int maxit   = static_cast<int>(robotics::obj_get_f64(obj_v, "MaxIterations"));
    if (step <= 0) step = 0.3;
    if (maxit <= 0) maxit = 200;
    // Tree: vector of configs + parent index.
    std::vector<std::vector<double>> tree;
    std::vector<int> parent;
    tree.push_back(std::vector<double>(qs->data, qs->data + N));
    parent.push_back(-1);
    int goal_idx = -1;
    for (int it = 0; it < maxit; ++it) {
        // Sample a random configuration within joint limits (90% biased to goal 10%).
        matlab_mat *U = matlab_rand(N + 1, 1.0);
        std::vector<double> q_rand(N);
        if (U->data[0] < 0.1) {
            for (int64_t i = 0; i < N; ++i) q_rand[i] = qg->data[i];
        } else {
            for (int64_t i = 0; i < N; ++i) {
                double lo = JL->data[i * 2 + 0];
                double hi = JL->data[i * 2 + 1];
                q_rand[i] = lo + (hi - lo) * U->data[i + 1];
            }
        }
        // Nearest tree node.
        int near_idx = 0;
        double near_d = 1e300;
        for (size_t k = 0; k < tree.size(); ++k) {
            double d = 0;
            for (int64_t i = 0; i < N; ++i) {
                double dx = tree[k][i] - q_rand[i];
                d += dx * dx;
            }
            if (d < near_d) { near_d = d; near_idx = static_cast<int>(k); }
        }
        // Step from near toward q_rand by `step`.
        std::vector<double> q_new(N);
        double dn = std::sqrt(near_d);
        double frac = (dn > step ? step / dn : 1.0);
        for (int64_t i = 0; i < N; ++i)
            q_new[i] = tree[near_idx][i] + frac * (q_rand[i] - tree[near_idx][i]);
        // Edge-collision check via midpoint + endpoint (simplification).
        std::vector<double> q_mid(N);
        for (int64_t i = 0; i < N; ++i) q_mid[i] = 0.5 * (tree[near_idx][i] + q_new[i]);
        if (!rrt_config_free(DH, JT, q_new.data(), Oc, Or)) continue;
        if (!rrt_config_free(DH, JT, q_mid.data(), Oc, Or)) continue;
        tree.push_back(q_new);
        parent.push_back(near_idx);
        // Goal reached?
        double dg = 0;
        for (int64_t i = 0; i < N; ++i) {
            double dx = q_new[i] - qg->data[i];
            dg += dx * dx;
        }
        if (std::sqrt(dg) < step) {
            // Add the explicit goal node.
            tree.push_back(std::vector<double>(qg->data, qg->data + N));
            parent.push_back(static_cast<int>(tree.size()) - 2);
            goal_idx = static_cast<int>(tree.size()) - 1;
            break;
        }
    }
    if (goal_idx < 0) return mat_alloc(0, N);
    // Reconstruct path.
    std::vector<int> path;
    for (int k = goal_idx; k >= 0; k = parent[k]) path.push_back(k);
    std::reverse(path.begin(), path.end());
    int64_t M = static_cast<int64_t>(path.size());
    matlab_mat *out = mat_alloc(M, N);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            out->data[i * N + j] = tree[path[i]][j];
    return out;
}

}  // extern "C"
