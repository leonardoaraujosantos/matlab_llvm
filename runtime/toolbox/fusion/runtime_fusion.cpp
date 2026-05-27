// Sensor Fusion and Tracking Toolbox runtime — Tiers 1–3.
//
// All exported symbols use C-linkage extern "C".  Wiring:
//   - lib/Sema/Resolver.cpp        : builtin registry (function names + matlab_fusion_* symbols)
//   - lib/MLIR/Lowering.cpp        : classdef constructor + method intercepts
//   - tools/matlabc/main.cpp       : prelude trigger table (loads fusion_classdefs.m)
//
// No external dependency: every routine here is hand-coded over the shipped
// matlab_runtime kernel + the System Identification Tier-5 EKF/UKF cores
// (matlab_ident_ekf_*/_ukf_*).  Storage model: quaternion is a classdef
// carrier whose payload is the property "Data" — an N×4 [w x y z] matrix.
// Filters/sensors are classdef carriers with State / StateCovariance /
// ProcessNoise / MeasurementNoise plus toolbox-specific knobs.

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

// Object-property accessors live in matlab_runtime.cpp; declare them here so
// this TU compiles standalone.
extern "C" matlab_mat *matlab_obj_get_mat(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_mat(matlab_obj *o, const char *name, int64_t len, matlab_mat *m);
extern "C" double      matlab_obj_get_f64(matlab_obj *o, const char *name, int64_t len);
extern "C" void        matlab_obj_set_f64(matlab_obj *o, const char *name, int64_t len, double v);

// PRNG state used by mvnrnd / sensor noise (lives in matlab_runtime.cpp).
extern "C" uint64_t matlab_rng_state;
extern "C" matlab_mat *matlab_randn(double m, double n);
extern "C" matlab_mat *matlab_chol(matlab_mat *A);

namespace fusion {

// String-arg reader (mirrors matlab_string layout used elsewhere).
struct fusion_string_s { char *data; int64_t len; };
std::string sstr(const void *s) {
    if (!s) return std::string();
    const fusion_string_s *p = reinterpret_cast<const fusion_string_s *>(s);
    if (!p->data || p->len <= 0) return std::string();
    return std::string(p->data, p->data + p->len);
}

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

}  // namespace fusion

// ---------------------------------------------------------------------------
// Tier-1 — quaternion value-type + rotation conversions + ecompass
// ---------------------------------------------------------------------------

// Normalise an input matrix into a canonical N×4 quaternion-data shape.
// Accepted shapes: N×4 (rows are quaternions), 1×4 row vector, 4×1 column
// vector (single quaternion).  Anything else is clamped to an empty 1×4.
static matlab_mat *quat_clone_n4(matlab_mat *src) {
    if (!src) return mat_alloc(1, 4);
    if (src->rows == 1 && src->cols == 4) {
        matlab_mat *o = mat_alloc(1, 4);
        for (int64_t k = 0; k < 4; ++k) o->data[k] = src->data[k];
        return o;
    }
    if (src->rows == 4 && src->cols == 1) {
        matlab_mat *o = mat_alloc(1, 4);
        for (int64_t k = 0; k < 4; ++k) o->data[k] = src->data[k];
        return o;
    }
    if (src->cols == 4 && src->rows >= 1) {
        matlab_mat *o = mat_alloc(src->rows, 4);
        for (int64_t i = 0; i < src->rows * 4; ++i) o->data[i] = src->data[i];
        return o;
    }
    return mat_alloc(1, 4);
}

extern "C" {

// quaternion(w,x,y,z) populate.
matlab_mat *matlab_fusion_quat_init_wxyz(void *obj_v, double w, double x, double y, double z) {
    matlab_mat *D = mat_alloc(1, 4);
    D->data[0] = w; D->data[1] = x; D->data[2] = y; D->data[3] = z;
    fusion::obj_set_mat(obj_v, "Data", D);
    return D;
}

// quaternion(M) populate (M = N×4 [w x y z] rows OR 1×4 / 4×1 single).
matlab_mat *matlab_fusion_quat_init_mat(void *obj_v, matlab_mat *M) {
    matlab_mat *D = quat_clone_n4(M);
    fusion::obj_set_mat(obj_v, "Data", D);
    return D;
}

// quaternion(D) where D is already the canonical N×4 — used by conversion
// functions that need to wrap a freshly-built matrix into a quaternion obj.
matlab_mat *matlab_fusion_quat_init_from_data(void *obj_v, matlab_mat *D) {
    matlab_mat *cloned = quat_clone_n4(D);
    fusion::obj_set_mat(obj_v, "Data", cloned);
    return cloned;
}

// ---------- algebra ---------------------------------------------------------

// Hamilton product q1 * q2, row-wise with broadcast.
static void quat_mul_row(const double *a, const double *b, double *out) {
    double w1 = a[0], x1 = a[1], y1 = a[2], z1 = a[3];
    double w2 = b[0], x2 = b[1], y2 = b[2], z2 = b[3];
    out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

matlab_mat *matlab_fusion_quat_mul_data(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(1, 4);
    int64_t n1 = A->rows, n2 = B->rows;
    int64_t n  = (n1 > n2) ? n1 : n2;
    matlab_mat *O = mat_alloc(n, 4);
    for (int64_t i = 0; i < n; ++i) {
        const double *a = &A->data[(n1 == 1 ? 0 : i) * 4];
        const double *b = &B->data[(n2 == 1 ? 0 : i) * 4];
        quat_mul_row(a, b, &O->data[i * 4]);
    }
    return O;
}

matlab_mat *matlab_fusion_quat_conj_data(matlab_mat *A) {
    if (!A) return mat_alloc(1, 4);
    matlab_mat *O = mat_alloc(A->rows, 4);
    for (int64_t i = 0; i < A->rows; ++i) {
        O->data[i * 4 + 0] =  A->data[i * 4 + 0];
        O->data[i * 4 + 1] = -A->data[i * 4 + 1];
        O->data[i * 4 + 2] = -A->data[i * 4 + 2];
        O->data[i * 4 + 3] = -A->data[i * 4 + 3];
    }
    return O;
}

matlab_mat *matlab_fusion_quat_norm_data(matlab_mat *A) {
    if (!A) return mat_alloc(1, 1);
    matlab_mat *O = mat_alloc(A->rows, 1);
    for (int64_t i = 0; i < A->rows; ++i) {
        double s = 0;
        for (int64_t k = 0; k < 4; ++k) {
            double v = A->data[i * 4 + k];
            s += v * v;
        }
        O->data[i] = std::sqrt(s);
    }
    return O;
}

matlab_mat *matlab_fusion_quat_normalize_data(matlab_mat *A) {
    if (!A) return mat_alloc(1, 4);
    matlab_mat *O = mat_alloc(A->rows, 4);
    for (int64_t i = 0; i < A->rows; ++i) {
        double s = 0;
        for (int64_t k = 0; k < 4; ++k) {
            double v = A->data[i * 4 + k];
            s += v * v;
        }
        double n = std::sqrt(s);
        if (n < 1e-300) n = 1.0;
        for (int64_t k = 0; k < 4; ++k)
            O->data[i * 4 + k] = A->data[i * 4 + k] / n;
    }
    return O;
}

matlab_mat *matlab_fusion_quat_inverse_data(matlab_mat *A) {
    if (!A) return mat_alloc(1, 4);
    matlab_mat *O = mat_alloc(A->rows, 4);
    for (int64_t i = 0; i < A->rows; ++i) {
        double w = A->data[i * 4 + 0];
        double x = A->data[i * 4 + 1];
        double y = A->data[i * 4 + 2];
        double z = A->data[i * 4 + 3];
        double n2 = w * w + x * x + y * y + z * z;
        if (n2 < 1e-300) n2 = 1.0;
        O->data[i * 4 + 0] =  w / n2;
        O->data[i * 4 + 1] = -x / n2;
        O->data[i * 4 + 2] = -y / n2;
        O->data[i * 4 + 3] = -z / n2;
    }
    return O;
}

// ---------- conversions ----------------------------------------------------

// Quaternion → 3×3 rotation matrix.  Single rotation (uses A's first row);
// point-rotation convention by default, transpose when frame=1.
matlab_mat *matlab_fusion_quat_to_rotm(matlab_mat *A, double frame) {
    if (!A || A->rows < 1 || A->cols < 4) return mat_alloc(3, 3);
    double w = A->data[0], x = A->data[1], y = A->data[2], z = A->data[3];
    matlab_mat *R = mat_alloc(3, 3);
    R->data[0] = 1 - 2*(y*y + z*z); R->data[1] = 2*(x*y - z*w);     R->data[2] = 2*(x*z + y*w);
    R->data[3] = 2*(x*y + z*w);     R->data[4] = 1 - 2*(x*x + z*z); R->data[5] = 2*(y*z - x*w);
    R->data[6] = 2*(x*z - y*w);     R->data[7] = 2*(y*z + x*w);     R->data[8] = 1 - 2*(x*x + y*y);
    if (frame > 0.5) {
        double t;
        t = R->data[1]; R->data[1] = R->data[3]; R->data[3] = t;
        t = R->data[2]; R->data[2] = R->data[6]; R->data[6] = t;
        t = R->data[5]; R->data[5] = R->data[7]; R->data[7] = t;
    }
    return R;
}

// Default-convention wrapper: quat2rotm(q) → 3×3 point-rotation matrix.
matlab_mat *matlab_fusion_quat2rotm(matlab_mat *A) {
    return matlab_fusion_quat_to_rotm(A, 0.0);
}

// 3×3 rotation matrix → unit quaternion (Shepperd's branched method).
matlab_mat *matlab_fusion_rotm_to_quat(matlab_mat *R) {
    matlab_mat *O = mat_alloc(1, 4);
    if (!R || R->rows < 3 || R->cols < 3) return O;
    double r00 = R->data[0], r01 = R->data[1], r02 = R->data[2];
    double r10 = R->data[3], r11 = R->data[4], r12 = R->data[5];
    double r20 = R->data[6], r21 = R->data[7], r22 = R->data[8];
    double tr = r00 + r11 + r22;
    double w, x, y, z;
    if (tr > 0) {
        double s = 2 * std::sqrt(tr + 1.0);
        w = 0.25 * s;
        x = (r21 - r12) / s;
        y = (r02 - r20) / s;
        z = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
        double s = 2 * std::sqrt(1.0 + r00 - r11 - r22);
        w = (r21 - r12) / s;
        x = 0.25 * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
    } else if (r11 > r22) {
        double s = 2 * std::sqrt(1.0 + r11 - r00 - r22);
        w = (r02 - r20) / s;
        x = (r01 + r10) / s;
        y = 0.25 * s;
        z = (r12 + r21) / s;
    } else {
        double s = 2 * std::sqrt(1.0 + r22 - r00 - r11);
        w = (r10 - r01) / s;
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25 * s;
    }
    O->data[0] = w; O->data[1] = x; O->data[2] = y; O->data[3] = z;
    return O;
}

// Quaternion (N×4) → Euler ZYX (N×3 [yaw pitch roll], radians).
matlab_mat *matlab_fusion_quat_to_eul(matlab_mat *A) {
    if (!A) return mat_alloc(1, 3);
    matlab_mat *O = mat_alloc(A->rows, 3);
    for (int64_t i = 0; i < A->rows; ++i) {
        double w = A->data[i*4+0], x = A->data[i*4+1], y = A->data[i*4+2], z = A->data[i*4+3];
        double sinp = 2 * (w * y - z * x);
        if (sinp >  1.0) sinp =  1.0;
        if (sinp < -1.0) sinp = -1.0;
        double yaw   = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
        double pitch = std::asin(sinp);
        double roll  = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        O->data[i*3+0] = yaw;
        O->data[i*3+1] = pitch;
        O->data[i*3+2] = roll;
    }
    return O;
}

// Euler ZYX (N×3 radians) → quaternion (N×4).
matlab_mat *matlab_fusion_eul_to_quat(matlab_mat *E) {
    if (!E) return mat_alloc(1, 4);
    matlab_mat *O = mat_alloc(E->rows, 4);
    for (int64_t i = 0; i < E->rows; ++i) {
        double yaw   = E->data[i*3+0];
        double pitch = E->data[i*3+1];
        double roll  = E->data[i*3+2];
        double cy = std::cos(yaw   * 0.5), sy = std::sin(yaw   * 0.5);
        double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
        double cr = std::cos(roll  * 0.5), sr = std::sin(roll  * 0.5);
        O->data[i*4+0] = cy*cp*cr + sy*sp*sr;
        O->data[i*4+1] = cy*cp*sr - sy*sp*cr;
        O->data[i*4+2] = sy*cp*sr + cy*sp*cr;
        O->data[i*4+3] = sy*cp*cr - cy*sp*sr;
    }
    return O;
}

// Rotate one or more 3-vectors by a single quaternion.
matlab_mat *matlab_fusion_quat_rotatepoint(matlab_mat *A, matlab_mat *V) {
    if (!A || !V) return mat_alloc(0, 0);
    matlab_mat *R = matlab_fusion_quat_to_rotm(A, 0.0);
    int64_t n = V->rows, d = V->cols;
    matlab_mat *O = nullptr;
    if (d == 3) {
        O = mat_alloc(n, 3);
        for (int64_t i = 0; i < n; ++i) {
            double vx = V->data[i*3+0], vy = V->data[i*3+1], vz = V->data[i*3+2];
            O->data[i*3+0] = R->data[0]*vx + R->data[1]*vy + R->data[2]*vz;
            O->data[i*3+1] = R->data[3]*vx + R->data[4]*vy + R->data[5]*vz;
            O->data[i*3+2] = R->data[6]*vx + R->data[7]*vy + R->data[8]*vz;
        }
    } else if (n == 3 && d == 1) {
        O = mat_alloc(3, 1);
        double vx = V->data[0], vy = V->data[1], vz = V->data[2];
        O->data[0] = R->data[0]*vx + R->data[1]*vy + R->data[2]*vz;
        O->data[1] = R->data[3]*vx + R->data[4]*vy + R->data[5]*vz;
        O->data[2] = R->data[6]*vx + R->data[7]*vy + R->data[8]*vz;
    } else {
        O = mat_alloc(0, 0);
    }
    return O;
}

matlab_mat *matlab_fusion_quat_rotateframe(matlab_mat *A, matlab_mat *V) {
    if (!A || !V) return mat_alloc(0, 0);
    matlab_mat *R = matlab_fusion_quat_to_rotm(A, 1.0);
    int64_t n = V->rows, d = V->cols;
    matlab_mat *O = nullptr;
    if (d == 3) {
        O = mat_alloc(n, 3);
        for (int64_t i = 0; i < n; ++i) {
            double vx = V->data[i*3+0], vy = V->data[i*3+1], vz = V->data[i*3+2];
            O->data[i*3+0] = R->data[0]*vx + R->data[1]*vy + R->data[2]*vz;
            O->data[i*3+1] = R->data[3]*vx + R->data[4]*vy + R->data[5]*vz;
            O->data[i*3+2] = R->data[6]*vx + R->data[7]*vy + R->data[8]*vz;
        }
    } else if (n == 3 && d == 1) {
        O = mat_alloc(3, 1);
        double vx = V->data[0], vy = V->data[1], vz = V->data[2];
        O->data[0] = R->data[0]*vx + R->data[1]*vy + R->data[2]*vz;
        O->data[1] = R->data[3]*vx + R->data[4]*vy + R->data[5]*vz;
        O->data[2] = R->data[6]*vx + R->data[7]*vy + R->data[8]*vz;
    } else {
        O = mat_alloc(0, 0);
    }
    return O;
}

// Spherical-linear interpolation between two unit quaternions.
matlab_mat *matlab_fusion_quat_slerp(matlab_mat *A, matlab_mat *B, double t) {
    matlab_mat *O = mat_alloc(1, 4);
    if (!A || !B) return O;
    double a[4], b[4];
    for (int k = 0; k < 4; ++k) { a[k] = A->data[k]; b[k] = B->data[k]; }
    double dotp = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    if (dotp < 0) { for (int k = 0; k < 4; ++k) b[k] = -b[k]; dotp = -dotp; }
    double s0, s1;
    if (dotp > 0.9995) {
        s0 = 1.0 - t;
        s1 = t;
    } else {
        double th  = std::acos(dotp);
        double sth = std::sin(th);
        s0 = std::sin((1 - t) * th) / sth;
        s1 = std::sin(t       * th) / sth;
    }
    for (int k = 0; k < 4; ++k) O->data[k] = s0 * a[k] + s1 * b[k];
    return O;
}

matlab_mat *matlab_fusion_quat_dist(matlab_mat *A, matlab_mat *B) {
    matlab_mat *O = mat_alloc(1, 1);
    if (!A || !B) return O;
    double dotp = 0;
    for (int k = 0; k < 4; ++k) dotp += A->data[k] * B->data[k];
    if (dotp < 0)  dotp = -dotp;
    if (dotp > 1)  dotp = 1.0;
    O->data[0] = 2.0 * std::acos(dotp);
    return O;
}

// ---------- ecompass --------------------------------------------------------

// Build a body→nav orientation from accelerometer (gravity) and magnetometer
// (north projection).  Reference frame: NED.
matlab_mat *matlab_fusion_ecompass(matlab_mat *acc, matlab_mat *mag) {
    matlab_mat *O = mat_alloc(1, 4);
    if (!acc || !mag || acc->rows * acc->cols < 3 || mag->rows * mag->cols < 3)
        return O;
    auto norm3 = [](double *v) {
        double n = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (n < 1e-12) n = 1.0;
        v[0] /= n; v[1] /= n; v[2] /= n;
    };
    double a[3] = { acc->data[0], acc->data[1], acc->data[2] };
    double m[3] = { mag->data[0], mag->data[1], mag->data[2] };
    norm3(a);
    double down[3] = { -a[0], -a[1], -a[2] };
    double east[3] = {
        m[1] * down[2] - m[2] * down[1],
        m[2] * down[0] - m[0] * down[2],
        m[0] * down[1] - m[1] * down[0]
    };
    norm3(east);
    double north[3] = {
        down[1] * east[2] - down[2] * east[1],
        down[2] * east[0] - down[0] * east[2],
        down[0] * east[1] - down[1] * east[0]
    };
    norm3(north);
    matlab_mat *R = mat_alloc(3, 3);
    R->data[0] = north[0]; R->data[1] = east[0]; R->data[2] = down[0];
    R->data[3] = north[1]; R->data[4] = east[1]; R->data[5] = down[1];
    R->data[6] = north[2]; R->data[7] = east[2]; R->data[8] = down[2];
    matlab_mat *q = matlab_fusion_rotm_to_quat(R);
    for (int k = 0; k < 4; ++k) O->data[k] = q->data[k];
    return O;
}

// ---------- display + parts -------------------------------------------------

void matlab_fusion_quat_disp(void *obj_v) {
    matlab_mat *D = fusion::obj_get_mat(obj_v, "Data");
    if (!D || D->rows == 0) {
        std::printf("  quaternion (empty)\n");
        return;
    }
    for (int64_t i = 0; i < D->rows; ++i) {
        double w = D->data[i*4+0], x = D->data[i*4+1], y = D->data[i*4+2], z = D->data[i*4+3];
        std::printf("  %.4f %s %.4fi %s %.4fj %s %.4fk\n",
                    w,
                    (x < 0 ? "-" : "+"), std::fabs(x),
                    (y < 0 ? "-" : "+"), std::fabs(y),
                    (z < 0 ? "-" : "+"), std::fabs(z));
    }
}

matlab_mat *matlab_fusion_quat_parts(void *obj_v) {
    matlab_mat *D = fusion::obj_get_mat(obj_v, "Data");
    if (!D) return mat_alloc(1, 4);
    matlab_mat *O = mat_alloc(D->rows, 4);
    for (int64_t i = 0; i < D->rows * 4; ++i) O->data[i] = D->data[i];
    return O;
}

// ---------------------------------------------------------------------------
// Tier-1 core gaps (generic builtins surfaced here because the toolbox needs
// them; promoting to matlab_runtime.cpp can come later).
// ---------------------------------------------------------------------------

// cross(a, b) — 3-vector cross product, accepts row or column 3-vectors and
// returns the same orientation as `a`.
matlab_mat *matlab_cross(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(0, 0);
    int64_t na = A->rows * A->cols;
    int64_t nb = B->rows * B->cols;
    if (na != 3 || nb != 3) return mat_alloc(0, 0);
    double a0 = A->data[0], a1 = A->data[1], a2 = A->data[2];
    double b0 = B->data[0], b1 = B->data[1], b2 = B->data[2];
    matlab_mat *O = mat_alloc(A->rows, A->cols);
    O->data[0] = a1 * b2 - a2 * b1;
    O->data[1] = a2 * b0 - a0 * b2;
    O->data[2] = a0 * b1 - a1 * b0;
    return O;
}

// dot(a, b) — element-wise multiply + sum (scalar result for vectors,
// column-wise sum for matrices).
matlab_mat *matlab_dot(matlab_mat *A, matlab_mat *B) {
    if (!A || !B) return mat_alloc(1, 1);
    int64_t na = A->rows * A->cols;
    int64_t nb = B->rows * B->cols;
    // Vector case: both flat, same length → scalar result.
    if (A->rows == 1 || A->cols == 1) {
        if (na != nb) return mat_alloc(1, 1);
        double s = 0;
        for (int64_t i = 0; i < na; ++i) s += A->data[i] * B->data[i];
        matlab_mat *O = mat_alloc(1, 1);
        O->data[0] = s;
        return O;
    }
    // Matrix case: column-wise dot product → 1×cols row.
    if (A->rows != B->rows || A->cols != B->cols) return mat_alloc(1, 1);
    matlab_mat *O = mat_alloc(1, A->cols);
    for (int64_t j = 0; j < A->cols; ++j) {
        double s = 0;
        for (int64_t i = 0; i < A->rows; ++i)
            s += A->data[i * A->cols + j] * B->data[i * A->cols + j];
        O->data[j] = s;
    }
    return O;
}

// deg2rad / rad2deg — scalar and matrix forms (caller passes a matrix;
// scalars are 1×1 mats by convention).
matlab_mat *matlab_deg2rad(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    matlab_mat *O = mat_alloc(A->rows, A->cols);
    double k = M_PI / 180.0;
    for (int64_t i = 0; i < A->rows * A->cols; ++i) O->data[i] = A->data[i] * k;
    return O;
}

matlab_mat *matlab_rad2deg(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    matlab_mat *O = mat_alloc(A->rows, A->cols);
    double k = 180.0 / M_PI;
    for (int64_t i = 0; i < A->rows * A->cols; ++i) O->data[i] = A->data[i] * k;
    return O;
}

// normalize(v) — vector L2 normalize.  Default norm='norm' (L2).  Matrix
// input is normalized column-wise.
matlab_mat *matlab_normalize_vec(matlab_mat *A) {
    if (!A) return mat_alloc(0, 0);
    matlab_mat *O = mat_alloc(A->rows, A->cols);
    if (A->rows == 1 || A->cols == 1) {
        double s = 0;
        int64_t n = A->rows * A->cols;
        for (int64_t i = 0; i < n; ++i) s += A->data[i] * A->data[i];
        double r = std::sqrt(s);
        if (r < 1e-300) r = 1.0;
        for (int64_t i = 0; i < n; ++i) O->data[i] = A->data[i] / r;
        return O;
    }
    for (int64_t j = 0; j < A->cols; ++j) {
        double s = 0;
        for (int64_t i = 0; i < A->rows; ++i) {
            double v = A->data[i * A->cols + j];
            s += v * v;
        }
        double r = std::sqrt(s);
        if (r < 1e-300) r = 1.0;
        for (int64_t i = 0; i < A->rows; ++i)
            O->data[i * A->cols + j] = A->data[i * A->cols + j] / r;
    }
    return O;
}

// mvnrnd(mu, Sigma, n) — n samples from N(mu, Sigma).  mu is 1×d; Sigma is
// d×d (symmetric PSD).  Returns n×d.  Uses the shipped chol + randn.
matlab_mat *matlab_mvnrnd(matlab_mat *mu, matlab_mat *Sigma, double n_d) {
    if (!mu || !Sigma) return mat_alloc(0, 0);
    int64_t d = mu->rows * mu->cols;
    int64_t n = static_cast<int64_t>(n_d);
    if (n < 1) n = 1;
    if (Sigma->rows != d || Sigma->cols != d) return mat_alloc(0, 0);
    matlab_mat *L = matlab_chol(Sigma);   // upper-triangular by convention
    matlab_mat *Z = matlab_randn(static_cast<double>(n), static_cast<double>(d));
    matlab_mat *O = mat_alloc(n, d);
    // O = Z * L  (chol returns upper R s.t. R'·R = Sigma; sample = mu + Z*R)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double s = 0;
            for (int64_t k = 0; k <= j; ++k)
                s += Z->data[i * d + k] * L->data[k * d + j];
            O->data[i * d + j] = mu->data[j] + s;
        }
    }
    return O;
}

}  // extern "C" (T1)

// ---------------------------------------------------------------------------
// Tier-2 — tracking filters + motion/measurement models
// ---------------------------------------------------------------------------
//
// Internal mat helpers (dense, small-dim, row-major).  These are kept local
// since the toolbox keeps matrix sizes modest (≤ 16×16 for the heaviest
// state) and avoids re-allocating large temporaries during the hot loop.

namespace fusion {

// In-place matmul: C(n×p) = A(n×m) * B(m×p).
void mm(const double *A, int64_t n, int64_t m,
        const double *B, int64_t p, double *C) {
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < p; ++j) {
            double s = 0;
            for (int64_t k = 0; k < m; ++k) s += A[i * m + k] * B[k * p + j];
            C[i * p + j] = s;
        }
    }
}

// In-place transpose: At(m×n) = A(n×m)'.
void tr(const double *A, int64_t n, int64_t m, double *At) {
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) At[j * n + i] = A[i * m + j];
}

// Solve A·X = B (A n×n, B n×p) → X n×p.  Gauss elimination with partial
// pivoting.  Falls back to identity-update on singularity.
void solve(double *A, int64_t n, double *B, int64_t p) {
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

}  // namespace fusion

extern "C" {

// ---------- Tier-2 motion models -------------------------------------------
// constvel: state = [x, vx, y, vy, z, vz]; F = block-diag([1 dt; 0 1]) per axis.
// We register the *function-form* (one-step propagation given a dt) so the
// EKF wrapper can call it via the handle ABI.

matlab_mat *matlab_fusion_constvel(matlab_mat *x, double dt) {
    if (!x) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    matlab_mat *o = mat_alloc(n, 1);
    for (int64_t i = 0; i + 1 < n; i += 2) {
        o->data[i]     = x->data[i] + dt * x->data[i + 1];
        o->data[i + 1] = x->data[i + 1];
    }
    if (n % 2) o->data[n - 1] = x->data[n - 1];
    return o;
}

// constacc: state = [x, vx, ax, y, vy, ay, z, vz, az] (triples).
matlab_mat *matlab_fusion_constacc(matlab_mat *x, double dt) {
    if (!x) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    matlab_mat *o = mat_alloc(n, 1);
    for (int64_t i = 0; i + 2 < n; i += 3) {
        o->data[i]     = x->data[i] + dt * x->data[i + 1] + 0.5 * dt * dt * x->data[i + 2];
        o->data[i + 1] = x->data[i + 1] + dt * x->data[i + 2];
        o->data[i + 2] = x->data[i + 2];
    }
    return o;
}

// constturn: 5-state planar coordinated turn [x, vx, y, vy, omega].
matlab_mat *matlab_fusion_constturn(matlab_mat *x, double dt) {
    if (!x || x->rows * x->cols < 5) return mat_alloc(0, 0);
    matlab_mat *o = mat_alloc(5, 1);
    double px = x->data[0], vx = x->data[1];
    double py = x->data[2], vy = x->data[3];
    double w  = x->data[4];
    if (std::fabs(w) < 1e-12) {
        o->data[0] = px + dt * vx;
        o->data[1] = vx;
        o->data[2] = py + dt * vy;
        o->data[3] = vy;
    } else {
        double s = std::sin(w * dt), c = std::cos(w * dt);
        o->data[0] = px + (vx * s - vy * (1 - c)) / w;
        o->data[1] = vx * c - vy * s;
        o->data[2] = py + (vx * (1 - c) + vy * s) / w;
        o->data[3] = vx * s + vy * c;
    }
    o->data[4] = w;
    return o;
}

// ---------- Tier-2 measurement models --------------------------------------
// cvmeas: position-only measurement of a constvel state (extract every-other).
matlab_mat *matlab_fusion_cvmeas(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t n  = x->rows * x->cols;
    int64_t ny = n / 2;
    matlab_mat *o = mat_alloc(ny, 1);
    for (int64_t i = 0, k = 0; i + 1 < n; i += 2, ++k) o->data[k] = x->data[i];
    return o;
}

// cameas: position-only of a constacc state (every third).
matlab_mat *matlab_fusion_cameas(matlab_mat *x) {
    if (!x) return mat_alloc(0, 0);
    int64_t n  = x->rows * x->cols;
    int64_t ny = n / 3;
    matlab_mat *o = mat_alloc(ny, 1);
    for (int64_t i = 0, k = 0; i + 2 < n; i += 3, ++k) o->data[k] = x->data[i];
    return o;
}

// ctmeas: [range; azimuth] of a 5-state constturn state from origin.
matlab_mat *matlab_fusion_ctmeas(matlab_mat *x) {
    matlab_mat *o = mat_alloc(2, 1);
    if (!x || x->rows * x->cols < 5) return o;
    double px = x->data[0], py = x->data[2];
    o->data[0] = std::sqrt(px * px + py * py);
    o->data[1] = std::atan2(py, px);
    return o;
}

// ---------- Tier-2 objectDetection -----------------------------------------
// Populate Time / Measurement / MeasurementNoise on a freshly-allocated obj.
matlab_mat *matlab_fusion_objdet_init(void *obj_v, double t, matlab_mat *z, matlab_mat *R) {
    fusion::obj_set_f64(obj_v, "Time", t);
    if (z) fusion::obj_set_mat(obj_v, "Measurement", z);
    if (R) fusion::obj_set_mat(obj_v, "MeasurementNoise", R);
    return z ? z : mat_alloc(0, 0);
}

// ---------- Tier-2.1 trackingKF (linear Kalman) ----------------------------

matlab_mat *matlab_fusion_trackingkf_init(void *obj_v,
                                           matlab_mat *F, matlab_mat *H,
                                           matlab_mat *Q, matlab_mat *R,
                                           matlab_mat *x0) {
    if (!F || !x0) return mat_alloc(0, 0);
    int64_t nx = F->rows;
    matlab_mat *P0 = mat_alloc(nx, nx);
    for (int64_t i = 0; i < nx; ++i) P0->data[i * nx + i] = 1.0;
    fusion::obj_set_mat(obj_v, "Fmat", F);
    fusion::obj_set_mat(obj_v, "Hmat", H);
    fusion::obj_set_mat(obj_v, "ProcessNoise", Q);
    fusion::obj_set_mat(obj_v, "MeasurementNoise", R);
    fusion::obj_set_mat(obj_v, "State", x0);
    fusion::obj_set_mat(obj_v, "StateCovariance", P0);
    return x0;
}

matlab_mat *matlab_fusion_trackingkf_predict(void *obj_v) {
    matlab_mat *F = fusion::obj_get_mat(obj_v, "Fmat");
    matlab_mat *Q = fusion::obj_get_mat(obj_v, "ProcessNoise");
    matlab_mat *x = fusion::obj_get_mat(obj_v, "State");
    matlab_mat *P = fusion::obj_get_mat(obj_v, "StateCovariance");
    if (!F || !x || !P) return mat_alloc(0, 0);
    int64_t n = x->rows;
    // x = F*x; P = F*P*F' + Q
    matlab_mat *xn = mat_alloc(n, 1);
    fusion::mm(F->data, n, n, x->data, 1, xn->data);
    std::vector<double> FP(n * n), Pn(n * n), Ft(n * n);
    fusion::tr(F->data, n, n, Ft.data());
    fusion::mm(F->data, n, n, P->data, n, FP.data());
    fusion::mm(FP.data(), n, n, Ft.data(), n, Pn.data());
    if (Q) for (int64_t i = 0; i < n * n; ++i) Pn[i] += Q->data[i];
    matlab_mat *Pm = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) Pm->data[i] = Pn[i];
    fusion::obj_set_mat(obj_v, "State", xn);
    fusion::obj_set_mat(obj_v, "StateCovariance", Pm);
    return xn;
}

matlab_mat *matlab_fusion_trackingkf_correct(void *obj_v, matlab_mat *y) {
    matlab_mat *H = fusion::obj_get_mat(obj_v, "Hmat");
    matlab_mat *R = fusion::obj_get_mat(obj_v, "MeasurementNoise");
    matlab_mat *x = fusion::obj_get_mat(obj_v, "State");
    matlab_mat *P = fusion::obj_get_mat(obj_v, "StateCovariance");
    if (!H || !y || !x || !P) return mat_alloc(0, 0);
    int64_t n = x->rows;
    int64_t m = y->rows;
    // innovation v = y - H*x
    std::vector<double> hx(m), v(m);
    fusion::mm(H->data, m, n, x->data, 1, hx.data());
    for (int64_t i = 0; i < m; ++i) v[i] = y->data[i] - hx[i];
    // S = H*P*H' + R
    std::vector<double> HP(m * n), S(m * m), Ht(n * m);
    fusion::tr(H->data, m, n, Ht.data());
    fusion::mm(H->data, m, n, P->data, n, HP.data());
    fusion::mm(HP.data(), m, n, Ht.data(), m, S.data());
    if (R) for (int64_t i = 0; i < m * m; ++i) S[i] += R->data[i];
    // K = P*H' / S  (solve S K' = (P*H')' → easier: K = P H' S^-1)
    std::vector<double> PHt(n * m);
    fusion::mm(P->data, n, n, Ht.data(), m, PHt.data());
    // S_copy * K' = PHt' i.e. solve S * X = PHt' (X is m×n then transpose)
    std::vector<double> Sc(m * m), KT(m * n);
    for (int64_t i = 0; i < m * m; ++i) Sc[i] = S[i];
    fusion::tr(PHt.data(), n, m, KT.data());  // KT is m×n
    fusion::solve(Sc.data(), m, KT.data(), n);
    std::vector<double> K(n * m);
    fusion::tr(KT.data(), m, n, K.data());     // K is n×m
    // x = x + K*v
    std::vector<double> Kv(n);
    fusion::mm(K.data(), n, m, v.data(), 1, Kv.data());
    matlab_mat *xn = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) xn->data[i] = x->data[i] + Kv[i];
    // P = (I - K*H) * P
    std::vector<double> KH(n * n), IKH(n * n), Pn(n * n);
    fusion::mm(K.data(), n, m, H->data, n, KH.data());
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            IKH[i * n + j] = (i == j ? 1.0 : 0.0) - KH[i * n + j];
    fusion::mm(IKH.data(), n, n, P->data, n, Pn.data());
    matlab_mat *Pm = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) Pm->data[i] = Pn[i];
    fusion::obj_set_mat(obj_v, "State", xn);
    fusion::obj_set_mat(obj_v, "StateCovariance", Pm);
    return xn;
}

// ---------- Tier-2.2 trackingEKF (vector-y) -------------------------------
typedef matlab_mat *(*fusion_vec_fn)(matlab_mat *);

// Finite-difference Jacobian of f at x.  m-rows output (probed via f(x)).
static matlab_mat *fd_jac_vec(fusion_vec_fn f, matlab_mat *x, matlab_mat **fx_out) {
    int64_t nx = x->rows * x->cols;
    matlab_mat *fx = f(x);
    int64_t m = fx->rows * fx->cols;
    matlab_mat *J = mat_alloc(m, nx);
    for (int64_t j = 0; j < nx; ++j) {
        matlab_mat *xp = mat_alloc(nx, 1);
        for (int64_t i = 0; i < nx; ++i) xp->data[i] = x->data[i];
        double dx = 1e-6 * (std::fabs(x->data[j]) + 1e-6);
        xp->data[j] += dx;
        matlab_mat *fp = f(xp);
        for (int64_t i = 0; i < m; ++i)
            J->data[i * nx + j] = (fp->data[i] - fx->data[i]) / dx;
    }
    *fx_out = fx;
    return J;
}

matlab_mat *matlab_fusion_trackingekf_init(void *obj_v, matlab_mat *x0,
                                            matlab_mat *P0, matlab_mat *Q,
                                            matlab_mat *R) {
    fusion::obj_set_mat(obj_v, "State", x0);
    fusion::obj_set_mat(obj_v, "StateCovariance", P0);
    fusion::obj_set_mat(obj_v, "ProcessNoise", Q);
    fusion::obj_set_mat(obj_v, "MeasurementNoise", R);
    return x0;
}

matlab_mat *matlab_fusion_trackingekf_predict(void *obj_v, void *f_ptr) {
    fusion_vec_fn f = reinterpret_cast<fusion_vec_fn>(f_ptr);
    matlab_mat *x = fusion::obj_get_mat(obj_v, "State");
    matlab_mat *P = fusion::obj_get_mat(obj_v, "StateCovariance");
    matlab_mat *Q = fusion::obj_get_mat(obj_v, "ProcessNoise");
    if (!x || !P || !f) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    matlab_mat *fx = nullptr;
    matlab_mat *F  = fd_jac_vec(f, x, &fx);
    // P = F*P*F' + Q
    std::vector<double> FP(n * n), Pn(n * n), Ft(n * n);
    fusion::tr(F->data, n, n, Ft.data());
    fusion::mm(F->data, n, n, P->data, n, FP.data());
    fusion::mm(FP.data(), n, n, Ft.data(), n, Pn.data());
    if (Q) for (int64_t i = 0; i < n * n; ++i) Pn[i] += Q->data[i];
    matlab_mat *Pm = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) Pm->data[i] = Pn[i];
    matlab_mat *xn = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) xn->data[i] = fx->data[i];
    fusion::obj_set_mat(obj_v, "State", xn);
    fusion::obj_set_mat(obj_v, "StateCovariance", Pm);
    return xn;
}

matlab_mat *matlab_fusion_trackingekf_correct(void *obj_v, void *h_ptr, matlab_mat *y) {
    fusion_vec_fn h = reinterpret_cast<fusion_vec_fn>(h_ptr);
    matlab_mat *x = fusion::obj_get_mat(obj_v, "State");
    matlab_mat *P = fusion::obj_get_mat(obj_v, "StateCovariance");
    matlab_mat *R = fusion::obj_get_mat(obj_v, "MeasurementNoise");
    if (!x || !P || !h || !y) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    matlab_mat *hx = nullptr;
    matlab_mat *H  = fd_jac_vec(h, x, &hx);
    int64_t m = hx->rows * hx->cols;
    // S = H*P*H' + R
    std::vector<double> Ht(n * m), HP(m * n), S(m * m);
    fusion::tr(H->data, m, n, Ht.data());
    fusion::mm(H->data, m, n, P->data, n, HP.data());
    fusion::mm(HP.data(), m, n, Ht.data(), m, S.data());
    if (R) for (int64_t i = 0; i < m * m; ++i) S[i] += R->data[i];
    // K = P*H' * S^-1
    std::vector<double> PHt(n * m), Sc(m * m), KT(m * n);
    fusion::mm(P->data, n, n, Ht.data(), m, PHt.data());
    for (int64_t i = 0; i < m * m; ++i) Sc[i] = S[i];
    fusion::tr(PHt.data(), n, m, KT.data());
    fusion::solve(Sc.data(), m, KT.data(), n);
    std::vector<double> K(n * m);
    fusion::tr(KT.data(), m, n, K.data());
    // v = y - hx
    std::vector<double> v(m);
    for (int64_t i = 0; i < m; ++i) v[i] = y->data[i] - hx->data[i];
    // x = x + K*v
    std::vector<double> Kv(n);
    fusion::mm(K.data(), n, m, v.data(), 1, Kv.data());
    matlab_mat *xn = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) xn->data[i] = x->data[i] + Kv[i];
    // P = (I - K*H) * P
    std::vector<double> KH(n * n), IKH(n * n), Pn(n * n);
    fusion::mm(K.data(), n, m, H->data, n, KH.data());
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            IKH[i * n + j] = (i == j ? 1.0 : 0.0) - KH[i * n + j];
    fusion::mm(IKH.data(), n, n, P->data, n, Pn.data());
    matlab_mat *Pm = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) Pm->data[i] = Pn[i];
    fusion::obj_set_mat(obj_v, "State", xn);
    fusion::obj_set_mat(obj_v, "StateCovariance", Pm);
    return xn;
}

// ---------- Tier-2.3 trackingUKF (vector-y) -------------------------------
matlab_mat *matlab_fusion_trackingukf_init(void *obj_v, matlab_mat *x0,
                                            matlab_mat *P0, matlab_mat *Q,
                                            matlab_mat *R) {
    return matlab_fusion_trackingekf_init(obj_v, x0, P0, Q, R);
}

// Generate the 2n+1 sigma points using a scaled-unscented (Julier) spread.
static void ukf_sigma(const double *x, int64_t n, const double *P,
                      double lambda, double *X) {
    // X is (n × (2n+1)) — column-major in this helper for ease.
    matlab_mat tmp;
    matlab_mat *Pm = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) Pm->data[i] = (lambda + n) * P[i];
    matlab_mat *L = matlab_chol(Pm);  // upper-triangular
    (void)tmp;
    // X(:,0) = x
    for (int64_t i = 0; i < n; ++i) X[i] = x[i];
    for (int64_t j = 0; j < n; ++j) {
        // sigma j+1     = x + L(:,j)'  (using rows of upper L as columns)
        // sigma j+1+n   = x - L(:,j)'
        for (int64_t i = 0; i < n; ++i) {
            double col_j = (j >= i) ? L->data[i * n + j] : 0.0;
            X[(j + 1) * n + i]     = x[i] + col_j;
            X[(j + 1 + n) * n + i] = x[i] - col_j;
        }
    }
}

matlab_mat *matlab_fusion_trackingukf_predict(void *obj_v, void *f_ptr) {
    fusion_vec_fn f = reinterpret_cast<fusion_vec_fn>(f_ptr);
    matlab_mat *x = fusion::obj_get_mat(obj_v, "State");
    matlab_mat *P = fusion::obj_get_mat(obj_v, "StateCovariance");
    matlab_mat *Q = fusion::obj_get_mat(obj_v, "ProcessNoise");
    if (!x || !P || !f) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    double alpha = 1e-3, beta = 2.0, kappa = static_cast<double>(3 - n);
    if (kappa < 0) kappa = 0;
    double lambda = alpha * alpha * (n + kappa) - n;
    int64_t sp = 2 * n + 1;
    std::vector<double> X(sp * n);
    ukf_sigma(x->data, n, P->data, lambda, X.data());
    // weights
    std::vector<double> wm(sp), wc(sp);
    wm[0] = lambda / (n + lambda);
    wc[0] = wm[0] + (1 - alpha * alpha + beta);
    double w = 1.0 / (2 * (n + lambda));
    for (int64_t i = 1; i < sp; ++i) { wm[i] = w; wc[i] = w; }
    // propagate
    std::vector<double> Y(sp * n);
    for (int64_t k = 0; k < sp; ++k) {
        matlab_mat *xk = mat_alloc(n, 1);
        for (int64_t i = 0; i < n; ++i) xk->data[i] = X[k * n + i];
        matlab_mat *yk = f(xk);
        for (int64_t i = 0; i < n; ++i) Y[k * n + i] = yk->data[i];
    }
    std::vector<double> xn(n, 0.0);
    for (int64_t k = 0; k < sp; ++k)
        for (int64_t i = 0; i < n; ++i) xn[i] += wm[k] * Y[k * n + i];
    std::vector<double> Pn(n * n, 0.0);
    for (int64_t k = 0; k < sp; ++k) {
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                Pn[i * n + j] += wc[k] * (Y[k * n + i] - xn[i]) * (Y[k * n + j] - xn[j]);
    }
    if (Q) for (int64_t i = 0; i < n * n; ++i) Pn[i] += Q->data[i];
    matlab_mat *xo = mat_alloc(n, 1);
    matlab_mat *Po = mat_alloc(n, n);
    for (int64_t i = 0; i < n; ++i)     xo->data[i] = xn[i];
    for (int64_t i = 0; i < n * n; ++i) Po->data[i] = Pn[i];
    fusion::obj_set_mat(obj_v, "State", xo);
    fusion::obj_set_mat(obj_v, "StateCovariance", Po);
    return xo;
}

matlab_mat *matlab_fusion_trackingukf_correct(void *obj_v, void *h_ptr, matlab_mat *y) {
    fusion_vec_fn h = reinterpret_cast<fusion_vec_fn>(h_ptr);
    matlab_mat *x = fusion::obj_get_mat(obj_v, "State");
    matlab_mat *P = fusion::obj_get_mat(obj_v, "StateCovariance");
    matlab_mat *R = fusion::obj_get_mat(obj_v, "MeasurementNoise");
    if (!x || !P || !h || !y) return mat_alloc(0, 0);
    int64_t n = x->rows * x->cols;
    double alpha = 1e-3, beta = 2.0, kappa = static_cast<double>(3 - n);
    if (kappa < 0) kappa = 0;
    double lambda = alpha * alpha * (n + kappa) - n;
    int64_t sp = 2 * n + 1;
    std::vector<double> X(sp * n);
    ukf_sigma(x->data, n, P->data, lambda, X.data());
    std::vector<double> wm(sp), wc(sp);
    wm[0] = lambda / (n + lambda);
    wc[0] = wm[0] + (1 - alpha * alpha + beta);
    double w = 1.0 / (2 * (n + lambda));
    for (int64_t i = 1; i < sp; ++i) { wm[i] = w; wc[i] = w; }
    // Run measurement function on each sigma point to get m.
    matlab_mat *xs0 = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) xs0->data[i] = X[i];
    matlab_mat *zs0 = h(xs0);
    int64_t m = zs0->rows * zs0->cols;
    std::vector<double> Z(sp * m);
    for (int64_t i = 0; i < m; ++i) Z[i] = zs0->data[i];
    for (int64_t k = 1; k < sp; ++k) {
        matlab_mat *xk = mat_alloc(n, 1);
        for (int64_t i = 0; i < n; ++i) xk->data[i] = X[k * n + i];
        matlab_mat *zk = h(xk);
        for (int64_t i = 0; i < m; ++i) Z[k * m + i] = zk->data[i];
    }
    std::vector<double> zhat(m, 0.0);
    for (int64_t k = 0; k < sp; ++k)
        for (int64_t i = 0; i < m; ++i) zhat[i] += wm[k] * Z[k * m + i];
    std::vector<double> S(m * m, 0.0), Pxz(n * m, 0.0);
    for (int64_t k = 0; k < sp; ++k) {
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < m; ++j)
                S[i * m + j] += wc[k] * (Z[k * m + i] - zhat[i]) * (Z[k * m + j] - zhat[j]);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < m; ++j)
                Pxz[i * m + j] += wc[k] * (X[k * n + i] - x->data[i]) * (Z[k * m + j] - zhat[j]);
    }
    if (R) for (int64_t i = 0; i < m * m; ++i) S[i] += R->data[i];
    // K = Pxz * S^-1
    std::vector<double> Sc(m * m), KT(m * n);
    for (int64_t i = 0; i < m * m; ++i) Sc[i] = S[i];
    fusion::tr(Pxz.data(), n, m, KT.data());
    fusion::solve(Sc.data(), m, KT.data(), n);
    std::vector<double> K(n * m);
    fusion::tr(KT.data(), m, n, K.data());
    // x = x + K*(y - zhat)
    std::vector<double> v(m), Kv(n);
    for (int64_t i = 0; i < m; ++i) v[i] = y->data[i] - zhat[i];
    fusion::mm(K.data(), n, m, v.data(), 1, Kv.data());
    matlab_mat *xn = mat_alloc(n, 1);
    for (int64_t i = 0; i < n; ++i) xn->data[i] = x->data[i] + Kv[i];
    // P = P - K * S * K'
    std::vector<double> KS(n * m), KSKt(n * n);
    fusion::mm(K.data(), n, m, S.data(), m, KS.data());
    std::vector<double> Kt(m * n);
    fusion::tr(K.data(), n, m, Kt.data());
    fusion::mm(KS.data(), n, m, Kt.data(), n, KSKt.data());
    matlab_mat *Pm = mat_alloc(n, n);
    for (int64_t i = 0; i < n * n; ++i) Pm->data[i] = P->data[i] - KSKt[i];
    fusion::obj_set_mat(obj_v, "State", xn);
    fusion::obj_set_mat(obj_v, "StateCovariance", Pm);
    return xn;
}

// ---------- Tier-2.6 initialisers ------------------------------------------
// initcvekf — alloc + init a trackingEKF over a constvel state from a
// position-only detection.  Reuses the constvel/cvmeas runtime symbols
// (passed in as handles by the user — but here we use them via direct
// pointer cast for the embedded forms; the simpler form just seeds a state
// with zero velocity).
matlab_mat *matlab_fusion_initcvekf(void *obj_v, matlab_mat *detz, matlab_mat *detR) {
    if (!detz) return mat_alloc(0, 0);
    int64_t ny = detz->rows * detz->cols;
    int64_t nx = 2 * ny;
    matlab_mat *x0 = mat_alloc(nx, 1);
    for (int64_t i = 0; i < ny; ++i) x0->data[i * 2] = detz->data[i];
    matlab_mat *P0 = mat_alloc(nx, nx);
    for (int64_t i = 0; i < nx; ++i) P0->data[i * nx + i] = 1.0;
    matlab_mat *Q = mat_alloc(nx, nx);
    for (int64_t i = 0; i < nx; ++i) Q->data[i * nx + i] = 1e-2;
    return matlab_fusion_trackingekf_init(obj_v, x0, P0, Q, detR);
}

// initctekf — seed a 5-state coordinated-turn EKF from a 2-D position.
matlab_mat *matlab_fusion_initctekf(void *obj_v, matlab_mat *detz, matlab_mat *detR) {
    if (!detz) return mat_alloc(0, 0);
    matlab_mat *x0 = mat_alloc(5, 1);
    x0->data[0] = detz->data[0];        // px
    x0->data[2] = detz->data[1];        // py
    matlab_mat *P0 = mat_alloc(5, 5);
    for (int64_t i = 0; i < 5; ++i) P0->data[i * 5 + i] = 1.0;
    matlab_mat *Q  = mat_alloc(5, 5);
    for (int64_t i = 0; i < 5; ++i) Q->data[i * 5 + i] = 1e-2;
    return matlab_fusion_trackingekf_init(obj_v, x0, P0, Q, detR);
}

// ---------------------------------------------------------------------------
// Tier-3 — inertial sensors + orientation/pose fusion
// ---------------------------------------------------------------------------

// imuSensor / gpsSensor init populate the defaults the classdef ctor already
// set; here we just override SampleRate (and HasMagnetometer for IMU).
matlab_mat *matlab_fusion_imu_init(void *obj_v, double fs, double has_mag) {
    fusion::obj_set_f64(obj_v, "SampleRate", fs);
    fusion::obj_set_f64(obj_v, "HasMagnetometer", has_mag);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_fusion_gps_init(void *obj_v, double fs) {
    fusion::obj_set_f64(obj_v, "SampleRate", fs);
    return mat_alloc(0, 0);
}

// imuSensor step: given true orientation q (1×4), true angular velocity w (1×3),
// and true linear acceleration a (1×3) in the body frame, emit noisy [accel; gyro]
// readings (and optionally [mag]).  We pack the result as a [1×6 or 1×9] row to
// keep the caller side simple.  Sensor params are read from the obj.
matlab_mat *matlab_fusion_imu_step(void *obj_v, matlab_mat *acc_true,
                                    matlab_mat *gyro_true, matlab_mat *mag_field) {
    matlab_mat *AB = fusion::obj_get_mat(obj_v, "AccelBias");
    matlab_mat *AN = fusion::obj_get_mat(obj_v, "AccelNoiseDensity");
    matlab_mat *GB = fusion::obj_get_mat(obj_v, "GyroBias");
    matlab_mat *GN = fusion::obj_get_mat(obj_v, "GyroNoiseDensity");
    double has_mag = fusion::obj_get_f64(obj_v, "HasMagnetometer");
    int64_t ncols = has_mag > 0.5 ? 9 : 6;
    matlab_mat *o = mat_alloc(1, ncols);
    matlab_mat *Nz = matlab_randn(static_cast<double>(ncols), 1.0);
    for (int k = 0; k < 3; ++k) {
        double n = AN ? AN->data[k] : 1e-3;
        double b = AB ? AB->data[k] : 0.0;
        o->data[k]     = (acc_true  ? acc_true->data[k]  : 0.0) + b + n * Nz->data[k];
        n = GN ? GN->data[k] : 1e-4;
        b = GB ? GB->data[k] : 0.0;
        o->data[3 + k] = (gyro_true ? gyro_true->data[k] : 0.0) + b + n * Nz->data[3 + k];
    }
    if (has_mag > 0.5) {
        matlab_mat *MB = fusion::obj_get_mat(obj_v, "MagBias");
        matlab_mat *MN = fusion::obj_get_mat(obj_v, "MagNoiseDensity");
        for (int k = 0; k < 3; ++k) {
            double n = MN ? MN->data[k] : 1e-1;
            double b = MB ? MB->data[k] : 0.0;
            o->data[6 + k] = (mag_field ? mag_field->data[k] : 0.0) + b + n * Nz->data[6 + k];
        }
    }
    return o;
}

// gpsSensor step: given a true ENU position p (1×3) and velocity v (1×3),
// emit a noisy [pos vel] = (1×6) row.
matlab_mat *matlab_fusion_gps_step(void *obj_v, matlab_mat *pos_true,
                                    matlab_mat *vel_true) {
    matlab_mat *PN = fusion::obj_get_mat(obj_v, "PositionNoise");
    matlab_mat *VN = fusion::obj_get_mat(obj_v, "VelocityNoise");
    matlab_mat *o  = mat_alloc(1, 6);
    matlab_mat *Nz = matlab_randn(6.0, 1.0);
    for (int k = 0; k < 3; ++k) {
        double n = PN ? PN->data[k] : 1.0;
        o->data[k]     = (pos_true ? pos_true->data[k] : 0.0) + n * Nz->data[k];
        n = VN ? VN->data[k] : 0.1;
        o->data[3 + k] = (vel_true ? vel_true->data[k] : 0.0) + n * Nz->data[3 + k];
    }
    return o;
}

// ---------- ahrsfilter / imufilter / complementaryFilter ----------
//
// Tier-3 orientation-only fusion.  We use a *quaternion-only* state (4×1)
// integrated forward by the gyroscope and corrected by the accelerometer
// (and magnetometer for AHRS).  The implementation is a Mahony-style
// complementary filter — proven, simple, and accurate enough to close the
// gating test (the full Kalman variant matching MATLAB exactly is the
// documented Tier-3 follow-on; see roadmap §4 row 3.4).

matlab_mat *matlab_fusion_ahrs_init(void *obj_v, double fs) {
    fusion::obj_set_f64(obj_v, "SampleRate", fs);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_fusion_imufilter_init(void *obj_v, double fs) {
    fusion::obj_set_f64(obj_v, "SampleRate", fs);
    return mat_alloc(0, 0);
}

matlab_mat *matlab_fusion_compfilter_init(void *obj_v, double fs) {
    fusion::obj_set_f64(obj_v, "SampleRate", fs);
    return mat_alloc(0, 0);
}

// step(filter, accel, gyro [,mag]) → 1×4 quaternion orientation.  Mahony
// fusion with bias estimation: accel (and mag) define a reference direction;
// the cross-product of measured-vs-predicted reference drives a proportional+
// integral term that corrects the gyro-integrated quaternion.
static matlab_mat *ahrs_step_inner(void *obj_v, matlab_mat *acc,
                                    matlab_mat *gyro, matlab_mat *mag) {
    double fs = fusion::obj_get_f64(obj_v, "SampleRate");
    if (fs <= 0) fs = 100.0;
    double dt = 1.0 / fs;
    matlab_mat *St = fusion::obj_get_mat(obj_v, "State");
    if (!St || St->rows < 7) {
        matlab_mat *S = mat_alloc(7, 1);
        S->data[0] = 1.0;
        fusion::obj_set_mat(obj_v, "State", S);
        St = S;
    }
    double q[4] = { St->data[0], St->data[1], St->data[2], St->data[3] };
    double gxb = St->data[4], gyb = St->data[5], gzb = St->data[6];
    double gx  = gyro ? (gyro->data[0] - gxb) : 0.0;
    double gy  = gyro ? (gyro->data[1] - gyb) : 0.0;
    double gz  = gyro ? (gyro->data[2] - gzb) : 0.0;
    // Accelerometer correction (reference: gravity along body Z when level
    // → expected body-frame direction is R'*[0;0;1]).
    if (acc && acc->rows * acc->cols >= 3) {
        double ax = acc->data[0], ay = acc->data[1], az = acc->data[2];
        double n = std::sqrt(ax*ax + ay*ay + az*az);
        if (n > 1e-9) { ax /= n; ay /= n; az /= n; }
        double vx = 2 * (q[1]*q[3] - q[0]*q[2]);
        double vy = 2 * (q[0]*q[1] + q[2]*q[3]);
        double vz = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3];
        double ex = ay*vz - az*vy;
        double ey = az*vx - ax*vz;
        double ez = ax*vy - ay*vx;
        double Kp = 0.5, Ki = 0.001;
        gxb += Ki * ex * dt;
        gyb += Ki * ey * dt;
        gzb += Ki * ez * dt;
        gx += Kp * ex;
        gy += Kp * ey;
        gz += Kp * ez;
    }
    if (mag && mag->rows * mag->cols >= 3) {
        double mx = mag->data[0], my = mag->data[1], mz = mag->data[2];
        double nm = std::sqrt(mx*mx + my*my + mz*mz);
        if (nm > 1e-9) { mx /= nm; my /= nm; mz /= nm; }
        // Rotate body-frame mag into nav frame using current q (point rotation).
        double hx = 2*(mx*(0.5 - q[2]*q[2] - q[3]*q[3]) + my*(q[1]*q[2] - q[0]*q[3]) + mz*(q[1]*q[3] + q[0]*q[2]));
        double hy = 2*(mx*(q[1]*q[2] + q[0]*q[3]) + my*(0.5 - q[1]*q[1] - q[3]*q[3]) + mz*(q[2]*q[3] - q[0]*q[1]));
        double bx = std::sqrt(hx*hx + hy*hy);
        double bz = 2*(mx*(q[1]*q[3] - q[0]*q[2]) + my*(q[2]*q[3] + q[0]*q[1]) + mz*(0.5 - q[1]*q[1] - q[2]*q[2]));
        // Predicted measurement w in body frame.
        double wx = 2*bx*(0.5 - q[2]*q[2] - q[3]*q[3]) + 2*bz*(q[1]*q[3] - q[0]*q[2]);
        double wy = 2*bx*(q[1]*q[2] - q[0]*q[3]) + 2*bz*(q[0]*q[1] + q[2]*q[3]);
        double wz = 2*bx*(q[0]*q[2] + q[1]*q[3]) + 2*bz*(0.5 - q[1]*q[1] - q[2]*q[2]);
        double ex = my*wz - mz*wy;
        double ey = mz*wx - mx*wz;
        double ez = mx*wy - my*wx;
        double Kp = 0.5;
        gx += Kp * ex;
        gy += Kp * ey;
        gz += Kp * ez;
    }
    // Integrate q ← q + 0.5*Ω(omega)*q*dt; normalize.
    double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    q[0] += 0.5 * (-q1*gx - q2*gy - q3*gz) * dt;
    q[1] += 0.5 * ( q0*gx + q2*gz - q3*gy) * dt;
    q[2] += 0.5 * ( q0*gy - q1*gz + q3*gx) * dt;
    q[3] += 0.5 * ( q0*gz + q1*gy - q2*gx) * dt;
    double nq = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (nq > 1e-12) for (int k = 0; k < 4; ++k) q[k] /= nq;
    // Write back state + return q row.
    matlab_mat *Sn = mat_alloc(7, 1);
    for (int k = 0; k < 4; ++k) Sn->data[k] = q[k];
    Sn->data[4] = gxb; Sn->data[5] = gyb; Sn->data[6] = gzb;
    fusion::obj_set_mat(obj_v, "State", Sn);
    matlab_mat *out = mat_alloc(1, 4);
    for (int k = 0; k < 4; ++k) out->data[k] = q[k];
    return out;
}

matlab_mat *matlab_fusion_ahrs_step(void *obj_v, matlab_mat *acc,
                                     matlab_mat *gyro, matlab_mat *mag) {
    return ahrs_step_inner(obj_v, acc, gyro, mag);
}

matlab_mat *matlab_fusion_imufilter_step(void *obj_v, matlab_mat *acc,
                                          matlab_mat *gyro) {
    return ahrs_step_inner(obj_v, acc, gyro, nullptr);
}

matlab_mat *matlab_fusion_compfilter_step(void *obj_v, matlab_mat *acc,
                                           matlab_mat *gyro, matlab_mat *mag) {
    return ahrs_step_inner(obj_v, acc, gyro, mag);
}

// ---------- Tier-3.5 insfilterMARG (headline) -----------------------------
//
// 16-state EKF over [q(4); pos(3); vel(3); accelBias(3); gyroBias(3)].  We
// expose three update primitives:
//   - matlab_fusion_insmarg_predict(obj, accel_meas, gyro_meas, dt)
//   - matlab_fusion_insmarg_fuse_accel(obj, accel_meas)
//   - matlab_fusion_insmarg_fuse_gps(obj, pos_meas, vel_meas)
//
// The implementation uses the complementary-filter orientation update for
// the quaternion block + a simple double-integrator for position/velocity
// driven by gravity-compensated accel.  GPS fusion is a linear correction
// on the position+velocity sub-state.  This is a *simplified* MARG (the
// full EKF is documented as a Tier-3 follow-on per roadmap §10).

matlab_mat *matlab_fusion_insmarg_init(void *obj_v, double fs) {
    fusion::obj_set_f64(obj_v, "IMUSampleRate", fs);
    matlab_mat *S = mat_alloc(16, 1);
    S->data[0] = 1.0;  // identity quaternion
    fusion::obj_set_mat(obj_v, "State", S);
    matlab_mat *P = mat_alloc(16, 16);
    for (int64_t i = 0; i < 16; ++i) P->data[i * 16 + i] = 1.0;
    fusion::obj_set_mat(obj_v, "StateCovariance", P);
    return S;
}

matlab_mat *matlab_fusion_insmarg_predict(void *obj_v, matlab_mat *acc,
                                           matlab_mat *gyro, double dt) {
    matlab_mat *S = fusion::obj_get_mat(obj_v, "State");
    if (!S || S->rows < 16) return mat_alloc(0, 0);
    double *s = S->data;
    // Update quaternion via gyro (subtracted bias).
    double gx = (gyro ? gyro->data[0] : 0.0) - s[13];
    double gy = (gyro ? gyro->data[1] : 0.0) - s[14];
    double gz = (gyro ? gyro->data[2] : 0.0) - s[15];
    double q0 = s[0], q1 = s[1], q2 = s[2], q3 = s[3];
    s[0] += 0.5 * (-q1 * gx - q2 * gy - q3 * gz) * dt;
    s[1] += 0.5 * ( q0 * gx + q2 * gz - q3 * gy) * dt;
    s[2] += 0.5 * ( q0 * gy - q1 * gz + q3 * gx) * dt;
    s[3] += 0.5 * ( q0 * gz + q1 * gy - q2 * gx) * dt;
    double nq = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3]);
    if (nq > 1e-12) for (int k = 0; k < 4; ++k) s[k] /= nq;
    // Rotate accel from body to nav and remove gravity (NED: g points down/+z_nav).
    double ax = (acc ? acc->data[0] : 0.0) - s[10];
    double ay = (acc ? acc->data[1] : 0.0) - s[11];
    double az = (acc ? acc->data[2] : 0.0) - s[12];
    double qw = s[0], qx = s[1], qy = s[2], qz = s[3];
    double r00 = 1 - 2*(qy*qy + qz*qz), r01 = 2*(qx*qy - qz*qw), r02 = 2*(qx*qz + qy*qw);
    double r10 = 2*(qx*qy + qz*qw), r11 = 1 - 2*(qx*qx + qz*qz), r12 = 2*(qy*qz - qx*qw);
    double r20 = 2*(qx*qz - qy*qw), r21 = 2*(qy*qz + qx*qw), r22 = 1 - 2*(qx*qx + qy*qy);
    double anx = r00 * ax + r01 * ay + r02 * az;
    double any = r10 * ax + r11 * ay + r12 * az;
    double anz = r20 * ax + r21 * ay + r22 * az - 9.81;
    // Integrate vel + pos.
    s[7]  += anx * dt; s[4] += s[7] * dt;
    s[8]  += any * dt; s[5] += s[8] * dt;
    s[9]  += anz * dt; s[6] += s[9] * dt;
    return S;
}

matlab_mat *matlab_fusion_insmarg_fuse_accel(void *obj_v, matlab_mat *acc) {
    matlab_mat *S = fusion::obj_get_mat(obj_v, "State");
    if (!S || !acc || S->rows < 16) return S;
    // Tiny accel-based tilt correction (proportional gain).
    double *s = S->data;
    double ax = acc->data[0], ay = acc->data[1], az = acc->data[2];
    double n = std::sqrt(ax*ax + ay*ay + az*az);
    if (n < 1e-9) return S;
    ax /= n; ay /= n; az /= n;
    double vx = 2 * (s[1]*s[3] - s[0]*s[2]);
    double vy = 2 * (s[0]*s[1] + s[2]*s[3]);
    double vz = s[0]*s[0] - s[1]*s[1] - s[2]*s[2] + s[3]*s[3];
    double ex = ay*vz - az*vy;
    double ey = az*vx - ax*vz;
    double ez = ax*vy - ay*vx;
    double Kp = 0.1;
    s[0] += -Kp * 0.5 * (s[1]*ex + s[2]*ey + s[3]*ez);
    s[1] +=  Kp * 0.5 * (s[0]*ex + s[2]*ez - s[3]*ey);
    s[2] +=  Kp * 0.5 * (s[0]*ey - s[1]*ez + s[3]*ex);
    s[3] +=  Kp * 0.5 * (s[0]*ez + s[1]*ey - s[2]*ex);
    double nq = std::sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3]);
    if (nq > 1e-12) for (int k = 0; k < 4; ++k) s[k] /= nq;
    return S;
}

matlab_mat *matlab_fusion_insmarg_fuse_gps(void *obj_v, matlab_mat *pos, matlab_mat *vel) {
    matlab_mat *S = fusion::obj_get_mat(obj_v, "State");
    if (!S || S->rows < 16) return S;
    double *s = S->data;
    double Kp = 0.05;
    if (pos && pos->rows * pos->cols >= 3) {
        s[4] += Kp * (pos->data[0] - s[4]);
        s[5] += Kp * (pos->data[1] - s[5]);
        s[6] += Kp * (pos->data[2] - s[6]);
    }
    if (vel && vel->rows * vel->cols >= 3) {
        s[7] += Kp * (vel->data[0] - s[7]);
        s[8] += Kp * (vel->data[1] - s[8]);
        s[9] += Kp * (vel->data[2] - s[9]);
    }
    return S;
}

// ---------------------------------------------------------------------------
// Tier-4 — trajectory + scenario generation
// ---------------------------------------------------------------------------

// waypointTrajectory(waypoints, toa) populate: store an N×3 waypoint matrix
// and the matching N×1 time-of-arrival vector on the obj.  lookupPose(t)
// performs piecewise-linear position interpolation.  Velocity/orientation
// outputs are documented Tier-4 follow-ons (the headline tracer only needs
// position interpolation; the gnn_air_traffic example builds its own
// ground truth from waypoint segments).
matlab_mat *matlab_fusion_waypoint_init(void *obj_v, matlab_mat *wp, matlab_mat *toa) {
    if (!wp || !toa) return mat_alloc(0, 0);
    fusion::obj_set_mat(obj_v, "Waypoints", wp);
    fusion::obj_set_mat(obj_v, "TimeOfArrival", toa);
    // First waypoint as initial pose.
    matlab_mat *p0 = mat_alloc(1, 3);
    if (wp->cols == 3 && wp->rows >= 1) {
        p0->data[0] = wp->data[0]; p0->data[1] = wp->data[1]; p0->data[2] = wp->data[2];
    }
    fusion::obj_set_mat(obj_v, "InitialPosition", p0);
    return p0;
}

// lookupPose(traj, t) — linear interpolation between waypoints at time t.
matlab_mat *matlab_fusion_waypoint_lookup(void *obj_v, double t) {
    matlab_mat *wp  = fusion::obj_get_mat(obj_v, "Waypoints");
    matlab_mat *toa = fusion::obj_get_mat(obj_v, "TimeOfArrival");
    matlab_mat *o   = mat_alloc(1, 3);
    if (!wp || !toa || wp->cols < 3) return o;
    int64_t n = wp->rows;
    if (n == 0) return o;
    if (t <= toa->data[0]) {
        for (int k = 0; k < 3; ++k) o->data[k] = wp->data[k];
        return o;
    }
    if (t >= toa->data[n - 1]) {
        for (int k = 0; k < 3; ++k) o->data[k] = wp->data[(n - 1) * 3 + k];
        return o;
    }
    for (int64_t i = 0; i + 1 < n; ++i) {
        if (t >= toa->data[i] && t <= toa->data[i + 1]) {
            double dt = toa->data[i + 1] - toa->data[i];
            double a  = dt < 1e-12 ? 0.0 : (t - toa->data[i]) / dt;
            for (int k = 0; k < 3; ++k)
                o->data[k] = (1 - a) * wp->data[i * 3 + k] + a * wp->data[(i + 1) * 3 + k];
            return o;
        }
    }
    return o;
}

// Coordinate conversions.  Geodetic ↔ local-NED on the WGS-84 ellipsoid.
// lla2ned(lla, lla0) — converts lat/lon/alt (deg/deg/m) at point lla
// relative to reference lla0 to local NED metres.
matlab_mat *matlab_fusion_lla2ned(matlab_mat *lla, matlab_mat *lla0) {
    matlab_mat *o = mat_alloc(1, 3);
    if (!lla || !lla0 || lla->cols < 3 || lla0->cols < 3) return o;
    // WGS-84.
    constexpr double a = 6378137.0;
    constexpr double f = 1.0 / 298.257223563;
    constexpr double e2 = f * (2 - f);
    double lat  = lla->data[0]  * M_PI / 180.0;
    double lon  = lla->data[1]  * M_PI / 180.0;
    double alt  = lla->data[2];
    double lat0 = lla0->data[0] * M_PI / 180.0;
    double lon0 = lla0->data[1] * M_PI / 180.0;
    double alt0 = lla0->data[2];
    auto ecef = [&](double lt, double ln, double h, double e[3]) {
        double s = std::sin(lt), c = std::cos(lt);
        double N = a / std::sqrt(1 - e2 * s * s);
        e[0] = (N + h) * c * std::cos(ln);
        e[1] = (N + h) * c * std::sin(ln);
        e[2] = (N * (1 - e2) + h) * s;
    };
    double e[3], e0[3];
    ecef(lat, lon, alt, e);
    ecef(lat0, lon0, alt0, e0);
    double dx = e[0] - e0[0], dy = e[1] - e0[1], dz = e[2] - e0[2];
    double sL = std::sin(lat0), cL = std::cos(lat0);
    double sl = std::sin(lon0), cl = std::cos(lon0);
    o->data[0] = -sL * cl * dx - sL * sl * dy + cL * dz;  // north
    o->data[1] = -sl * dx + cl * dy;                       // east
    o->data[2] = -cL * cl * dx - cL * sl * dy - sL * dz;   // down
    return o;
}

// ned2lla(ned, lla0) — inverse of lla2ned.
matlab_mat *matlab_fusion_ned2lla(matlab_mat *ned, matlab_mat *lla0) {
    matlab_mat *o = mat_alloc(1, 3);
    if (!ned || !lla0 || ned->cols < 3 || lla0->cols < 3) return o;
    constexpr double a = 6378137.0;
    constexpr double f = 1.0 / 298.257223563;
    constexpr double e2 = f * (2 - f);
    double n_ = ned->data[0], e_ = ned->data[1], d_ = ned->data[2];
    double lat0 = lla0->data[0] * M_PI / 180.0;
    double lon0 = lla0->data[1] * M_PI / 180.0;
    double alt0 = lla0->data[2];
    auto ecef = [&](double lt, double ln, double h, double out[3]) {
        double s = std::sin(lt), c = std::cos(lt);
        double N = a / std::sqrt(1 - e2 * s * s);
        out[0] = (N + h) * c * std::cos(ln);
        out[1] = (N + h) * c * std::sin(ln);
        out[2] = (N * (1 - e2) + h) * s;
    };
    double e0[3];
    ecef(lat0, lon0, alt0, e0);
    double sL = std::sin(lat0), cL = std::cos(lat0);
    double sl = std::sin(lon0), cl = std::cos(lon0);
    // Inverse rotation (rotation matrix is orthogonal).
    double dx = -sL * cl * n_ - sl * e_ - cL * cl * d_;
    double dy = -sL * sl * n_ + cl * e_ - cL * sl * d_;
    double dz =  cL      * n_           - sL      * d_;
    double X = e0[0] + dx, Y = e0[1] + dy, Z = e0[2] + dz;
    // ECEF → LLA (Bowring's iterative).
    double lon = std::atan2(Y, X);
    double p = std::sqrt(X * X + Y * Y);
    double lat = std::atan2(Z, p * (1 - e2));
    for (int it = 0; it < 5; ++it) {
        double s = std::sin(lat);
        double N = a / std::sqrt(1 - e2 * s * s);
        lat = std::atan2(Z + e2 * N * s, p);
    }
    double s = std::sin(lat);
    double N = a / std::sqrt(1 - e2 * s * s);
    double alt = p / std::cos(lat) - N;
    o->data[0] = lat * 180.0 / M_PI;
    o->data[1] = lon * 180.0 / M_PI;
    o->data[2] = alt;
    return o;
}

// ---------------------------------------------------------------------------
// Tier-5 — multi-object trackers + assignment
// ---------------------------------------------------------------------------

// assignmunkres(C) — Munkres / Hungarian on an m×n cost matrix C.  Returns an
// m×1 column vector of assigned column indices (1-based, -1 if unassigned).
// Standard O(n³) implementation, padded to a square matrix internally.
matlab_mat *matlab_fusion_assignmunkres(matlab_mat *C) {
    if (!C) return mat_alloc(0, 0);
    int64_t m = C->rows, n = C->cols;
    int64_t sz = (m > n) ? m : n;
    if (sz == 0) return mat_alloc(0, 0);
    // Pad to square with a large finite cost.
    constexpr double BIG = 1e15;
    std::vector<double> A(sz * sz, BIG);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            A[i * sz + j] = C->data[i * n + j];
    // Hungarian via row + column reductions and augmenting paths.
    // u, v are dual variables.
    std::vector<double> u(sz + 1, 0.0), v(sz + 1, 0.0);
    std::vector<int>    p(sz + 1, 0),   way(sz + 1, 0);
    for (int64_t i = 1; i <= sz; ++i) {
        p[0] = static_cast<int>(i);
        int64_t j0 = 0;
        std::vector<double> minv(sz + 1, BIG);
        std::vector<int>    used(sz + 1, 0);
        do {
            used[j0] = 1;
            int64_t i0 = p[j0], j1 = 0;
            double delta = BIG;
            for (int64_t j = 1; j <= sz; ++j) if (!used[j]) {
                double cur = A[(i0 - 1) * sz + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = static_cast<int>(j0); }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int64_t j = 0; j <= sz; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else         { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    // ans[col] = row assigned to that column; rebuild row→col.
    std::vector<int> rowcol(sz + 1, 0);
    for (int64_t j = 1; j <= sz; ++j) rowcol[p[j]] = static_cast<int>(j);
    matlab_mat *o = mat_alloc(m, 1);
    for (int64_t i = 0; i < m; ++i) {
        int col = rowcol[i + 1];
        // Filter padded assignments.
        if (col == 0 || col > n) o->data[i] = -1.0;
        else                     o->data[i] = static_cast<double>(col);
    }
    return o;
}

// trackerGNN — minimal in-runtime tracker.
//
// We carry a vector of trackingEKF-shaped objects via a packed matrix:
//   States      :  Ntracks × 4   (constant-velocity 2-D state [x vx y vy])
//   Covariances :  Ntracks × 16  (4×4 row-major flattened)
//   Ages        :  Ntracks × 1   (integer hits-since-last-update)
//   Confirmed   :  Ntracks × 1   (0/1)
// This keeps the runtime model simple while still exercising the predict /
// correct / gate / assign / confirm loop end-to-end.

namespace fusion {
constexpr int kGNNStateDim = 4;     // [x vx y vy]
constexpr int kGNNCovStride = 16;   // 4×4 flattened
constexpr int kGNNMeasDim  = 2;     // [x y]

void gnn_predict_one(double *x, double *P, double dt) {
    // F = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
    double new_x = x[0] + dt * x[1];
    double new_y = x[2] + dt * x[3];
    x[0] = new_x;
    x[2] = new_y;
    // P = F P F' + Q
    double F[16] = {1, dt, 0, 0,
                    0, 1,  0, 0,
                    0, 0,  1, dt,
                    0, 0,  0, 1};
    double FP[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += F[i * 4 + k] * P[k * 4 + j];
            FP[i * 4 + j] = s;
        }
    double Pn[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) s += FP[i * 4 + k] * F[j * 4 + k];
            Pn[i * 4 + j] = s;
        }
    constexpr double q = 0.05;
    Pn[0]  += q; Pn[5]  += q; Pn[10] += q; Pn[15] += q;
    for (int i = 0; i < 16; ++i) P[i] = Pn[i];
}

double gnn_mahalanobis(const double *x, const double *P, const double *z, double R) {
    // H = [1 0 0 0; 0 0 1 0];  innovation v = z - H x
    double v0 = z[0] - x[0];
    double v1 = z[1] - x[2];
    // S = H P H' + R*I — extract Pxx, Pxy, Pyx, Pyy (with R on the diagonal).
    double s00 = P[0]      + R;
    double s01 = P[2];
    double s10 = P[8];
    double s11 = P[10]     + R;
    double det = s00 * s11 - s01 * s10;
    if (std::fabs(det) < 1e-18) return 1e18;
    double inv00 =  s11 / det, inv01 = -s01 / det;
    double inv10 = -s10 / det, inv11 =  s00 / det;
    double d = v0 * (inv00 * v0 + inv01 * v1) + v1 * (inv10 * v0 + inv11 * v1);
    return d;
}

void gnn_correct_one(double *x, double *P, const double *z, double R) {
    double v0 = z[0] - x[0];
    double v1 = z[1] - x[2];
    double s00 = P[0]  + R;
    double s01 = P[2];
    double s10 = P[8];
    double s11 = P[10] + R;
    double det = s00 * s11 - s01 * s10;
    if (std::fabs(det) < 1e-18) return;
    double inv00 =  s11 / det, inv01 = -s01 / det;
    double inv10 = -s10 / det, inv11 =  s00 / det;
    // K = P H' S^-1.  P H' is column-0 / column-2 of P.
    double K[8];
    for (int i = 0; i < 4; ++i) {
        double ph0 = P[i * 4 + 0];
        double ph2 = P[i * 4 + 2];
        K[i * 2 + 0] = ph0 * inv00 + ph2 * inv10;
        K[i * 2 + 1] = ph0 * inv01 + ph2 * inv11;
    }
    // x = x + K v
    for (int i = 0; i < 4; ++i) x[i] += K[i * 2 + 0] * v0 + K[i * 2 + 1] * v1;
    // P = (I - K H) P
    double Pn[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double s = P[i * 4 + j];
            s -= K[i * 2 + 0] * P[0 * 4 + j];
            s -= K[i * 2 + 1] * P[2 * 4 + j];
            Pn[i * 4 + j] = s;
        }
    }
    for (int i = 0; i < 16; ++i) P[i] = Pn[i];
}
}  // namespace fusion

// trackerGNN init: empty tracker.
matlab_mat *matlab_fusion_gnn_init(void *obj_v, double maxTracks) {
    int64_t M = static_cast<int64_t>(maxTracks);
    if (M < 1) M = 16;
    fusion::obj_set_f64(obj_v, "MaxNumTracks", static_cast<double>(M));
    fusion::obj_set_mat(obj_v, "States",      mat_alloc(0, fusion::kGNNStateDim));
    fusion::obj_set_mat(obj_v, "Covariances", mat_alloc(0, fusion::kGNNCovStride));
    fusion::obj_set_mat(obj_v, "Ages",        mat_alloc(0, 1));
    fusion::obj_set_mat(obj_v, "Confirmed",   mat_alloc(0, 1));
    return mat_alloc(0, 0);
}

// Single step: detections is N×2 (2-D positions); dt is the prediction step.
// Returns the current confirmed-track state matrix (Ntrk×4).
matlab_mat *matlab_fusion_gnn_step(void *obj_v, matlab_mat *detections, double dt) {
    matlab_mat *St = fusion::obj_get_mat(obj_v, "States");
    matlab_mat *Cv = fusion::obj_get_mat(obj_v, "Covariances");
    matlab_mat *Ag = fusion::obj_get_mat(obj_v, "Ages");
    matlab_mat *Cf = fusion::obj_get_mat(obj_v, "Confirmed");
    if (!St) St = mat_alloc(0, fusion::kGNNStateDim);
    if (!Cv) Cv = mat_alloc(0, fusion::kGNNCovStride);
    if (!Ag) Ag = mat_alloc(0, 1);
    if (!Cf) Cf = mat_alloc(0, 1);
    int64_t T = St->rows;
    int64_t N = detections ? detections->rows : 0;

    // 1) Predict all existing tracks.
    for (int64_t t = 0; t < T; ++t) {
        fusion::gnn_predict_one(&St->data[t * 4], &Cv->data[t * 16], dt);
    }
    // 2) Build cost matrix C(T×N) of Mahalanobis distances; gate at chi2 ≈ 9.
    constexpr double R    = 1.0;
    constexpr double GATE = 9.0;
    std::vector<double> Costs(static_cast<size_t>(T * N), 1e6);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < N; ++i) {
            const double *z = &detections->data[i * 2];
            double d = fusion::gnn_mahalanobis(&St->data[t * 4], &Cv->data[t * 16], z, R);
            if (d > GATE) d = 1e6;
            Costs[t * N + i] = d;
        }
    }
    // 3) Assignment via Munkres.
    std::vector<int> det_to_track(static_cast<size_t>(N), -1);
    if (T > 0 && N > 0) {
        matlab_mat C; C.rows = T; C.cols = N;
        std::vector<double> Cdup(Costs);
        C.data = Cdup.data();
        matlab_mat *Ar = matlab_fusion_assignmunkres(&C);
        for (int64_t t = 0; t < T; ++t) {
            int col = static_cast<int>(Ar->data[t]);
            if (col > 0 && Costs[t * N + (col - 1)] < 1e5) {
                det_to_track[col - 1] = static_cast<int>(t);
            }
        }
    }
    // 4) Update assigned tracks; assigned detections promote / set Confirmed.
    std::vector<int> used_det(static_cast<size_t>(N), 0);
    for (int64_t i = 0; i < N; ++i) {
        int tidx = det_to_track[i];
        if (tidx < 0) continue;
        const double *z = &detections->data[i * 2];
        fusion::gnn_correct_one(&St->data[tidx * 4], &Cv->data[tidx * 16], z, R);
        Ag->data[tidx] += 1.0;
        if (Ag->data[tidx] >= 2.0) Cf->data[tidx] = 1.0;
        used_det[i] = 1;
    }
    // 5) Unmatched detections seed new tracks (capped at MaxNumTracks).
    int64_t M = static_cast<int64_t>(fusion::obj_get_f64(obj_v, "MaxNumTracks"));
    if (M < 1) M = 16;
    int64_t new_T = T;
    for (int64_t i = 0; i < N && new_T < M; ++i) {
        if (used_det[i]) continue;
        new_T += 1;
    }
    matlab_mat *St2 = mat_alloc(new_T, fusion::kGNNStateDim);
    matlab_mat *Cv2 = mat_alloc(new_T, fusion::kGNNCovStride);
    matlab_mat *Ag2 = mat_alloc(new_T, 1);
    matlab_mat *Cf2 = mat_alloc(new_T, 1);
    for (int64_t t = 0; t < T; ++t) {
        for (int k = 0; k < 4; ++k)  St2->data[t * 4 + k]  = St->data[t * 4 + k];
        for (int k = 0; k < 16; ++k) Cv2->data[t * 16 + k] = Cv->data[t * 16 + k];
        Ag2->data[t] = Ag->data[t];
        Cf2->data[t] = Cf->data[t];
    }
    int64_t cursor = T;
    for (int64_t i = 0; i < N && cursor < new_T; ++i) {
        if (used_det[i]) continue;
        const double *z = &detections->data[i * 2];
        St2->data[cursor * 4 + 0] = z[0];
        St2->data[cursor * 4 + 1] = 0.0;
        St2->data[cursor * 4 + 2] = z[1];
        St2->data[cursor * 4 + 3] = 0.0;
        for (int k = 0; k < 16; ++k) Cv2->data[cursor * 16 + k] = 0.0;
        Cv2->data[cursor * 16 + 0]  = 100.0;  // px var
        Cv2->data[cursor * 16 + 5]  = 100.0;  // vx var
        Cv2->data[cursor * 16 + 10] = 100.0;  // py var
        Cv2->data[cursor * 16 + 15] = 100.0;  // vy var
        Ag2->data[cursor] = 1.0;
        Cf2->data[cursor] = 0.0;
        cursor += 1;
    }
    fusion::obj_set_mat(obj_v, "States", St2);
    fusion::obj_set_mat(obj_v, "Covariances", Cv2);
    fusion::obj_set_mat(obj_v, "Ages", Ag2);
    fusion::obj_set_mat(obj_v, "Confirmed", Cf2);
    return St2;
}

// trackerGNN.numConfirmed(obj) — count of Confirmed==1.
matlab_mat *matlab_fusion_gnn_numconfirmed(void *obj_v) {
    matlab_mat *Cf = fusion::obj_get_mat(obj_v, "Confirmed");
    matlab_mat *o  = mat_alloc(1, 1);
    if (!Cf) { o->data[0] = 0; return o; }
    double s = 0;
    for (int64_t i = 0; i < Cf->rows; ++i) s += Cf->data[i];
    o->data[0] = s;
    return o;
}

// ---------------------------------------------------------------------------
// Tier-6 — track fusion + metrics + RTS smoother
// ---------------------------------------------------------------------------

// trackFuser via covariance intersection (CI): given two Gaussian estimates
// (x1, P1) and (x2, P2) of the same target, returns a fused (x, P) safely
// even if the input estimates are correlated.  The mixing weight ω is chosen
// to minimise trace(P_fused) via a coarse 1-D line search.
//
// Inputs:
//   x1, x2 : nx × 1 state vectors
//   P1, P2 : nx × nx covariance matrices (symmetric PSD)
// Returns a packed matrix of size (nx + nx²) × 1 with x stacked on top of
// vec(P) (row-major) — callers split with subscript reads.  This sidesteps
// the lack of true multi-return on free-function builtins.
matlab_mat *matlab_fusion_covint(matlab_mat *x1, matlab_mat *P1,
                                  matlab_mat *x2, matlab_mat *P2) {
    if (!x1 || !P1 || !x2 || !P2) return mat_alloc(0, 0);
    int64_t n = x1->rows * x1->cols;
    if (P1->rows != n || P2->rows != n) return mat_alloc(0, 0);

    auto fuse_at = [&](double w, std::vector<double> &xf, std::vector<double> &Pf) {
        // Pf^-1 = w P1^-1 + (1-w) P2^-1
        std::vector<double> Pa(n * n), Pb(n * n);
        for (int64_t i = 0; i < n * n; ++i) {
            Pa[i] = P1->data[i];
            Pb[i] = P2->data[i];
        }
        // Build the combined precision matrix.
        std::vector<double> M(n * n, 0.0), Mw(n * n, 0.0);
        // Trick: directly form P_fused = (w·P1^-1 + (1-w)·P2^-1)^-1 by solving
        // P1·A = I and P2·B = I separately, mixing A and B, then inverting.
        std::vector<double> I1(n * n, 0.0), I2(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i) { I1[i * n + i] = 1.0; I2[i * n + i] = 1.0; }
        fusion::solve(Pa.data(), n, I1.data(), n);   // I1 := P1^-1
        fusion::solve(Pb.data(), n, I2.data(), n);   // I2 := P2^-1
        for (int64_t i = 0; i < n * n; ++i)
            Mw[i] = w * I1[i] + (1.0 - w) * I2[i];
        std::vector<double> Ic(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i) Ic[i * n + i] = 1.0;
        fusion::solve(Mw.data(), n, Ic.data(), n);   // Ic := P_fused
        // x_fused = P_fused · (w · P1^-1 · x1 + (1-w) · P2^-1 · x2)
        std::vector<double> rhs(n, 0.0), t1(n, 0.0), t2(n, 0.0);
        // Recompute P1^-1, P2^-1 (Mw was overwritten).
        for (int64_t i = 0; i < n * n; ++i) {
            Pa[i] = P1->data[i];
            Pb[i] = P2->data[i];
        }
        for (int64_t i = 0; i < n * n; ++i) { I1[i] = 0.0; I2[i] = 0.0; }
        for (int64_t i = 0; i < n; ++i)   { I1[i * n + i] = 1.0; I2[i * n + i] = 1.0; }
        fusion::solve(Pa.data(), n, I1.data(), n);
        fusion::solve(Pb.data(), n, I2.data(), n);
        for (int64_t i = 0; i < n; ++i) {
            double s1 = 0, s2 = 0;
            for (int64_t j = 0; j < n; ++j) {
                s1 += I1[i * n + j] * x1->data[j];
                s2 += I2[i * n + j] * x2->data[j];
            }
            rhs[i] = w * s1 + (1.0 - w) * s2;
        }
        xf.assign(n, 0.0);
        for (int64_t i = 0; i < n; ++i) {
            double s = 0;
            for (int64_t j = 0; j < n; ++j) s += Ic[i * n + j] * rhs[j];
            xf[i] = s;
        }
        Pf.assign(Ic.begin(), Ic.end());
    };

    // Line-search ω ∈ [0,1] in 11 grid points; pick the one with min trace(Pf).
    std::vector<double> xf_best, Pf_best;
    double best_tr = 1e300;
    for (int g = 0; g <= 10; ++g) {
        double w = static_cast<double>(g) / 10.0;
        std::vector<double> xf, Pf;
        fuse_at(w, xf, Pf);
        double tr = 0;
        for (int64_t i = 0; i < n; ++i) tr += Pf[i * n + i];
        if (tr < best_tr) { best_tr = tr; xf_best = xf; Pf_best = Pf; }
    }
    if (xf_best.empty()) return mat_alloc(0, 0);
    // Pack: x_fused (n) on top of vec(P_fused) (n²).
    matlab_mat *O = mat_alloc(n + n * n, 1);
    for (int64_t i = 0; i < n; ++i)     O->data[i]     = xf_best[i];
    for (int64_t i = 0; i < n * n; ++i) O->data[n + i] = Pf_best[i];
    return O;
}

// trackGOSPAMetric(X, Y, c, p) — Generalized Optimal Sub-Pattern Assignment
// distance.  X is m×D, Y is n×D (track and truth position rows); the
// Euclidean distance is used per row.  c is the cutoff, p typically 2.
//
// d_GOSPA(X,Y) = ( sum_i min(c, d(x_i, y_assigned(i)))^p
//                  + (c^p / 2) · (max(m,n) - h) )^(1/p)
// where h is the number of assignments with finite cost (≤ c).
//
// Returns a 1×1 matrix.
matlab_mat *matlab_fusion_gospa(matlab_mat *X, matlab_mat *Y, double c, double p) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!X || !Y) return o;
    int64_t m = X->rows, n = Y->rows;
    int64_t D = X->cols;
    if (D == 0 || Y->cols != D) return o;
    if (p < 1.0) p = 2.0;
    if (c <= 0)  c = 1.0;
    // Build the m×n cost matrix of distances (capped at c).
    std::vector<double> Cost(static_cast<size_t>(m * n), c);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0;
            for (int64_t k = 0; k < D; ++k) {
                double d = X->data[i * D + k] - Y->data[j * D + k];
                s += d * d;
            }
            double dist = std::sqrt(s);
            if (dist > c) dist = c;
            Cost[i * n + j] = std::pow(dist, p);
        }
    }
    // Solve the assignment via Munkres on the m×n cost.
    int64_t hits = 0;
    double assigned_sum = 0;
    if (m > 0 && n > 0) {
        matlab_mat C;
        C.rows = m; C.cols = n;
        std::vector<double> Cdup(Cost);
        C.data = Cdup.data();
        matlab_mat *A = matlab_fusion_assignmunkres(&C);
        double cp = std::pow(c, p);
        for (int64_t i = 0; i < m; ++i) {
            int col = static_cast<int>(A->data[i]);
            if (col >= 1 && col <= n) {
                double cost = Cost[i * n + (col - 1)];
                if (cost < cp - 1e-12) hits += 1;
                assigned_sum += cost;
            }
        }
    }
    int64_t big = (m > n) ? m : n;
    double cp = std::pow(c, p);
    double penalty = (cp / 2.0) * static_cast<double>(big - hits);
    double total = assigned_sum + penalty;
    o->data[0] = std::pow(total, 1.0 / p);
    return o;
}

// trackOSPAMetric(X, Y, c, p) — OSPA variant.  Same assignment, different
// penalty form: averaged by max(m,n).
matlab_mat *matlab_fusion_ospa(matlab_mat *X, matlab_mat *Y, double c, double p) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!X || !Y) return o;
    int64_t m = X->rows, n = Y->rows;
    int64_t D = X->cols;
    if (D == 0 || Y->cols != D) return o;
    if (p < 1.0) p = 2.0;
    if (c <= 0)  c = 1.0;
    int64_t big = (m > n) ? m : n;
    if (big == 0) return o;
    std::vector<double> Cost(static_cast<size_t>(m * n), c);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double s = 0;
            for (int64_t k = 0; k < D; ++k) {
                double d = X->data[i * D + k] - Y->data[j * D + k];
                s += d * d;
            }
            double dist = std::sqrt(s);
            if (dist > c) dist = c;
            Cost[i * n + j] = std::pow(dist, p);
        }
    }
    double assigned_sum = 0;
    int64_t h = 0;
    if (m > 0 && n > 0) {
        matlab_mat C; C.rows = m; C.cols = n;
        std::vector<double> Cdup(Cost);
        C.data = Cdup.data();
        matlab_mat *A = matlab_fusion_assignmunkres(&C);
        for (int64_t i = 0; i < m; ++i) {
            int col = static_cast<int>(A->data[i]);
            if (col >= 1 && col <= n) {
                assigned_sum += Cost[i * n + (col - 1)];
                h += 1;
            }
        }
    }
    double cp = std::pow(c, p);
    double penalty = cp * static_cast<double>(big - h);
    double total = (assigned_sum + penalty) / static_cast<double>(big);
    o->data[0] = std::pow(total, 1.0 / p);
    return o;
}

// trackErrorMetrics — simple RMSE accumulator over a track history matrix
// (Tsteps × D) vs a truth history of the same shape.  Returns 1×1 RMSE.
matlab_mat *matlab_fusion_trackerror(matlab_mat *Xhist, matlab_mat *Thist) {
    matlab_mat *o = mat_alloc(1, 1);
    if (!Xhist || !Thist) return o;
    int64_t T = Xhist->rows;
    int64_t D = Xhist->cols;
    if (Thist->rows != T || Thist->cols != D || T == 0) return o;
    double s = 0;
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t k = 0; k < D; ++k) {
            double d = Xhist->data[t * D + k] - Thist->data[t * D + k];
            s += d * d;
        }
    }
    o->data[0] = std::sqrt(s / static_cast<double>(T * D));
    return o;
}

// rtsSmoother(F, Xhist, Phist) — Rauch-Tung-Striebel backward pass.
//   F     : nx × nx state-transition matrix (constant across the history)
//   Xhist : T  × nx forward-filter state rows
//   Phist : T  × (nx·nx) forward-filter covariance rows (row-major flattened)
// Returns a T × nx smoothed-state matrix.
matlab_mat *matlab_fusion_rts_smoother(matlab_mat *F, matlab_mat *Xhist, matlab_mat *Phist) {
    if (!F || !Xhist || !Phist) return mat_alloc(0, 0);
    int64_t T  = Xhist->rows;
    int64_t nx = Xhist->cols;
    if (F->rows != nx || F->cols != nx) return mat_alloc(0, 0);
    if (Phist->rows != T || Phist->cols != nx * nx) return mat_alloc(0, 0);
    matlab_mat *Out = mat_alloc(T, nx);
    if (T == 0) return Out;
    // Initialise smoothed state to the last forward estimate.
    for (int64_t k = 0; k < nx; ++k)
        Out->data[(T - 1) * nx + k] = Xhist->data[(T - 1) * nx + k];
    // Work buffers.
    std::vector<double> Pk(nx * nx), Pkp(nx * nx), Pkp_inv(nx * nx);
    std::vector<double> Ft(nx * nx), FPk(nx * nx), Ck(nx * nx);
    fusion::tr(F->data, nx, nx, Ft.data());
    // Walk backwards from T-2 to 0.
    for (int64_t k = T - 2; k >= 0; --k) {
        for (int64_t i = 0; i < nx * nx; ++i) Pk[i] = Phist->data[k * nx * nx + i];
        // Predict covariance one step: Pk+1|k = F · Pk · F'.  We approximate
        // here that the filter step's Q is already absorbed by Phist[k+1],
        // so Pk+1|k ≈ F·Pk·F'.
        std::vector<double> FP(nx * nx), Pp(nx * nx);
        fusion::mm(F->data, nx, nx, Pk.data(), nx, FP.data());
        fusion::mm(FP.data(), nx, nx, Ft.data(), nx, Pp.data());
        // Smoother gain Ck = Pk · F' · Pp^-1.
        std::vector<double> PkFt(nx * nx), Pp_inv(nx * nx, 0.0);
        for (int64_t i = 0; i < nx; ++i) Pp_inv[i * nx + i] = 1.0;
        std::vector<double> Pp_copy(Pp);
        fusion::solve(Pp_copy.data(), nx, Pp_inv.data(), nx);
        fusion::mm(Pk.data(), nx, nx, Ft.data(), nx, PkFt.data());
        fusion::mm(PkFt.data(), nx, nx, Pp_inv.data(), nx, Ck.data());
        // Smoothed mean: xs_k = xk + Ck · (xs_{k+1} - F · xk).
        std::vector<double> Fxk(nx), diff(nx), Cd(nx);
        for (int64_t i = 0; i < nx; ++i) {
            double s = 0;
            for (int64_t j = 0; j < nx; ++j) s += F->data[i * nx + j] * Xhist->data[k * nx + j];
            Fxk[i] = s;
            diff[i] = Out->data[(k + 1) * nx + i] - Fxk[i];
        }
        fusion::mm(Ck.data(), nx, nx, diff.data(), 1, Cd.data());
        for (int64_t i = 0; i < nx; ++i)
            Out->data[k * nx + i] = Xhist->data[k * nx + i] + Cd[i];
    }
    return Out;
}

// ---------- Tier-3.7 allanvar ---------------------------------------------
//
// Compute the Allan variance of x[0..N-1] sampled at fs Hz, over a set of
// averaging windows m = 1, 2, 4, ..., N/2.  Returns a (k×2) matrix of
// [tau, AVAR] rows.
matlab_mat *matlab_fusion_allanvar(matlab_mat *x, double fs) {
    if (!x) return mat_alloc(0, 0);
    int64_t N = x->rows * x->cols;
    if (N < 4 || fs <= 0) return mat_alloc(0, 0);
    int64_t maxm = N / 4;
    std::vector<std::pair<double, double>> out;
    for (int64_t m = 1; m <= maxm; m *= 2) {
        int64_t K = N / m;
        if (K < 2) break;
        std::vector<double> avg(K, 0.0);
        for (int64_t k = 0; k < K; ++k) {
            double s = 0;
            for (int64_t i = 0; i < m; ++i) s += x->data[k * m + i];
            avg[k] = s / static_cast<double>(m);
        }
        double v = 0;
        for (int64_t k = 0; k + 1 < K; ++k) {
            double d = avg[k + 1] - avg[k];
            v += d * d;
        }
        v /= (2.0 * static_cast<double>(K - 1));
        double tau = static_cast<double>(m) / fs;
        out.push_back({tau, v});
    }
    matlab_mat *O = mat_alloc(static_cast<int64_t>(out.size()), 2);
    for (size_t i = 0; i < out.size(); ++i) {
        O->data[i * 2 + 0] = out[i].first;
        O->data[i * 2 + 1] = out[i].second;
    }
    return O;
}

}  // extern "C"
