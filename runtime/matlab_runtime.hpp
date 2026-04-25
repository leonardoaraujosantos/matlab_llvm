#ifndef MATLAB_RUNTIME_HPP
#define MATLAB_RUNTIME_HPP

// C++ wrapper for the matlab runtime. The full C ABI lives in
// matlab_runtime.h (typed `matlab_mat *`); this header instead
// forward-declares the same functions with `void *` so the signatures
// line up with emit-cpp's pointer convention. The runtime is binary-
// compatible either way.

#include <initializer_list>
#include <ostream>
#include <stdint.h>

extern "C" {
// The subset of the runtime exercised by the Matrix wrapper. Non-matrix
// runtime functions (matlab_disp_str, matlab_obj_new, …) are still
// declared by the per-module `extern "C"` block emit-cpp generates.
void *matlab_mat_from_buf(const double *buf, double m, double n);
void  matlab_disp_mat(void *A);
void *matlab_matmul_mm(void *A, void *B);
void *matlab_add_mm(void *A, void *B);
void *matlab_sub_mm(void *A, void *B);
void *matlab_emul_mm(void *A, void *B);
void *matlab_ediv_mm(void *A, void *B);
void *matlab_epow_mm(void *A, void *B);
void *matlab_add_ms(void *A, double s);
void *matlab_sub_ms(void *A, double s);
void *matlab_emul_ms(void *A, double s);
void *matlab_ediv_ms(void *A, double s);
void *matlab_add_sm(double s, void *A);
void *matlab_sub_sm(double s, void *A);
void *matlab_emul_sm(double s, void *A);
void *matlab_ediv_sm(double s, void *A);
void *matlab_transpose(void *A);
void *matlab_inv(void *A);
void *matlab_diag(void *A);
void *matlab_mldivide_mm(void *A, void *B);
void *matlab_mrdivide_mm(void *A, void *B);
void *matlab_eig(void *A);
void *matlab_eig_V(void *A);
void *matlab_eig_D(void *A);
void *matlab_svd(void *A);
void *matlab_neg_m(void *A);
void *matlab_reshape(void *A, double m, double n);
void *matlab_repmat(void *A, double m, double n);
void *matlab_matpow(void *A, double n);
void *matlab_sum(void *A);
void *matlab_prod(void *A);
void *matlab_mean(void *A);
void *matlab_min(void *A);
void *matlab_max(void *A);
void *matlab_sqrt_m(void *A);
void *matlab_abs_m(void *A);
void *matlab_exp_m(void *A);
void *matlab_log_m(void *A);
void *matlab_sin_m(void *A);
void *matlab_cos_m(void *A);
void *matlab_tan_m(void *A);
double matlab_numel(void *A);
double matlab_length(void *A);
double matlab_det(void *A);
} // extern "C"

class Matrix {
public:
  Matrix() : p_(nullptr) {}
  Matrix(void *p) : p_(p) {}
  Matrix(const double *buf, double m, double n)
      : p_(matlab_mat_from_buf(buf, m, n)) {}
  // Convenience ctor for the `Matrix A((void*)slot, m, n)` shape emit-cpp
  // produces — the initial `(void*)` cast came from the tensor lowering's
  // alloca trampoline.
  Matrix(void *buf, double m, double n)
      : p_(matlab_mat_from_buf(static_cast<const double *>(buf), m, n)) {}
  Matrix(std::initializer_list<double> data, double m, double n)
      : p_(matlab_mat_from_buf(data.begin(), m, n)) {}

  operator void *() const { return p_; }
  void *raw() const { return p_; }

  Matrix operator+(const Matrix &b) const { return matlab_add_mm(p_, b.p_); }
  Matrix operator-(const Matrix &b) const { return matlab_sub_mm(p_, b.p_); }
  Matrix operator*(const Matrix &b) const { return matlab_matmul_mm(p_, b.p_); }
  Matrix operator+(double s) const { return matlab_add_ms(p_, s); }
  Matrix operator-(double s) const { return matlab_sub_ms(p_, s); }
  Matrix operator*(double s) const { return matlab_emul_ms(p_, s); }
  Matrix operator/(double s) const { return matlab_ediv_ms(p_, s); }

  Matrix emul(const Matrix &b) const { return matlab_emul_mm(p_, b.p_); }
  Matrix ediv(const Matrix &b) const { return matlab_ediv_mm(p_, b.p_); }
  Matrix epow(const Matrix &b) const { return matlab_epow_mm(p_, b.p_); }
  Matrix t() const { return matlab_transpose(p_); }
  Matrix inv() const { return matlab_inv(p_); }
  Matrix diag() const { return matlab_diag(p_); }
  Matrix mldivide(const Matrix &b) const { return matlab_mldivide_mm(p_, b.p_); }
  Matrix mrdivide(const Matrix &b) const { return matlab_mrdivide_mm(p_, b.p_); }
  Matrix eig() const { return matlab_eig(p_); }
  Matrix eigV() const { return matlab_eig_V(p_); }
  Matrix eigD() const { return matlab_eig_D(p_); }
  Matrix svd() const { return matlab_svd(p_); }
  Matrix neg() const { return matlab_neg_m(p_); }
  Matrix reshape(double m, double n) const { return matlab_reshape(p_, m, n); }
  Matrix repmat(double m, double n) const { return matlab_repmat(p_, m, n); }
  Matrix pow(double n) const { return matlab_matpow(p_, n); }
  Matrix operator-() const { return matlab_neg_m(p_); }

  Matrix sum() const { return matlab_sum(p_); }
  Matrix prod() const { return matlab_prod(p_); }
  Matrix mean() const { return matlab_mean(p_); }
  Matrix min() const { return matlab_min(p_); }
  Matrix max() const { return matlab_max(p_); }

  Matrix sqrt() const { return matlab_sqrt_m(p_); }
  Matrix abs() const { return matlab_abs_m(p_); }
  Matrix exp() const { return matlab_exp_m(p_); }
  Matrix log() const { return matlab_log_m(p_); }
  Matrix sin() const { return matlab_sin_m(p_); }
  Matrix cos() const { return matlab_cos_m(p_); }
  Matrix tan() const { return matlab_tan_m(p_); }

  double numel() const { return matlab_numel(p_); }
  double length() const { return matlab_length(p_); }
  double det() const { return matlab_det(p_); }

  friend std::ostream &operator<<(std::ostream &os, const Matrix &m) {
    matlab_disp_mat(m.p_);
    return os;
  }

private:
  void *p_;
};

inline Matrix operator+(double s, const Matrix &m) {
  return matlab_add_sm(s, m.raw());
}
inline Matrix operator-(double s, const Matrix &m) {
  return matlab_sub_sm(s, m.raw());
}
inline Matrix operator*(double s, const Matrix &m) {
  return matlab_emul_sm(s, m.raw());
}
inline Matrix operator/(double s, const Matrix &m) {
  return matlab_ediv_sm(s, m.raw());
}

#endif // MATLAB_RUNTIME_HPP
