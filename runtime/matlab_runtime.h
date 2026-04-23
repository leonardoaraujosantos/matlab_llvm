#ifndef MATLAB_RUNTIME_H
#define MATLAB_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types — the layout lives in matlab_runtime.c; generated code only
// ever passes pointers to these.
typedef struct matlab_mat    matlab_mat;
typedef struct matlab_struct_s matlab_struct;
typedef struct matlab_cell_s   matlab_cell;

// Parfor callback.
typedef void (*matlab_parfor_body_t)(double iv, void *state);

// I/O.
void matlab_disp_str(const char *s, int64_t n);
void matlab_disp_f64(double v);
void matlab_disp_vec_f64(const double *data, int64_t n);
void matlab_disp_mat_f64(const double *data, int64_t m, int64_t n);
void matlab_disp_mat(matlab_mat *A);
void matlab_fprintf_str(const char *fmt, int64_t n);
void matlab_fprintf_f64(const char *fmt, int64_t n, double v);
void matlab_fprintf_f64_2(const char *fmt, int64_t n, double a, double b);
void matlab_fprintf_f64_3(const char *fmt, int64_t n,
                          double a, double b, double c);
void matlab_fprintf_f64_4(const char *fmt, int64_t n,
                          double a, double b, double c, double d);
double matlab_input_num(const char *prompt, int64_t plen);

// Parallel / reductions.
void matlab_parfor_dispatch(double start, double step, double end,
                            matlab_parfor_body_t body, void *state);
void matlab_reduce_add_f64(double *ptr, double delta);

// Matrix constructors.
matlab_mat *matlab_mat_from_buf(const double *buf, double m, double n);
matlab_mat *matlab_mat_from_scalar(double x);
matlab_mat *matlab_empty_mat(void);
matlab_mat *matlab_zeros(double m, double n);
matlab_mat *matlab_ones(double m, double n);
matlab_mat *matlab_eye(double m, double n);
matlab_mat *matlab_magic(double nd);
matlab_mat *matlab_rand(double m, double n);
matlab_mat *matlab_randn(double m, double n);
matlab_mat *matlab_range(double start, double step, double end);
matlab_mat *matlab_repmat(matlab_mat *A, double m, double n);

// Linear algebra.
matlab_mat *matlab_matmul_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_inv(matlab_mat *A);
matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_mrdivide_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_svd(matlab_mat *A_in);
matlab_mat *matlab_eig(matlab_mat *A_in);
matlab_mat *matlab_eig_V(matlab_mat *A_in);
matlab_mat *matlab_eig_D(matlab_mat *A_in);
double      matlab_det(matlab_mat *A);
matlab_mat *matlab_transpose(matlab_mat *A);
matlab_mat *matlab_diag(matlab_mat *A);
matlab_mat *matlab_reshape(matlab_mat *A, double m, double n);
matlab_mat *matlab_matpow(matlab_mat *A, double n);

// Element-wise binary ops (matrix/matrix, matrix/scalar, scalar/matrix).
matlab_mat *matlab_add_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_sub_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_emul_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ediv_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_epow_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_add_ms(matlab_mat *A, double s);
matlab_mat *matlab_sub_ms(matlab_mat *A, double s);
matlab_mat *matlab_emul_ms(matlab_mat *A, double s);
matlab_mat *matlab_ediv_ms(matlab_mat *A, double s);
matlab_mat *matlab_epow_ms(matlab_mat *A, double s);
matlab_mat *matlab_add_sm(double s, matlab_mat *A);
matlab_mat *matlab_sub_sm(double s, matlab_mat *A);
matlab_mat *matlab_emul_sm(double s, matlab_mat *A);
matlab_mat *matlab_ediv_sm(double s, matlab_mat *A);
matlab_mat *matlab_epow_sm(double s, matlab_mat *A);

// Element-wise comparisons (return 0/1 matrices).
matlab_mat *matlab_gt_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ge_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_lt_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_le_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_eq_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_ne_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_gt_ms(matlab_mat *A, double s);
matlab_mat *matlab_ge_ms(matlab_mat *A, double s);
matlab_mat *matlab_lt_ms(matlab_mat *A, double s);
matlab_mat *matlab_le_ms(matlab_mat *A, double s);
matlab_mat *matlab_eq_ms(matlab_mat *A, double s);
matlab_mat *matlab_ne_ms(matlab_mat *A, double s);
matlab_mat *matlab_gt_sm(double s, matlab_mat *A);
matlab_mat *matlab_ge_sm(double s, matlab_mat *A);
matlab_mat *matlab_lt_sm(double s, matlab_mat *A);
matlab_mat *matlab_le_sm(double s, matlab_mat *A);
matlab_mat *matlab_eq_sm(double s, matlab_mat *A);
matlab_mat *matlab_ne_sm(double s, matlab_mat *A);

// Element-wise unary ops on matrices.
matlab_mat *matlab_neg_m(matlab_mat *A);
matlab_mat *matlab_exp_m(matlab_mat *A);
matlab_mat *matlab_log_m(matlab_mat *A);
matlab_mat *matlab_sin_m(matlab_mat *A);
matlab_mat *matlab_cos_m(matlab_mat *A);
matlab_mat *matlab_tan_m(matlab_mat *A);
matlab_mat *matlab_sqrt_m(matlab_mat *A);
matlab_mat *matlab_abs_m(matlab_mat *A);

// Column-wise / scalar reductions.
matlab_mat *matlab_sum(matlab_mat *A);
matlab_mat *matlab_prod(matlab_mat *A);
matlab_mat *matlab_mean(matlab_mat *A);
matlab_mat *matlab_min(matlab_mat *A);
matlab_mat *matlab_max(matlab_mat *A);
matlab_mat *matlab_min_mm(matlab_mat *A, matlab_mat *B);
matlab_mat *matlab_max_mm(matlab_mat *A, matlab_mat *B);

// Shape / predicates.
matlab_mat *matlab_size(matlab_mat *A);
double matlab_size_dim(matlab_mat *A, double dim);
double matlab_length(matlab_mat *A);
double matlab_numel(matlab_mat *A);
double matlab_ndims(matlab_mat *A);
double matlab_end_of_dim(matlab_mat *A, double dim);
double matlab_isempty(matlab_mat *A);
double matlab_isequal(matlab_mat *A, matlab_mat *B);

// Subscripting.
double      matlab_subscript1_s(matlab_mat *A, double i);
double      matlab_subscript2_s(matlab_mat *A, double i, double j);
matlab_mat *matlab_slice1(matlab_mat *A, matlab_mat *idx);
matlab_mat *matlab_slice2(matlab_mat *A, matlab_mat *rows, matlab_mat *cols);
void matlab_slice_store1(matlab_mat *A, matlab_mat *idx, matlab_mat *V);
void matlab_slice_store1_scalar(matlab_mat *A, matlab_mat *idx, double v);
void matlab_slice_store2(matlab_mat *A, matlab_mat *rows, matlab_mat *cols,
                         matlab_mat *V);
void matlab_slice_store2_scalar(matlab_mat *A, matlab_mat *rows,
                                matlab_mat *cols, double v);
matlab_mat *matlab_find(matlab_mat *A);
matlab_mat *matlab_erase_rows(matlab_mat *A, matlab_mat *rows);
matlab_mat *matlab_erase_cols(matlab_mat *A, matlab_mat *cols);

// Scalar math builtins.
double matlab_exp_s(double x);
double matlab_log_s(double x);
double matlab_sin_s(double x);
double matlab_cos_s(double x);
double matlab_tan_s(double x);
double matlab_sqrt_s(double x);
double matlab_abs_s(double x);

// Try/catch error flag.
void    matlab_set_error(void);
int32_t matlab_check_error(void);
void    matlab_clear_error(void);

// Structs.
matlab_struct *matlab_struct_new(void);
void matlab_struct_set_f64(matlab_struct *s, const char *name, int64_t len,
                           double v);
void matlab_struct_set_mat(matlab_struct *s, const char *name, int64_t len,
                           matlab_mat *m);
double matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len);
matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name,
                                  int64_t len);
double matlab_struct_has_field(matlab_struct *s, const char *name, int64_t len);
matlab_struct *matlab_struct_get_child_struct(matlab_struct *s,
                                              const char *name, int64_t len);

// Cells.
matlab_cell *matlab_cell_new(double n);
void matlab_cell_set_f64(matlab_cell *c, double i1, double v);
void matlab_cell_set_mat(matlab_cell *c, double i1, matlab_mat *m);
double matlab_cell_get_f64(matlab_cell *c, double i1);
matlab_mat *matlab_cell_get_mat(matlab_cell *c, double i1);
double matlab_cell_numel(matlab_cell *c);
double matlab_iscell(matlab_cell *c);

// Global / persistent.
double matlab_global_get_f64(int32_t id);
void   matlab_global_set_f64(int32_t id, double v);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MATLAB_RUNTIME_H
