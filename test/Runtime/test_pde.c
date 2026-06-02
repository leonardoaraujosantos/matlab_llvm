/* test_pde.c — direct C runtime test for runtime_pde.cpp.
 *
 * Bypasses the MATLAB frontend / MLIR / JIT layers — calls the
 * runtime entries directly so we can debug the numerical core
 * without any lowering shenanigans.  See docs/pde_toolbox_roadmap.md.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Public ABI from matlab_runtime.h — but we don't include it because
 * it pulls in declarations that conflict with the runtime's internal
 * types.  Forward-declare the bits we need. */
typedef struct matlab_mat {
    double *data;
    int64_t rows;
    int64_t cols;
} matlab_mat;

typedef struct matlab_struct_s matlab_struct;

extern matlab_mat *mat_alloc(int64_t m, int64_t n);
extern matlab_mat *matlab_pde_mesh_rect_tri(double x0, double x1,
                                            double y0, double y1,
                                            double Nx, double Ny);
extern matlab_mat *matlab_pde_boundary_nodes_rect(matlab_struct *mesh);
extern matlab_struct *matlab_pde_assemble_poisson_2d(matlab_struct *mesh,
                                                    double c, double a, double f);
extern matlab_struct *matlab_pde_apply_dirichlet(matlab_struct *sys,
                                                 matlab_mat *node_ids, double u_val);
extern matlab_mat *matlab_pde_sys_K(matlab_struct *sys);
extern matlab_mat *matlab_pde_sys_F(matlab_struct *sys);

extern matlab_struct *matlab_pde_mesh_cuboid_tet(double W, double D, double H,
                                                  double Nx, double Ny, double Nz);
extern matlab_mat *matlab_pde_face_nodes(matlab_struct *mesh, double face_id);
extern matlab_mat *matlab_pde_assemble_elast_3d(matlab_struct *mesh,
                                                double E, double nu);
extern matlab_mat *matlab_pde_face_pressure_3d(matlab_struct *mesh,
                                                double face_id, double pressure);
extern matlab_struct *matlab_pde_apply_fixed_3d(matlab_mat *K, matlab_mat *F,
                                                matlab_mat *node_ids);
extern matlab_mat *matlab_pde_mesh_nodes(matlab_struct *mesh);
extern matlab_mat *matlab_pde_mesh_tets(matlab_struct *mesh);
extern matlab_mat *matlab_pde_mesh_faces(matlab_struct *mesh);
extern matlab_mat *matlab_pde_mesh_triangles(matlab_struct *mesh);
extern matlab_mat *matlab_pde_von_mises_3d(matlab_struct *mesh, matlab_mat *u,
                                            double E, double nu);
extern double matlab_pde_peak_disp_3d(matlab_mat *u);

/* Dense linear solve via LU with partial pivoting — copied from the
 * runtime since we don't want to pull in matlab_runtime.h. */
extern matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B);

/* #122 regression: a model reaching the structural femodel fallback
 * without a MaterialProperties struct. */
extern matlab_struct *matlab_struct_new(void);
extern void matlab_struct_set_mat(matlab_struct *s, const char *name,
                                  int64_t len, matlab_mat *m);
extern matlab_struct *matlab_pde_solve_femodel(matlab_struct *model);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); } \
} while (0)

static double vec_max_abs(matlab_mat *v) {
    double m = 0.0;
    int64_t n = v->rows * v->cols;
    for (int64_t i = 0; i < n; ++i) {
        double a = v->data[i] < 0 ? -v->data[i] : v->data[i];
        if (a > m) m = a;
    }
    return m;
}

static double vec_max(matlab_mat *v) {
    double m = -1e300;
    int64_t n = v->rows * v->cols;
    for (int64_t i = 0; i < n; ++i) {
        if (v->data[i] > m) m = v->data[i];
    }
    return m;
}

static void test_tier1_poisson_square(void) {
    printf("  test_tier1_poisson_square:\n");
    matlab_struct *mesh = (matlab_struct *)matlab_pde_mesh_rect_tri(0, 1, 0, 1, 21, 21);
    matlab_mat *bnd  = matlab_pde_boundary_nodes_rect(mesh);
    printf("    Nbnd = %lld\n", (long long)bnd->rows);
    matlab_struct *sys  = matlab_pde_assemble_poisson_2d(mesh, 1.0, 0.0, 1.0);
    matlab_mat *K0 = matlab_pde_sys_K(sys);
    matlab_mat *F0 = matlab_pde_sys_F(sys);
    printf("    K is %lld x %lld\n", (long long)K0->rows, (long long)K0->cols);
    printf("    F is %lld x %lld\n", (long long)F0->rows, (long long)F0->cols);

    matlab_struct *sys2 = matlab_pde_apply_dirichlet(sys, bnd, 0.0);
    matlab_mat *K = matlab_pde_sys_K(sys2);
    matlab_mat *F = matlab_pde_sys_F(sys2);
    matlab_mat *u = matlab_mldivide_mm(K, F);

    /* Centre node is (i=10, j=10) → idx = 10*21 + 10 = 220 (0-based). */
    double u_center = u->data[220];
    printf("    u(0.5, 0.5) = %.6f (analytic ≈ 0.0737)\n", u_center);
    CHECK(u_center > 0.07 && u_center < 0.08, "u_center out of range");
}

static void test_tier2_clamped_plate(void) {
    printf("  test_tier2_clamped_plate:\n");
    /* 1 m × 1 m × 0.05 m plate, 8×8×2 mesh. */
    matlab_struct *mesh = matlab_pde_mesh_cuboid_tet(1.0, 1.0, 0.05, 8, 8, 2);
    matlab_mat *nodes = matlab_pde_mesh_nodes(mesh);
    matlab_mat *tets  = matlab_pde_mesh_tets(mesh);
    matlab_mat *faces = matlab_pde_mesh_faces(mesh);
    printf("    Nn=%lld, Nt=%lld, Nf=%lld\n",
           (long long)nodes->rows, (long long)tets->rows,
           (long long)faces->rows);

    double E = 2.0e11, nu = 0.30;
    matlab_mat *K = matlab_pde_assemble_elast_3d(mesh, E, nu);
    matlab_mat *F = matlab_pde_face_pressure_3d(mesh, 2.0, 1.0e5);
    printf("    K is %lld x %lld\n", (long long)K->rows, (long long)K->cols);
    printf("    F is %lld x 1 (peak |F| = %.4e)\n",
           (long long)F->rows, vec_max_abs(F));

    matlab_mat *fixed_nodes = matlab_pde_face_nodes(mesh, 1.0);
    printf("    fixed_nodes count = %lld\n", (long long)fixed_nodes->rows);

    matlab_struct *sys = matlab_pde_apply_fixed_3d(K, F, fixed_nodes);
    matlab_mat *Kc = matlab_pde_sys_K(sys);
    matlab_mat *Fc = matlab_pde_sys_F(sys);
    matlab_mat *u  = matlab_mldivide_mm(Kc, Fc);
    double peak = matlab_pde_peak_disp_3d(u);
    printf("    peak |u| = %.6e m  (%.2f microns)\n", peak, peak * 1e6);
    CHECK(peak > 1e-9 && peak < 1e-3, "peak displacement out of range");

    matlab_mat *vm = matlab_pde_von_mises_3d(mesh, u, E, nu);
    double peak_vm = vec_max(vm);
    printf("    peak von Mises = %.2e Pa  (%.2f MPa)\n", peak_vm, peak_vm / 1e6);
    CHECK(peak_vm > 1e4 && peak_vm < 1e9, "peak von Mises out of range");
}

/* #122: matlab_pde_solve_femodel must not SIGSEGV when the model has no
 * MaterialProperties struct.  matlab_struct_get_mat returns a non-NULL
 * EMPTY matlab_mat for a missing field; before the fix that empty matrix
 * was reinterpreted as a matlab_struct and struct_find_field walked its
 * garbage nfields/names → crash.  (This path is reached for a 2-D scalar
 * Poisson model when its Mesh round-trips empty under the -dap worker.) */
static void test_femodel_missing_material_props(void) {
    printf("  [#122] femodel without MaterialProperties (no SIGSEGV)\n");
    /* A real tet mesh struct, parked under Geometry so femodel enters its
     * body (past the Mesh/Geometry guard) and goes on to read the absent
     * MaterialProperties — the exact crashing path. */
    matlab_struct *mesh = matlab_pde_mesh_cuboid_tet(1.0, 1.0, 1.0,
                                                     2.0, 2.0, 2.0);
    CHECK(mesh != NULL, "mesh build failed");
    matlab_struct *model = matlab_struct_new();
    matlab_struct_set_mat(model, "Geometry", 8, (matlab_mat *)mesh);
    /* No MaterialProperties set. */
    matlab_struct *R = matlab_pde_solve_femodel(model);
    CHECK(R != NULL, "femodel returned NULL");
    printf("    ok — no crash, result non-null\n");
}

int main(void) {
    printf("test_pde:\n");
    test_tier1_poisson_square();
    test_tier2_clamped_plate();
    test_femodel_missing_material_props();
    printf("all tests passed.\n");
    return 0;
}
