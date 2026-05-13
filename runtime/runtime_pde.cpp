/* runtime_pde.cpp — Partial Differential Equation Toolbox runtime.
 *
 * See docs/pde_toolbox_roadmap.md for the full surface.  This file
 * provides the **function-form** numerical core that the MATLAB-side
 * `createpde` / `femodel` wrappers compose on top of.  The high-level
 * classdef API is layered on later; the runtime entries here are the
 * stable ABI the MLIR lowering targets.
 *
 * Tier-1 (2-D scalar elliptic PDE):
 *   matlab_pde_mesh_rect_tri      — uniform triangular mesh on [x0,x1]×[y0,y1]
 *   matlab_pde_assemble_poisson_2d — assemble K + F for -∇·(c∇u) + au = f
 *   matlab_pde_apply_dirichlet     — zero-out row/col + 1 on diag for fixed DOFs
 *   matlab_pde_boundary_nodes_rect — list of nodes on the rectangle boundary
 *
 * Tier-2 (3-D linear elasticity):
 *   matlab_pde_mesh_cuboid_tet     — structured Nx×Ny×Nz hex grid, 6 tets/hex
 *   matlab_pde_face_nodes_cuboid   — list of node ids on a given outer face
 *   matlab_pde_assemble_elast_3d   — global stiffness K (3N × 3N) for E, ν
 *   matlab_pde_face_pressure_3d    — surface pressure load on a face id
 *   matlab_pde_apply_fixed_3d      — zero-row + diag-1 for fixed DOFs in vector u
 *   matlab_pde_von_mises_3d        — elementwise von Mises from displacement
 *
 * Tier-3 / 4 (transient / nonlinear) layer on top using the existing
 * ode23s_v + Newmark + Picard infrastructure.
 *
 * Everything below uses **dense** matrices and existing `matlab_mat *`
 * descriptors — the sparse-matrix path is a Tier-5 optimisation.
 * Practical limit is ~3000 DOFs (~30 MB for K plus the LU factor).  The
 * Tier-2 gating examples stay below that comfortably.
 *
 * The MATLAB-faithful classdef API (`femodel(AnalysisType=...)`,
 * `materialProperties(YoungsModulus=...)`, etc.) lives as a Sema +
 * MLIR layer on top; see lib/Sema/Resolver.cpp and
 * lib/MLIR/Passes/LowerTensorOps.cpp.  The class itself is just a typed
 * struct (`matlab_struct *`) whose `solve` method dispatches to the
 * entries in this file.
 */

#include "runtime_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {

/* Forward declarations for cross-TU helpers defined in matlab_runtime.cpp.
 * The public ABI lives in matlab_runtime.h but the C++ inclusion of that
 * header pulls in declarations that conflict with the unaliased
 * runtime_internal.h types, so we just forward-declare the ones we use. */
matlab_struct *matlab_struct_new(void);
void   matlab_struct_set_f64(matlab_struct *s, const char *name, int64_t len, double v);
void   matlab_struct_set_mat(matlab_struct *s, const char *name, int64_t len, matlab_mat *m);
double matlab_struct_get_f64(matlab_struct *s, const char *name, int64_t len);
matlab_mat *matlab_struct_get_mat(matlab_struct *s, const char *name, int64_t len);

/* --- Tier-1: 2-D mesh on a rectangle ------------------------------ *
 * Builds a uniform Nx×Ny grid of vertices, each cell split into two
 * triangles along the (i,j)→(i+1,j+1) diagonal.  Output is a struct:
 *   .Nodes      : (Nn × 2)  x,y per node
 *   .Triangles  : (Nt × 3)  1-indexed node ids per triangle
 *   .Nx, .Ny    : grid resolution
 *   .x0, .x1, .y0, .y1 : domain extents
 *
 * Node numbering: nid(i,j) = j*Nx + i + 1 (1-based), with i ∈ [0,Nx),
 * j ∈ [0,Ny).  Nn = Nx*Ny, Nt = 2*(Nx-1)*(Ny-1).
 */
matlab_struct *matlab_pde_mesh_rect_tri(double x0, double x1,
                                        double y0, double y1,
                                        double Nx_d, double Ny_d) {
    int64_t Nx = (int64_t)Nx_d;
    int64_t Ny = (int64_t)Ny_d;
    if (Nx < 2) Nx = 2;
    if (Ny < 2) Ny = 2;
    int64_t Nn = Nx * Ny;
    int64_t Nt = 2 * (Nx - 1) * (Ny - 1);

    matlab_mat *nodes = mat_alloc(Nn, 2);
    matlab_mat *tris  = mat_alloc(Nt, 3);

    double dx = (x1 - x0) / (double)(Nx - 1);
    double dy = (y1 - y0) / (double)(Ny - 1);

    for (int64_t j = 0; j < Ny; ++j) {
        for (int64_t i = 0; i < Nx; ++i) {
            int64_t idx = j * Nx + i;
            nodes->data[idx * 2 + 0] = x0 + (double)i * dx;
            nodes->data[idx * 2 + 1] = y0 + (double)j * dy;
        }
    }

    int64_t k = 0;
    for (int64_t j = 0; j < Ny - 1; ++j) {
        for (int64_t i = 0; i < Nx - 1; ++i) {
            int64_t bl = j * Nx + i;            /* bottom-left */
            int64_t br = j * Nx + i + 1;        /* bottom-right */
            int64_t tl = (j + 1) * Nx + i;      /* top-left */
            int64_t tr = (j + 1) * Nx + i + 1;  /* top-right */
            /* lower triangle: bl, br, tr */
            tris->data[k * 3 + 0] = (double)(bl + 1);
            tris->data[k * 3 + 1] = (double)(br + 1);
            tris->data[k * 3 + 2] = (double)(tr + 1);
            k++;
            /* upper triangle: bl, tr, tl */
            tris->data[k * 3 + 0] = (double)(bl + 1);
            tris->data[k * 3 + 1] = (double)(tr + 1);
            tris->data[k * 3 + 2] = (double)(tl + 1);
            k++;
        }
    }

    matlab_struct *m = matlab_struct_new();
    matlab_struct_set_mat(m, "Nodes",     5, nodes);
    matlab_struct_set_mat(m, "Triangles", 9, tris);
    matlab_struct_set_f64(m, "Nx", 2, (double)Nx);
    matlab_struct_set_f64(m, "Ny", 2, (double)Ny);
    matlab_struct_set_f64(m, "x0", 2, x0);
    matlab_struct_set_f64(m, "x1", 2, x1);
    matlab_struct_set_f64(m, "y0", 2, y0);
    matlab_struct_set_f64(m, "y1", 2, y1);
    return m;
}

/* List of boundary node ids (1-based, column vector) for the
 * rectangle mesh.  Boundary = i == 0 || i == Nx-1 || j == 0 || j == Ny-1.
 */
matlab_mat *matlab_pde_boundary_nodes_rect(matlab_struct *mesh) {
    if (!mesh) return mat_alloc(0, 1);
    int64_t Nx = (int64_t)matlab_struct_get_f64(mesh, "Nx", 2);
    int64_t Ny = (int64_t)matlab_struct_get_f64(mesh, "Ny", 2);
    std::vector<int64_t> ids;
    ids.reserve((size_t)(2 * (Nx + Ny)));
    for (int64_t j = 0; j < Ny; ++j) {
        for (int64_t i = 0; i < Nx; ++i) {
            if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
                ids.push_back(j * Nx + i + 1);
            }
        }
    }
    matlab_mat *out = mat_alloc((int64_t)ids.size(), 1);
    for (size_t k = 0; k < ids.size(); ++k) out->data[k] = (double)ids[k];
    return out;
}

/* --- Tier-1: 2-D P1 FEM assembly ---------------------------------- *
 * Linear triangular elements.  For each triangle with nodes (i, j, k):
 *   Area, gradient of basis functions ∇φ are constant per element.
 *   Element stiffness:  Ke_pq = c · area · (∇φ_p · ∇φ_q)
 *   Element mass-like:  Me_pq = a · area · ⟨φ_p φ_q⟩ (lumped diagonal/3)
 *   Element load:       fe_p  = f · area / 3
 *
 * Output struct:
 *   .K : (Nn × Nn) global stiffness
 *   .F : (Nn × 1)  global load
 *
 * c, a, f are scalar constants.  Spatially-varying / nonlinear is
 * Tier-4 (lib/sym + extra runtime entries).
 */
matlab_struct *matlab_pde_assemble_poisson_2d(matlab_struct *mesh,
                                              double c, double a, double f) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes",     5);
    matlab_mat *tris  = matlab_struct_get_mat(mesh, "Triangles", 9);
    int64_t Nn = nodes->rows;
    int64_t Nt = tris->rows;

    matlab_mat *K = mat_alloc(Nn, Nn);
    matlab_mat *F = mat_alloc(Nn, 1);

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t i0 = (int64_t)tris->data[e * 3 + 0] - 1;
        int64_t i1 = (int64_t)tris->data[e * 3 + 1] - 1;
        int64_t i2 = (int64_t)tris->data[e * 3 + 2] - 1;
        double x0 = nodes->data[i0 * 2 + 0], y0 = nodes->data[i0 * 2 + 1];
        double x1 = nodes->data[i1 * 2 + 0], y1 = nodes->data[i1 * 2 + 1];
        double x2 = nodes->data[i2 * 2 + 0], y2 = nodes->data[i2 * 2 + 1];

        double twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        double area = 0.5 * twoA;
        if (area <= 0) area = -area; /* defensive */

        /* ∇φ_i = (1/2A) * [y_j - y_k, x_k - x_j] cyclic. */
        double b[3] = { (y1 - y2), (y2 - y0), (y0 - y1) };
        double cc[3] = { (x2 - x1), (x0 - x2), (x1 - x0) };
        double inv2A = 1.0 / twoA;
        for (int p = 0; p < 3; ++p) {
            b[p]  *= inv2A;
            cc[p] *= inv2A;
        }

        int64_t loc[3] = { i0, i1, i2 };
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) {
                double Ke = c * area * (b[p] * b[q] + cc[p] * cc[q]);
                /* Lumped mass: a · area / 3 on diagonal-equivalent. */
                if (p == q) Ke += a * area / 3.0;
                K->data[loc[p] * Nn + loc[q]] += Ke;
            }
            F->data[loc[p]] += f * area / 3.0;
        }
    }

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, K);
    matlab_struct_set_mat(out, "F", 1, F);
    return out;
}

/* --- Tier-1: apply Dirichlet u = u_val on a set of node ids -------- *
 * For each constrained node id n (1-based):
 *   F[n] = u_val
 *   K[n, :] = 0
 *   K[:, n] -= K[:, n] * u_val   (move to RHS)
 *   K[n, n] = 1
 *
 * Returns a new struct .K, .F (does not mutate the inputs).  Suitable
 * for u_val = 0; for non-zero u_val the column-elimination uses the
 * full algebra.
 */
matlab_struct *matlab_pde_apply_dirichlet(matlab_struct *sys,
                                          matlab_mat *node_ids, double u_val) {
    matlab_mat *K = matlab_struct_get_mat(sys, "K", 1);
    matlab_mat *F = matlab_struct_get_mat(sys, "F", 1);
    int64_t Nn = K->rows;

    /* Copy. */
    matlab_mat *K2 = mat_alloc(Nn, Nn);
    matlab_mat *F2 = mat_alloc(Nn, 1);
    memcpy(K2->data, K->data, sizeof(double) * (size_t)(Nn * Nn));
    memcpy(F2->data, F->data, sizeof(double) * (size_t)Nn);

    int64_t Nd = node_ids->rows * node_ids->cols;
    std::vector<int8_t> is_fixed((size_t)Nn, 0);
    for (int64_t k = 0; k < Nd; ++k) {
        int64_t n = (int64_t)node_ids->data[k] - 1;
        if (n >= 0 && n < Nn) is_fixed[(size_t)n] = 1;
    }

    /* For each fixed dof n: F -= K(:,n) * u_val, then zero row+col + diag 1. */
    if (u_val != 0.0) {
        for (int64_t n = 0; n < Nn; ++n) {
            if (!is_fixed[(size_t)n]) continue;
            for (int64_t r = 0; r < Nn; ++r) {
                F2->data[r] -= K2->data[r * Nn + n] * u_val;
            }
        }
    }
    for (int64_t n = 0; n < Nn; ++n) {
        if (!is_fixed[(size_t)n]) continue;
        for (int64_t c = 0; c < Nn; ++c) {
            K2->data[n * Nn + c] = 0.0;
            K2->data[c * Nn + n] = 0.0;
        }
        K2->data[n * Nn + n] = 1.0;
        F2->data[n] = u_val;
    }

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, K2);
    matlab_struct_set_mat(out, "F", 1, F2);
    return out;
}

/* --- Tier-2: structured tet mesh of a cuboid ---------------------- *
 * Inputs: width W (x-extent), depth D (y-extent), height H (z-extent),
 *         resolution Nx, Ny, Nz cells along each axis.
 * The cuboid occupies [0,W] × [0,D] × [0,H].
 * Each hex cell is split into 6 tetrahedra using the standard
 * Tetrahedrons-per-hex tiling (Kuhn / "5-tet" decomposition is also
 * valid; we use 6-tet here because it generalises to N-cell grids
 * without orientation tricks).
 *
 * Face id assignment for the outer surface (matches MathWorks convention
 * for `multicuboid`):
 *   1 = z = 0  (bottom)
 *   2 = z = H  (top)
 *   3 = y = 0  (front, -y)
 *   4 = y = D  (back,  +y)
 *   5 = x = 0  (left)
 *   6 = x = W  (right)
 *
 * Output struct:
 *   .Nodes      : (Nn × 3) x,y,z
 *   .Tets       : (Nt × 4) 1-indexed node ids per tet
 *   .Faces      : (Nbnd × 4) [face_id, n1, n2, n3] per boundary triangle
 *   .Nx,Ny,Nz   : cell counts
 *   .W,D,H      : extents
 */
matlab_struct *matlab_pde_mesh_cuboid_tet(double W, double D, double H,
                                          double Nxd, double Nyd, double Nzd) {
    int64_t Nx = (int64_t)Nxd; if (Nx < 1) Nx = 1;
    int64_t Ny = (int64_t)Nyd; if (Ny < 1) Ny = 1;
    int64_t Nz = (int64_t)Nzd; if (Nz < 1) Nz = 1;
    int64_t Px = Nx + 1, Py = Ny + 1, Pz = Nz + 1;
    int64_t Nn = Px * Py * Pz;
    int64_t Nhex = Nx * Ny * Nz;
    int64_t Nt = 6 * Nhex;

    matlab_mat *nodes = mat_alloc(Nn, 3);
    matlab_mat *tets  = mat_alloc(Nt, 4);

    auto nid = [&](int64_t i, int64_t j, int64_t k) -> int64_t {
        return (k * Py + j) * Px + i;  /* 0-based */
    };

    double dx = W / (double)Nx;
    double dy = D / (double)Ny;
    double dz = H / (double)Nz;

    for (int64_t k = 0; k < Pz; ++k) {
        for (int64_t j = 0; j < Py; ++j) {
            for (int64_t i = 0; i < Px; ++i) {
                int64_t idx = nid(i, j, k);
                nodes->data[idx * 3 + 0] = (double)i * dx;
                nodes->data[idx * 3 + 1] = (double)j * dy;
                nodes->data[idx * 3 + 2] = (double)k * dz;
            }
        }
    }

    /* For a unit cube with corners labelled 0..7:
     *   0 = (0,0,0)  1 = (1,0,0)  2 = (1,1,0)  3 = (0,1,0)
     *   4 = (0,0,1)  5 = (1,0,1)  6 = (1,1,1)  7 = (0,1,1)
     * Decomposition into 6 tets (positive volume), all sharing the
     * diagonal 0-6:
     *   T1: 0, 1, 2, 6
     *   T2: 0, 2, 3, 6
     *   T3: 0, 3, 7, 6
     *   T4: 0, 7, 4, 6
     *   T5: 0, 4, 5, 6
     *   T6: 0, 5, 1, 6
     */
    static const int Tdef[6][4] = {
        {0, 1, 2, 6},
        {0, 2, 3, 6},
        {0, 3, 7, 6},
        {0, 7, 4, 6},
        {0, 4, 5, 6},
        {0, 5, 1, 6},
    };

    int64_t te = 0;
    for (int64_t k = 0; k < Nz; ++k) {
        for (int64_t j = 0; j < Ny; ++j) {
            for (int64_t i = 0; i < Nx; ++i) {
                int64_t corners[8] = {
                    nid(i,     j,     k    ),
                    nid(i + 1, j,     k    ),
                    nid(i + 1, j + 1, k    ),
                    nid(i,     j + 1, k    ),
                    nid(i,     j,     k + 1),
                    nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1),
                    nid(i,     j + 1, k + 1),
                };
                for (int t = 0; t < 6; ++t) {
                    tets->data[te * 4 + 0] = (double)(corners[Tdef[t][0]] + 1);
                    tets->data[te * 4 + 1] = (double)(corners[Tdef[t][1]] + 1);
                    tets->data[te * 4 + 2] = (double)(corners[Tdef[t][2]] + 1);
                    tets->data[te * 4 + 3] = (double)(corners[Tdef[t][3]] + 1);
                    te++;
                }
            }
        }
    }

    /* Boundary face triangulation.  Each outer face of each surface
     * hex cell contributes 2 triangles in a consistent orientation
     * (outward normal). */
    std::vector<int64_t> face_id;
    std::vector<int64_t> face_n1, face_n2, face_n3;
    face_id.reserve((size_t)(2 * 2 * (Nx * Ny + Nx * Nz + Ny * Nz)));

    auto add_tri = [&](int64_t fid, int64_t a, int64_t b, int64_t c) {
        face_id.push_back(fid);
        face_n1.push_back(a + 1);
        face_n2.push_back(b + 1);
        face_n3.push_back(c + 1);
    };

    /* z = 0 (face 1) — outward normal -z. */
    for (int64_t j = 0; j < Ny; ++j)
        for (int64_t i = 0; i < Nx; ++i) {
            int64_t a = nid(i,     j,     0);
            int64_t b = nid(i + 1, j,     0);
            int64_t cc = nid(i + 1, j + 1, 0);
            int64_t d = nid(i,     j + 1, 0);
            add_tri(1, a, cc, b);
            add_tri(1, a, d, cc);
        }
    /* z = H (face 2) — outward normal +z. */
    for (int64_t j = 0; j < Ny; ++j)
        for (int64_t i = 0; i < Nx; ++i) {
            int64_t a = nid(i,     j,     Nz);
            int64_t b = nid(i + 1, j,     Nz);
            int64_t cc = nid(i + 1, j + 1, Nz);
            int64_t d = nid(i,     j + 1, Nz);
            add_tri(2, a, b, cc);
            add_tri(2, a, cc, d);
        }
    /* y = 0 (face 3) — outward normal -y. */
    for (int64_t k = 0; k < Nz; ++k)
        for (int64_t i = 0; i < Nx; ++i) {
            int64_t a = nid(i,     0, k    );
            int64_t b = nid(i + 1, 0, k    );
            int64_t cc = nid(i + 1, 0, k + 1);
            int64_t d = nid(i,     0, k + 1);
            add_tri(3, a, b, cc);
            add_tri(3, a, cc, d);
        }
    /* y = D (face 4) — outward normal +y. */
    for (int64_t k = 0; k < Nz; ++k)
        for (int64_t i = 0; i < Nx; ++i) {
            int64_t a = nid(i,     Ny, k    );
            int64_t b = nid(i + 1, Ny, k    );
            int64_t cc = nid(i + 1, Ny, k + 1);
            int64_t d = nid(i,     Ny, k + 1);
            add_tri(4, a, cc, b);
            add_tri(4, a, d, cc);
        }
    /* x = 0 (face 5) — outward normal -x. */
    for (int64_t k = 0; k < Nz; ++k)
        for (int64_t j = 0; j < Ny; ++j) {
            int64_t a = nid(0, j,     k    );
            int64_t b = nid(0, j + 1, k    );
            int64_t cc = nid(0, j + 1, k + 1);
            int64_t d = nid(0, j,     k + 1);
            add_tri(5, a, cc, b);
            add_tri(5, a, d, cc);
        }
    /* x = W (face 6) — outward normal +x. */
    for (int64_t k = 0; k < Nz; ++k)
        for (int64_t j = 0; j < Ny; ++j) {
            int64_t a = nid(Nx, j,     k    );
            int64_t b = nid(Nx, j + 1, k    );
            int64_t cc = nid(Nx, j + 1, k + 1);
            int64_t d = nid(Nx, j,     k + 1);
            add_tri(6, a, b, cc);
            add_tri(6, a, cc, d);
        }

    int64_t Nf = (int64_t)face_id.size();
    matlab_mat *faces = mat_alloc(Nf, 4);
    for (int64_t k = 0; k < Nf; ++k) {
        faces->data[k * 4 + 0] = (double)face_id[(size_t)k];
        faces->data[k * 4 + 1] = (double)face_n1[(size_t)k];
        faces->data[k * 4 + 2] = (double)face_n2[(size_t)k];
        faces->data[k * 4 + 3] = (double)face_n3[(size_t)k];
    }

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Nodes", 5, nodes);
    matlab_struct_set_mat(out, "Tets",  4, tets);
    matlab_struct_set_mat(out, "Faces", 5, faces);
    matlab_struct_set_f64(out, "Nx", 2, (double)Nx);
    matlab_struct_set_f64(out, "Ny", 2, (double)Ny);
    matlab_struct_set_f64(out, "Nz", 2, (double)Nz);
    matlab_struct_set_f64(out, "W",  1, W);
    matlab_struct_set_f64(out, "D",  1, D);
    matlab_struct_set_f64(out, "H",  1, H);
    return out;
}

/* All unique node ids belonging to a given face id (1-based).
 * Output: column vector of node ids (sorted, deduplicated). */
matlab_mat *matlab_pde_face_nodes(matlab_struct *mesh, double face_id_d) {
    int64_t fid = (int64_t)face_id_d;
    matlab_mat *faces = matlab_struct_get_mat(mesh, "Faces", 5);
    int64_t Nf = faces->rows;
    std::vector<int64_t> ids;
    ids.reserve((size_t)Nf * 3);
    for (int64_t k = 0; k < Nf; ++k) {
        if ((int64_t)faces->data[k * 4 + 0] != fid) continue;
        for (int c = 1; c <= 3; ++c) {
            ids.push_back((int64_t)faces->data[k * 4 + c]);
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    matlab_mat *out = mat_alloc((int64_t)ids.size(), 1);
    for (size_t k = 0; k < ids.size(); ++k) out->data[k] = (double)ids[k];
    return out;
}

/* --- Tier-2: 3-D linear elasticity FEM assembly ------------------- *
 *
 * Isotropic linear elasticity, Cauchy strain ε = ½(∇u + ∇u^T), stress
 * σ = D·ε.  4-node linear tet ("constant-strain tet"): the strain is
 * constant per element, computed from the 4 node coordinates and 3
 * displacements per node (12 DOFs per element).
 *
 * D (6×6) for isotropic material with Young's modulus E and
 * Poisson's ratio ν:
 *
 *   λ = Eν / ((1+ν)(1-2ν))    µ = E / (2(1+ν))
 *   D = [[λ+2µ, λ,    λ,    0, 0, 0],
 *        [λ,    λ+2µ, λ,    0, 0, 0],
 *        [λ,    λ,    λ+2µ, 0, 0, 0],
 *        [0,    0,    0,    µ, 0, 0],
 *        [0,    0,    0,    0, µ, 0],
 *        [0,    0,    0,    0, 0, µ]]
 *
 * Strain-displacement matrix B (6×12) for a 4-node tet:
 *   ε = B · u_e
 * where u_e = [u1x u1y u1z u2x u2y u2z u3x u3y u3z u4x u4y u4z]^T.
 *
 * The shape-function gradients are constants: ∇N_i = (1/6V)·b_i where
 * b_i is the cofactor vector of node i in the 4×4 augmented tet matrix.
 *
 * Returns the global stiffness matrix K (3N × 3N) as a real matlab_mat.
 */

static void elast_compute_grad(const double X[4][3], double dN[4][3],
                               double *vol_out) {
    /* The shape-function gradient ∇N_i is the cofactor of node i in
     * the 4x4 augmented matrix
     *   [ 1  x1 y1 z1 ]
     *   [ 1  x2 y2 z2 ]
     *   [ 1  x3 y3 z3 ]
     *   [ 1  x4 y4 z4 ]
     * divided by 6V.  Standard formula:
     *   x21 = x2 - x1,  ...  etc.
     *   a_i, b_i, c_i, d_i are determinants of 3x3 sub-blocks.
     * We use a compact derivation: gradient of the linear interpolant
     * x = N_i x_i is identity, so ∇N_i can be solved from the local
     * system.  */
    double M[4][4];
    for (int i = 0; i < 4; ++i) {
        M[i][0] = 1.0;
        M[i][1] = X[i][0];
        M[i][2] = X[i][1];
        M[i][3] = X[i][2];
    }
    /* Volume via det / 6. */
    double a = M[0][1], b = M[0][2], cc = M[0][3];
    double det = 0.0;
    /* Expansion: det(M) — using shifted-row trick:
     * Subtract row 0 from rows 1..3, then 3x3 det of M[1..3][1..3]. */
    double A[3][3];
    for (int i = 0; i < 3; ++i) {
        A[i][0] = M[i + 1][1] - a;
        A[i][1] = M[i + 1][2] - b;
        A[i][2] = M[i + 1][3] - cc;
    }
    det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    double V = det / 6.0;
    if (V < 0) V = -V;
    *vol_out = V;

    /* Now solve [1 x y z]^T grad_coeffs = [1 0 0 0] etc.  Equivalent
     * to inverting M^T, but we only need the spatial-gradient columns
     * (last 3 entries of each column of M^{-T}).
     *
     * Direct closed-form for tet shape functions: for node i, the
     * gradient ∇N_i is (1/6V) · (-1)^(i+1) · cof_i where cof_i is the
     * cofactor vector built from the OTHER three nodes' coordinates.
     */
    static const int permI[4][3] = {
        {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
    };
    /* For each node i: build a 3×3 matrix S whose rows are the (j, k, l)
     * coordinate differences, then ∇N_i^T = (1/6V) · cofactor_row_of_M^{-1}.
     * A cleaner formula: solve [x_j - x_i, x_k - x_i, x_l - x_i]^T · ∇N_i =
     * [-1, -1, -1]^T (since N_i = 1 at node i and 0 at j, k, l).
     * Then for j, k, l: ∇N_j satisfies the same system with rhs [+1,0,0]^T etc.
     * Easiest: solve via LU once per element.
     */
    for (int i = 0; i < 4; ++i) {
        int j = permI[i][0], k = permI[i][1], l = permI[i][2];
        double S[3][3];
        for (int r = 0; r < 3; ++r) {
            S[0][r] = X[j][r] - X[i][r];
            S[1][r] = X[k][r] - X[i][r];
            S[2][r] = X[l][r] - X[i][r];
        }
        /* N_i = 1 at i, 0 at j, k, l → gradient g satisfies
         *   (X_j - X_i)·g = -1, (X_k - X_i)·g = -1, (X_l - X_i)·g = -1.
         * Solve S · g = [-1, -1, -1]. */
        /* 3×3 inverse / Cramer's rule. */
        double sdet = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1])
                    - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0])
                    + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
        if (sdet == 0.0) sdet = 1e-30;
        double rhs[3] = {-1.0, -1.0, -1.0};
        /* g_x = det(S with col 0 replaced) / det(S), etc. */
        double Sx[3][3], Sy[3][3], Sz[3][3];
        memcpy(Sx, S, sizeof(S));
        memcpy(Sy, S, sizeof(S));
        memcpy(Sz, S, sizeof(S));
        for (int r = 0; r < 3; ++r) {
            Sx[r][0] = rhs[r];
            Sy[r][1] = rhs[r];
            Sz[r][2] = rhs[r];
        }
        double dx = Sx[0][0] * (Sx[1][1] * Sx[2][2] - Sx[1][2] * Sx[2][1])
                  - Sx[0][1] * (Sx[1][0] * Sx[2][2] - Sx[1][2] * Sx[2][0])
                  + Sx[0][2] * (Sx[1][0] * Sx[2][1] - Sx[1][1] * Sx[2][0]);
        double dy = Sy[0][0] * (Sy[1][1] * Sy[2][2] - Sy[1][2] * Sy[2][1])
                  - Sy[0][1] * (Sy[1][0] * Sy[2][2] - Sy[1][2] * Sy[2][0])
                  + Sy[0][2] * (Sy[1][0] * Sy[2][1] - Sy[1][1] * Sy[2][0]);
        double dz = Sz[0][0] * (Sz[1][1] * Sz[2][2] - Sz[1][2] * Sz[2][1])
                  - Sz[0][1] * (Sz[1][0] * Sz[2][2] - Sz[1][2] * Sz[2][0])
                  + Sz[0][2] * (Sz[1][0] * Sz[2][1] - Sz[1][1] * Sz[2][0]);
        dN[i][0] = dx / sdet;
        dN[i][1] = dy / sdet;
        dN[i][2] = dz / sdet;
    }
}

static void elast_D_matrix(double E, double nu, double D[6][6]) {
    double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu  = E / (2.0 * (1.0 + nu));
    memset(D, 0, sizeof(double) * 36);
    D[0][0] = lam + 2.0 * mu;  D[0][1] = lam;            D[0][2] = lam;
    D[1][0] = lam;             D[1][1] = lam + 2.0 * mu; D[1][2] = lam;
    D[2][0] = lam;             D[2][1] = lam;            D[2][2] = lam + 2.0 * mu;
    D[3][3] = mu;
    D[4][4] = mu;
    D[5][5] = mu;
}

/* Build the 6×12 B matrix for a 4-node tet given the 4 shape-function
 * gradients dN (4×3).  Voigt order: εxx, εyy, εzz, γxy, γyz, γxz. */
static void elast_B_matrix(const double dN[4][3], double B[6][12]) {
    memset(B, 0, sizeof(double) * 72);
    for (int i = 0; i < 4; ++i) {
        double bx = dN[i][0], by = dN[i][1], bz = dN[i][2];
        int c = i * 3;
        /* εxx = ∂u/∂x */
        B[0][c + 0] = bx;
        /* εyy = ∂v/∂y */
        B[1][c + 1] = by;
        /* εzz = ∂w/∂z */
        B[2][c + 2] = bz;
        /* γxy = ∂u/∂y + ∂v/∂x */
        B[3][c + 0] = by;
        B[3][c + 1] = bx;
        /* γyz = ∂v/∂z + ∂w/∂y */
        B[4][c + 1] = bz;
        B[4][c + 2] = by;
        /* γxz = ∂u/∂z + ∂w/∂x */
        B[5][c + 0] = bz;
        B[5][c + 2] = bx;
    }
}

matlab_mat *matlab_pde_assemble_elast_3d(matlab_struct *mesh,
                                         double E, double nu) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *tets  = matlab_struct_get_mat(mesh, "Tets",  4);
    int64_t Nn = nodes->rows;
    int64_t Nt = tets->rows;
    int64_t Ndof = 3 * Nn;

    matlab_mat *K = mat_alloc(Ndof, Ndof);

    double D[6][6];
    elast_D_matrix(E, nu, D);

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t ids[4];
        double X[4][3];
        for (int i = 0; i < 4; ++i) {
            ids[i] = (int64_t)tets->data[e * 4 + i] - 1;
            X[i][0] = nodes->data[ids[i] * 3 + 0];
            X[i][1] = nodes->data[ids[i] * 3 + 1];
            X[i][2] = nodes->data[ids[i] * 3 + 2];
        }
        double dN[4][3];
        double V;
        elast_compute_grad(X, dN, &V);
        double B[6][12];
        elast_B_matrix(dN, B);

        /* Ke = V · B^T · D · B  (12 × 12). */
        double DB[6][12];
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 12; ++c) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += D[r][k] * B[k][c];
                DB[r][c] = s;
            }
        double Ke[12][12];
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 12; ++c) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += B[k][r] * DB[k][c];
                Ke[r][c] = V * s;
            }
        /* Scatter into global K. */
        for (int p = 0; p < 4; ++p) {
            for (int q = 0; q < 4; ++q) {
                int64_t gp = ids[p] * 3;
                int64_t gq = ids[q] * 3;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        K->data[(gp + a) * Ndof + (gq + b)] +=
                            Ke[p * 3 + a][q * 3 + b];
                    }
            }
        }
    }
    return K;
}

/* Surface pressure load on `face_id`.  For each boundary triangle on
 * that face, compute the outward unit normal and the triangle area,
 * then distribute -p * n * area / 3 to each of the 3 corner nodes
 * (the negative sign comes from MathWorks convention: positive pressure
 * acts INTO the body, opposite to the outward normal).
 *
 * Returns: F (3N × 1).
 */
matlab_mat *matlab_pde_face_pressure_3d(matlab_struct *mesh,
                                        double face_id_d,
                                        double pressure) {
    int64_t fid = (int64_t)face_id_d;
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *faces = matlab_struct_get_mat(mesh, "Faces", 5);
    int64_t Nn = nodes->rows;
    int64_t Nf = faces->rows;
    int64_t Ndof = 3 * Nn;
    matlab_mat *F = mat_alloc(Ndof, 1);

    for (int64_t k = 0; k < Nf; ++k) {
        if ((int64_t)faces->data[k * 4 + 0] != fid) continue;
        int64_t i0 = (int64_t)faces->data[k * 4 + 1] - 1;
        int64_t i1 = (int64_t)faces->data[k * 4 + 2] - 1;
        int64_t i2 = (int64_t)faces->data[k * 4 + 3] - 1;
        double *p0 = nodes->data + i0 * 3;
        double *p1 = nodes->data + i1 * 3;
        double *p2 = nodes->data + i2 * 3;
        double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        double e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        /* Cross product e1 × e2; magnitude is 2*area, direction is the
         * outward normal (faces were built with outward orientation). */
        double n[3] = {
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        };
        double mag = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (mag == 0) continue;
        double area = 0.5 * mag;
        double inv_mag = 1.0 / mag;
        n[0] *= inv_mag; n[1] *= inv_mag; n[2] *= inv_mag;
        /* Force per node = -p * n * area / 3.  Negative because
         * positive pressure pushes INTO the body. */
        double s = -pressure * area / 3.0;
        int64_t ids[3] = {i0, i1, i2};
        for (int c = 0; c < 3; ++c) {
            F->data[ids[c] * 3 + 0] += s * n[0];
            F->data[ids[c] * 3 + 1] += s * n[1];
            F->data[ids[c] * 3 + 2] += s * n[2];
        }
    }
    return F;
}

/* Apply fixed-DOF Dirichlet (u = 0) for a vector of node ids. Zeros
 * the rows + cols + sets diagonal to 1 + zeros F entries for all 3
 * DOFs of each fixed node.
 *
 * Note: this is a same-as-Tier-1 path but extended to vector DOFs.
 * For non-zero displacement BCs, the column-elimination step would
 * need to be wired the same way.  All Tier-2 examples use u = 0.
 */
matlab_struct *matlab_pde_apply_fixed_3d(matlab_mat *K, matlab_mat *F,
                                         matlab_mat *node_ids) {
    int64_t Ndof = K->rows;
    int64_t Nn = Ndof / 3;
    matlab_mat *K2 = mat_alloc(Ndof, Ndof);
    matlab_mat *F2 = mat_alloc(Ndof, 1);
    memcpy(K2->data, K->data, sizeof(double) * (size_t)(Ndof * Ndof));
    memcpy(F2->data, F->data, sizeof(double) * (size_t)Ndof);

    int64_t Nd = node_ids->rows * node_ids->cols;
    std::vector<int8_t> fixed_dof((size_t)Ndof, 0);
    for (int64_t k = 0; k < Nd; ++k) {
        int64_t n = (int64_t)node_ids->data[k] - 1;
        if (n < 0 || n >= Nn) continue;
        fixed_dof[(size_t)(n * 3 + 0)] = 1;
        fixed_dof[(size_t)(n * 3 + 1)] = 1;
        fixed_dof[(size_t)(n * 3 + 2)] = 1;
    }
    for (int64_t r = 0; r < Ndof; ++r) {
        if (!fixed_dof[(size_t)r]) continue;
        for (int64_t c = 0; c < Ndof; ++c) {
            K2->data[r * Ndof + c] = 0.0;
            K2->data[c * Ndof + r] = 0.0;
        }
        K2->data[r * Ndof + r] = 1.0;
        F2->data[r] = 0.0;
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, K2);
    matlab_struct_set_mat(out, "F", 1, F2);
    return out;
}

/* Reshape a 3N × 1 displacement vector into an N × 3 matrix
 * (rows = nodes, cols = ux/uy/uz). */
matlab_mat *matlab_pde_reshape_disp_3d(matlab_mat *u_flat) {
    int64_t Ndof = u_flat->rows * u_flat->cols;
    int64_t Nn = Ndof / 3;
    matlab_mat *U = mat_alloc(Nn, 3);
    for (int64_t i = 0; i < Nn; ++i) {
        U->data[i * 3 + 0] = u_flat->data[i * 3 + 0];
        U->data[i * 3 + 1] = u_flat->data[i * 3 + 1];
        U->data[i * 3 + 2] = u_flat->data[i * 3 + 2];
    }
    return U;
}

/* Compute per-element von Mises stress.  Returns an Nt × 1 matrix
 * with one entry per tet.  Stress σ = D · B · u_e per element.
 *
 * Von Mises:
 *   σ_vm = sqrt(½·((σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)² +
 *                 6·(σxy² + σyz² + σxz²)))
 */
matlab_mat *matlab_pde_von_mises_3d(matlab_struct *mesh, matlab_mat *u_flat,
                                    double E, double nu) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *tets  = matlab_struct_get_mat(mesh, "Tets",  4);
    int64_t Nt = tets->rows;

    double D[6][6];
    elast_D_matrix(E, nu, D);

    matlab_mat *vm = mat_alloc(Nt, 1);

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t ids[4];
        double X[4][3];
        for (int i = 0; i < 4; ++i) {
            ids[i] = (int64_t)tets->data[e * 4 + i] - 1;
            X[i][0] = nodes->data[ids[i] * 3 + 0];
            X[i][1] = nodes->data[ids[i] * 3 + 1];
            X[i][2] = nodes->data[ids[i] * 3 + 2];
        }
        double dN[4][3];
        double V;
        elast_compute_grad(X, dN, &V);
        double B[6][12];
        elast_B_matrix(dN, B);

        double ue[12];
        for (int i = 0; i < 4; ++i) {
            ue[i * 3 + 0] = u_flat->data[ids[i] * 3 + 0];
            ue[i * 3 + 1] = u_flat->data[ids[i] * 3 + 1];
            ue[i * 3 + 2] = u_flat->data[ids[i] * 3 + 2];
        }
        double eps[6] = {0};
        for (int r = 0; r < 6; ++r) {
            double s = 0.0;
            for (int c = 0; c < 12; ++c) s += B[r][c] * ue[c];
            eps[r] = s;
        }
        double sig[6] = {0};
        for (int r = 0; r < 6; ++r) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) s += D[r][c] * eps[c];
            sig[r] = s;
        }
        double sxx = sig[0], syy = sig[1], szz = sig[2];
        double sxy = sig[3], syz = sig[4], sxz = sig[5];
        double vmv = 0.5 * ((sxx - syy) * (sxx - syy)
                          + (syy - szz) * (syy - szz)
                          + (szz - sxx) * (szz - sxx)
                          + 6.0 * (sxy * sxy + syz * syz + sxz * sxz));
        vm->data[e] = sqrt(vmv);
    }
    return vm;
}

/* Field accessors for the per-tier system structs.
 *
 * Bypasses the generic matlab_struct_get_mat path so the MATLAB-side
 * field access doesn't have to fight the Sema type-inference default
 * (which assumes struct fields are f64).  Each accessor is a thin
 * wrapper that returns the matrix held under a fixed field name.
 */
matlab_mat *matlab_pde_sys_K(matlab_struct *sys) {
    return matlab_struct_get_mat(sys, "K", 1);
}
matlab_mat *matlab_pde_sys_F(matlab_struct *sys) {
    return matlab_struct_get_mat(sys, "F", 1);
}
matlab_mat *matlab_pde_sys_M(matlab_struct *sys) {
    return matlab_struct_get_mat(sys, "M", 1);
}
matlab_mat *matlab_pde_mesh_nodes(matlab_struct *mesh) {
    return matlab_struct_get_mat(mesh, "Nodes", 5);
}
matlab_mat *matlab_pde_mesh_triangles(matlab_struct *mesh) {
    return matlab_struct_get_mat(mesh, "Triangles", 9);
}
matlab_mat *matlab_pde_mesh_tets(matlab_struct *mesh) {
    return matlab_struct_get_mat(mesh, "Tets", 4);
}
matlab_mat *matlab_pde_mesh_faces(matlab_struct *mesh) {
    return matlab_struct_get_mat(mesh, "Faces", 5);
}

/* matlab_pde_solve_femodel — kernel for the femodel classdef's solve()
 * method.  Reads the femodel's flattened field layout directly from
 * the underlying struct (so the MATLAB-side solve() body stays a thin
 * one-call wrapper instead of fighting the prelude-function
 * func-to-llvm conversion).
 *
 * Expected fields on `model` (set by the classdef ctor / set*()
 * helpers in runtime/pde_classdefs.m):
 *   .Mesh                 — volumetric mesh struct (Nodes, Tets, Faces).
 *   .Geometry             — fallback mesh when Mesh is empty.
 *   .MaterialProperties   — sub-struct with YoungsModulus / PoissonsRatio.
 *   .FixedFaces           — Nx1 column of face ids where Constraint='fixed'.
 *   .PressureFaces        — Nx2 [face_id, pressure_value] table.
 *
 * Returns a struct with:
 *   .Mesh     — the underlying mesh (passed back so the result class
 *               can hand it to the user).
 *   .u        — 3N x 1 displacement vector.
 *   .vm       — N x 1 per-node von Mises.
 */
matlab_mat *matlab_sparse_full(void *Sv);
matlab_mat *matlab_sparse_diag(void *Sv);
matlab_struct *matlab_sparse_pcg(void *Sv, matlab_mat *b,
                                 double tol, double maxit_d);
matlab_mat *matlab_sparse_pcg_x(matlab_struct *r);
void *matlab_pde_sys_K_sparse(matlab_struct *sys);
matlab_mat *matlab_pde_node_von_mises_3d(matlab_struct *mesh, matlab_mat *u_flat,
                                          double E, double nu);
matlab_mat *matlab_pde_face_pressure_3d(matlab_struct *mesh,
                                        double face_id_d, double pressure);
matlab_mat *matlab_pde_face_nodes(matlab_struct *mesh, double face_id_d);

static bool field_holds_struct(matlab_struct *s, const char *name, int64_t len);
matlab_struct *matlab_pde_assemble_poisson_3d_sparse(matlab_struct *mesh,
                                                     double c, double a, double f);
matlab_struct *matlab_pde_apply_dirichlet_3d_sparse(void *K_sparse,
                                                     matlab_mat *F,
                                                     matlab_mat *node_ids,
                                                     double u_val);
matlab_mat *matlab_pde_face_scalar_load_3d(matlab_struct *mesh,
                                            double face_id_d, double q);
matlab_mat *matlab_pde_sys_F(matlab_struct *sys);

matlab_struct *matlab_pde_solve_femodel(matlab_struct *model) {
    /* Pull mesh — fall back to Geometry if Mesh is empty. */
    matlab_struct *mesh = nullptr;
    if (field_holds_struct(model, "Mesh", 4)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Mesh", 4);
    } else if (field_holds_struct(model, "Geometry", 8)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Geometry", 8);
    } else {
        return matlab_struct_new();  /* empty result */
    }
    /* Material properties. */
    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    double E  = matlab_struct_get_f64(props, "YoungsModulus", 13);
    double nu = matlab_struct_get_f64(props, "PoissonsRatio", 13);

    /* Sparse 3-D elasticity assembly. */
    extern void *matlab_pde_assemble_elast_3d_sparse(matlab_struct *mesh,
                                                     double E, double nu);
    void *K_sp = matlab_pde_assemble_elast_3d_sparse(mesh, E, nu);

    /* Build F by walking PressureFaces rows. */
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    int64_t Nn = nodes->rows;
    matlab_mat *F = mat_alloc(3 * Nn, 1);
    matlab_mat *pf = matlab_struct_get_mat(model, "PressureFaces", 13);
    if (pf && pf->rows > 0 && pf->cols >= 2) {
        for (int64_t i = 0; i < pf->rows; ++i) {
            double fid = pf->data[i * pf->cols + 0];
            double p   = pf->data[i * pf->cols + 1];
            matlab_mat *Fk = matlab_pde_face_pressure_3d(mesh, fid, p);
            for (int64_t k = 0; k < 3 * Nn; ++k) F->data[k] += Fk->data[k];
        }
    }

    /* Union fixed nodes across all FixedFaces entries. */
    matlab_mat *ff = matlab_struct_get_mat(model, "FixedFaces", 10);
    std::vector<double> fixed_nodes_vec;
    if (ff && ff->rows > 0) {
        for (int64_t i = 0; i < ff->rows; ++i) {
            double fid = ff->data[i];
            matlab_mat *ids_here = matlab_pde_face_nodes(mesh, fid);
            for (int64_t k = 0; k < ids_here->rows; ++k) {
                fixed_nodes_vec.push_back(ids_here->data[k]);
            }
        }
    }
    matlab_mat *fixed_ids = mat_alloc((int64_t)fixed_nodes_vec.size(), 1);
    for (size_t k = 0; k < fixed_nodes_vec.size(); ++k)
        fixed_ids->data[k] = fixed_nodes_vec[k];

    /* Apply Dirichlet + solve via PCG. */
    extern matlab_struct *matlab_pde_apply_fixed_3d_sparse(void *K_sparse,
                                                          matlab_mat *F,
                                                          matlab_mat *node_ids);
    matlab_struct *sys2 = matlab_pde_apply_fixed_3d_sparse(K_sp, F, fixed_ids);
    void *Kc      = matlab_pde_sys_K_sparse(sys2);
    matlab_mat *Fc = matlab_pde_sys_F(sys2);

    matlab_struct *pcg_res = matlab_sparse_pcg(Kc, Fc, 1e-6, 4000);
    matlab_mat *u = matlab_sparse_pcg_x(pcg_res);

    /* Per-node von Mises. */
    matlab_mat *vm = matlab_pde_node_von_mises_3d(mesh, u, E, nu);

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Mesh", 4, (matlab_mat *)mesh);
    matlab_struct_set_mat(out, "u",    1, u);
    matlab_struct_set_mat(out, "vm",   2, vm);
    return out;
}

/* Field accessors for the kernel result struct. */
matlab_mat *matlab_pde_kernel_mesh(matlab_struct *r) {
    return matlab_struct_get_mat(r, "Mesh", 4);
}
matlab_mat *matlab_pde_kernel_u(matlab_struct *r) {
    return matlab_struct_get_mat(r, "u", 1);
}
matlab_mat *matlab_pde_kernel_vm(matlab_struct *r) {
    return matlab_struct_get_mat(r, "vm", 2);
}

/* --- femodel setters (runtime builtins) ---------------------------
 *
 * MATLAB-side surface looks like
 *   model = pde_set_material      (model, props);
 *   model = pde_set_face_fixed    (model, face_id);
 *   model = pde_set_face_pressure (model, face_id, p);
 * with each call returning a (mutated) femodel instance.  The
 * implementation MUTATES the underlying matlab_obj IN PLACE for
 * efficiency — semantically MATLAB returns by value, but since the
 * struct's backing storage is heap-allocated, returning the same
 * pointer is observationally equivalent.
 */
matlab_struct *matlab_pde_set_material(matlab_struct *model,
                                        matlab_struct *props) {
    matlab_struct_set_mat(model, "MaterialProperties", 18,
                           (matlab_mat *)props);
    return model;
}

matlab_struct *matlab_pde_set_face_fixed(matlab_struct *model,
                                          double face_id) {
    matlab_mat *cur = matlab_struct_get_mat(model, "FixedFaces", 10);
    int64_t n = (cur && cur->rows > 0) ? cur->rows : 0;
    matlab_mat *next = mat_alloc(n + 1, 1);
    for (int64_t i = 0; i < n; ++i) next->data[i] = cur->data[i];
    next->data[n] = face_id;
    matlab_struct_set_mat(model, "FixedFaces", 10, next);
    return model;
}

matlab_struct *matlab_pde_set_face_pressure(matlab_struct *model,
                                             double face_id, double p) {
    matlab_mat *cur = matlab_struct_get_mat(model, "PressureFaces", 13);
    int64_t n = (cur && cur->rows > 0) ? cur->rows : 0;
    matlab_mat *next = mat_alloc(n + 1, 2);
    for (int64_t i = 0; i < n; ++i) {
        next->data[i * 2 + 0] = cur->data[i * 2 + 0];
        next->data[i * 2 + 1] = cur->data[i * 2 + 1];
    }
    next->data[n * 2 + 0] = face_id;
    next->data[n * 2 + 1] = p;
    matlab_struct_set_mat(model, "PressureFaces", 13, next);
    return model;
}

/* Tier-3 scalar setters: Temperature / Voltage Dirichlet on faces,
 * Heat / ChargeDensity sources.  Each follows the same flat (Nx2)
 * append pattern as PressureFaces.
 */
static matlab_struct *pde_append_face_pair(matlab_struct *model,
                                           const char *field, int field_len,
                                           double face_id, double value) {
    matlab_mat *cur = matlab_struct_get_mat(model, field, field_len);
    int64_t n = (cur && cur->rows > 0) ? cur->rows : 0;
    matlab_mat *next = mat_alloc(n + 1, 2);
    for (int64_t i = 0; i < n; ++i) {
        next->data[i * 2 + 0] = cur->data[i * 2 + 0];
        next->data[i * 2 + 1] = cur->data[i * 2 + 1];
    }
    next->data[n * 2 + 0] = face_id;
    next->data[n * 2 + 1] = value;
    matlab_struct_set_mat(model, field, field_len, next);
    return model;
}

matlab_struct *matlab_pde_set_face_temperature(matlab_struct *model,
                                                double face_id, double T) {
    return pde_append_face_pair(model, "TemperatureFaces", 16, face_id, T);
}

matlab_struct *matlab_pde_set_face_heat(matlab_struct *model,
                                         double face_id, double q) {
    return pde_append_face_pair(model, "HeatFaces", 9, face_id, q);
}

matlab_struct *matlab_pde_set_face_voltage(matlab_struct *model,
                                            double face_id, double V) {
    return pde_append_face_pair(model, "VoltageFaces", 12, face_id, V);
}

matlab_struct *matlab_pde_set_face_charge(matlab_struct *model,
                                           double face_id, double rho) {
    return pde_append_face_pair(model, "ChargeFaces", 11, face_id, rho);
}

matlab_struct *matlab_pde_set_body_heat(matlab_struct *model, double q) {
    matlab_struct_set_f64(model, "BodyHeat", 8, q);
    return model;
}

matlab_struct *matlab_pde_set_body_charge(matlab_struct *model, double rho) {
    matlab_struct_set_f64(model, "BodyCharge", 10, rho);
    return model;
}

matlab_struct *matlab_pde_set_face_potential(matlab_struct *model,
                                              double face_id, double A) {
    return pde_append_face_pair(model, "MagneticPotentialFaces", 22,
                                 face_id, A);
}

matlab_struct *matlab_pde_set_face_current(matlab_struct *model,
                                            double face_id, double J) {
    return pde_append_face_pair(model, "CurrentFaces", 12, face_id, J);
}

matlab_struct *matlab_pde_set_body_current(matlab_struct *model, double J) {
    matlab_struct_set_f64(model, "BodyCurrent", 11, J);
    return model;
}

/* Test whether a struct field holds a populated value.  Returns 1
 * when the field exists AND its stored pointer is non-null AND, if
 * the stored value is a matlab_mat, it has positive rows.  A missing
 * field comes back from struct_get_mat as `mat_alloc(0, 0)` (an
 * empty matlab_mat with rows == cols == 0), which we detect to
 * distinguish "field never set" from "field set to a real mesh
 * struct that happens to start with a zero-valued first word". */
static bool field_holds_struct(matlab_struct *s, const char *name, int64_t len) {
    matlab_mat *box = matlab_struct_get_mat(s, name, len);
    if (!box) return false;
    /* matlab_mat layout: 8 bytes data ptr + 8 bytes rows + 8 bytes cols.
     * A real struct pointer has the same first 4 bytes interpreted as
     * its first field (or magic).  We disambiguate by checking
     * whether the first 8 bytes look like a heap pointer vs a low
     * integer; matlab_struct's first field is a nfields/capacity
     * pair (int32_t + int32_t).  Simpler: check if box->rows is a
     * plausibly-small int (< 1e9) — if it is, treat the box as a
     * real matlab_mat; if rows is huge (likely a misinterpreted
     * pointer), treat as a struct (so the caller knows it's the
     * struct-shape underneath).
     *
     * In our case we just need to know if the field was set to
     * SOMETHING.  Mesh / Geometry fields, when set, hold a
     * matlab_struct* (kind=1 stored as a generic ptr).  When not
     * set, struct_get_mat returns the freshly-allocated empty
     * matlab_mat with rows=cols=0.  The distinguishing test: rows
     * == 0 AND cols == 0 AND data is NULL means "missing".
     */
    /* mat_alloc(0, 0) returns rows=0 cols=0 (and a calloc'd dummy
     * data pointer).  A struct cast as matlab_mat has rows/cols
     * positions holding heap pointers — huge non-zero values.  Use
     * rows==0 && cols==0 as the empty-field discriminator. */
    if (box->rows == 0 && box->cols == 0) return false;
    return true;
}

matlab_struct *matlab_pde_generate_mesh(matlab_struct *model) {
    /* If Mesh is empty (the kwarg ctor never set it), default to
     * Geometry — for multicuboid / voxelize-style inputs, the
     * geometry struct IS the volumetric mesh. */
    if (!field_holds_struct(model, "Mesh", 4)) {
        matlab_mat *geom_box = matlab_struct_get_mat(model, "Geometry", 8);
        matlab_struct_set_mat(model, "Mesh", 4, geom_box);
    }
    return model;
}

/* ------- Tier-3 scalar AnalysisType kernels --------------------- *
 *
 * Both thermal-steady and electrostatic discretise as
 * `-∇·(c ∇u) + 0·u = f` on the volumetric tet mesh, with c chosen
 * from MaterialProperties (k for thermal, ε for electrostatic), and
 * Dirichlet BCs from FaceBC.Temperature / .Voltage entries (flat
 * TemperatureFaces / VoltageFaces tables).  The same scalar Poisson
 * sparse assembler + PCG path covers both.
 *
 * Returns struct { Mesh, u } — the user wraps it into a
 * ThermalResults / ElectrostaticResults instance via the kwarg-ctor
 * sugar at the call site.
 */

static matlab_struct *solve_scalar_poisson(matlab_struct *model,
                                            double c, double body_f,
                                            const char *bc_table,
                                            int bc_table_len,
                                            const char *flux_table,
                                            int flux_table_len) {
    matlab_struct *mesh = nullptr;
    if (field_holds_struct(model, "Mesh", 4)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Mesh", 4);
    } else if (field_holds_struct(model, "Geometry", 8)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Geometry", 8);
    } else {
        return matlab_struct_new();
    }

    matlab_struct *sys = matlab_pde_assemble_poisson_3d_sparse(mesh, c, 0.0, body_f);
    void *K_sp = matlab_struct_get_mat(sys, "K", 1);
    matlab_mat *F = matlab_pde_sys_F(sys);

    /* Optional surface flux loads. */
    matlab_mat *flux_t = matlab_struct_get_mat(model, flux_table, flux_table_len);
    if (flux_t && flux_t->rows > 0 && flux_t->cols >= 2) {
        for (int64_t i = 0; i < flux_t->rows; ++i) {
            double fid = flux_t->data[i * flux_t->cols + 0];
            double q   = flux_t->data[i * flux_t->cols + 1];
            matlab_mat *Fk = matlab_pde_face_scalar_load_3d(mesh, fid, q);
            for (int64_t k = 0; k < F->rows; ++k) F->data[k] += Fk->data[k];
        }
    }

    /* Walk Dirichlet table; chain successive apply_dirichlet_3d
     * calls so each BC value gets its correct row+col elimination. */
    matlab_mat *bc_t = matlab_struct_get_mat(model, bc_table, bc_table_len);
    void *K_cur = K_sp;
    matlab_mat *F_cur = F;
    if (bc_t && bc_t->rows > 0 && bc_t->cols >= 2) {
        for (int64_t i = 0; i < bc_t->rows; ++i) {
            double fid   = bc_t->data[i * bc_t->cols + 0];
            double u_val = bc_t->data[i * bc_t->cols + 1];
            matlab_mat *ids = matlab_pde_face_nodes(mesh, fid);
            matlab_struct *sys2 = matlab_pde_apply_dirichlet_3d_sparse(
                K_cur, F_cur, ids, u_val);
            K_cur = matlab_struct_get_mat(sys2, "K", 1);
            F_cur = matlab_pde_sys_F(sys2);
        }
    }

    matlab_struct *pcg_res = matlab_sparse_pcg(K_cur, F_cur, 1e-6, 4000);
    matlab_mat *u = matlab_sparse_pcg_x(pcg_res);

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Mesh", 4, (matlab_mat *)mesh);
    matlab_struct_set_mat(out, "u",    1, u);
    return out;
}

matlab_struct *matlab_pde_solve_thermal_steady(matlab_struct *model) {
    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    double k = matlab_struct_get_f64(props, "ThermalConductivity", 19);
    double q_body = matlab_struct_get_f64(model, "BodyHeat", 8);
    return solve_scalar_poisson(model, k, q_body,
                                 "TemperatureFaces", 16,
                                 "HeatFaces", 9);
}

/* matlab_pde_solve_magnetostatic — scalar magnetic vector potential
 * formulation in 2-D / 3-D.  The PDE is
 *   -∇·((1/μ) ∇A_z) = J_z
 * where μ = μ0·μr.  We use μr directly as the K-coefficient (μ0 =
 * 4π·10⁻⁷ is constant and would scale every K entry by the same
 * factor — adding nothing but conditioning grief).  The Dirichlet
 * BC field is the magnetic-potential value at a face; the source
 * is volumetric current density (BodyCurrent) or face current sheet
 * (CurrentFaces table).
 */
matlab_struct *matlab_pde_solve_magnetostatic(matlab_struct *model) {
    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    double mu_r = matlab_struct_get_f64(props, "RelativePermeability", 20);
    if (mu_r <= 0) mu_r = 1.0;
    /* K-coefficient is 1/μr (the "reluctance") so larger μr ↔ softer
     * material ↔ smaller K diagonal. */
    double c = 1.0 / mu_r;
    double J_body = matlab_struct_get_f64(model, "BodyCurrent", 11);
    return solve_scalar_poisson(model, c, J_body,
                                 "MagneticPotentialFaces", 22,
                                 "CurrentFaces", 12);
}

/* matlab_pde_solve_dc_conduction — Ohm's law on the volumetric
 * conductor: -∇·(σ ∇V) = 0 (no internal sources in v1 — current
 * injection through face traction is the standard load).
 *
 * Same shape as electrostatic; uses ElectricalConductivity from the
 * MaterialProperties as the K-coefficient.
 */
matlab_struct *matlab_pde_solve_dc_conduction(matlab_struct *model) {
    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    double sigma = matlab_struct_get_f64(props, "ElectricalConductivity", 22);
    if (sigma <= 0) sigma = 1.0;
    double J_body = matlab_struct_get_f64(model, "BodyCurrent", 11);
    /* Same BC tables as electrostatic — VoltageFaces / ChargeFaces /
     * BodyCharge map to V / J / I_body in the DC-conduction
     * interpretation; the equations are mathematically identical. */
    return solve_scalar_poisson(model, sigma, J_body,
                                 "VoltageFaces", 12,
                                 "ChargeFaces", 11);
}

/* --- structuralTransient via explicit central-difference Newmark - *
 *
 * Solves M·ü + K·u = F(t) on the 3-D linear elasticity tet mesh
 * using a central-difference (Newmark-β with β=0, γ=½) integrator.
 * Lumped mass — M is diagonal so M^-1 is a vector divide.  No
 * damping in v1 (Rayleigh damping = α M + β K lands in a follow-up
 * slice).
 *
 * Time step is set by model.TimeStep (defaults to 1e-5 s — small
 * enough for the elasticity wave-speed CFL on the test meshes).
 * Number of steps from model.NumSteps (default 200).
 *
 * Returns struct { Mesh, u_final, u_history (3N × Nt), tlist (Nt × 1) }.
 */

static void elast_build_K_F_M_diag(matlab_struct *mesh,
                                   double E, double nu, double rho,
                                   void **K_sparse_out,
                                   matlab_mat **M_diag_out,
                                   int64_t *Ndof_out) {
    extern void *matlab_pde_assemble_elast_3d_sparse(matlab_struct *mesh,
                                                     double E, double nu);
    /* K via existing sparse assembler. */
    *K_sparse_out = matlab_pde_assemble_elast_3d_sparse(mesh, E, nu);

    /* Lumped mass: M_ii = ρ · V_inc / 4 summed over incident tets,
     * replicated across the 3 DOFs of each node. */
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *tets  = matlab_struct_get_mat(mesh, "Tets",  4);
    int64_t Nn = nodes->rows;
    int64_t Nt = tets->rows;
    int64_t Ndof = 3 * Nn;
    matlab_mat *M = mat_alloc(Ndof, 1);

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t ids[4];
        double X[4][3];
        for (int i = 0; i < 4; ++i) {
            ids[i] = (int64_t)tets->data[e * 4 + i] - 1;
            X[i][0] = nodes->data[ids[i] * 3 + 0];
            X[i][1] = nodes->data[ids[i] * 3 + 1];
            X[i][2] = nodes->data[ids[i] * 3 + 2];
        }
        double dN[4][3];
        double Vol;
        extern void elast_compute_grad(const double X[4][3], double dN[4][3],
                                        double *vol_out);
        elast_compute_grad(X, dN, &Vol);
        double m_each = rho * Vol / 4.0;
        for (int i = 0; i < 4; ++i) {
            int64_t base = ids[i] * 3;
            M->data[base + 0] += m_each;
            M->data[base + 1] += m_each;
            M->data[base + 2] += m_each;
        }
    }
    *M_diag_out = M;
    *Ndof_out = Ndof;
}

matlab_struct *matlab_pde_set_time_step(matlab_struct *model, double dt) {
    matlab_struct_set_f64(model, "TimeStep", 8, dt);
    return model;
}

matlab_struct *matlab_pde_set_num_steps(matlab_struct *model, double n) {
    matlab_struct_set_f64(model, "NumSteps", 8, n);
    return model;
}

matlab_struct *matlab_pde_solve_structural_transient(matlab_struct *model) {
    matlab_struct *mesh = nullptr;
    if (field_holds_struct(model, "Mesh", 4)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Mesh", 4);
    } else if (field_holds_struct(model, "Geometry", 8)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Geometry", 8);
    } else {
        return matlab_struct_new();
    }

    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    double E   = matlab_struct_get_f64(props, "YoungsModulus", 13);
    double nu  = matlab_struct_get_f64(props, "PoissonsRatio", 13);
    double rho = matlab_struct_get_f64(props, "MassDensity",   11);
    if (rho <= 0) rho = 1.0;

    double dt = matlab_struct_get_f64(model, "TimeStep", 8);
    if (dt <= 0) dt = 1e-5;
    int64_t nsteps = (int64_t)matlab_struct_get_f64(model, "NumSteps", 8);
    if (nsteps <= 0) nsteps = 200;

    void *K_sp = nullptr;
    matlab_mat *Mdiag = nullptr;
    int64_t Ndof = 0;
    elast_build_K_F_M_diag(mesh, E, nu, rho, &K_sp, &Mdiag, &Ndof);

    /* Static F from pressure faces. */
    matlab_mat *F = mat_alloc(Ndof, 1);
    matlab_mat *pf = matlab_struct_get_mat(model, "PressureFaces", 13);
    if (pf && pf->rows > 0 && pf->cols >= 2) {
        for (int64_t i = 0; i < pf->rows; ++i) {
            double fid = pf->data[i * pf->cols + 0];
            double p   = pf->data[i * pf->cols + 1];
            matlab_mat *Fk = matlab_pde_face_pressure_3d(mesh, fid, p);
            for (int64_t k = 0; k < Ndof; ++k) F->data[k] += Fk->data[k];
        }
    }

    /* Build the union of fixed DOFs (vector elasticity: 3 per node). */
    matlab_mat *ff = matlab_struct_get_mat(model, "FixedFaces", 10);
    std::vector<int8_t> fixed_dof((size_t)Ndof, 0);
    if (ff && ff->rows > 0) {
        for (int64_t i = 0; i < ff->rows; ++i) {
            double fid = ff->data[i];
            matlab_mat *ids = matlab_pde_face_nodes(mesh, fid);
            for (int64_t k = 0; k < ids->rows; ++k) {
                int64_t n = (int64_t)ids->data[k] - 1;
                if (n < 0 || n * 3 + 2 >= Ndof) continue;
                fixed_dof[(size_t)(n * 3 + 0)] = 1;
                fixed_dof[(size_t)(n * 3 + 1)] = 1;
                fixed_dof[(size_t)(n * 3 + 2)] = 1;
            }
        }
    }

    /* Central-difference time stepping.  Start from rest (u=v=0).
     *   a_n  = M^{-1} (F - K u_n)
     *   v_{n+1/2} = v_{n-1/2} + dt · a_n  (with v_{-1/2} = 0)
     *   u_{n+1}   = u_n + dt · v_{n+1/2}
     *   apply Dirichlet (u_{n+1}[fixed] = 0, v_{n+1/2}[fixed] = 0).
     */
    std::vector<double> u((size_t)Ndof, 0.0);
    std::vector<double> v((size_t)Ndof, 0.0);
    std::vector<double> Ku((size_t)Ndof, 0.0);

    /* History buffer for the displacement.  3N × (nsteps + 1).  For
     * large meshes / many timesteps this gets big; users can drop
     * NumSteps to keep it tractable. */
    matlab_mat *Uhist = mat_alloc(Ndof, nsteps + 1);
    /* t = 0 column is all zeros (rest start). */

    struct sparse_view {
        uint32_t magic, _pad;
        int64_t *row_ptr;
        int64_t *col_idx;
        double  *vals;
        int64_t rows, cols, nnz;
    };
    sparse_view *S = (sparse_view *)K_sp;

    for (int64_t step = 0; step < nsteps; ++step) {
        /* Compute K · u. */
        for (int64_t r = 0; r < Ndof; ++r) {
            double s = 0.0;
            int64_t lo = S->row_ptr[r];
            int64_t hi = S->row_ptr[r + 1];
            for (int64_t k = lo; k < hi; ++k)
                s += S->vals[k] * u[(size_t)S->col_idx[k]];
            Ku[(size_t)r] = s;
        }
        /* a_n = M^{-1} · (F - K u). */
        for (int64_t i = 0; i < Ndof; ++i) {
            double a = (F->data[i] - Ku[(size_t)i]) / Mdiag->data[i];
            v[(size_t)i] += dt * a;
        }
        /* u_{n+1} = u_n + dt · v_{n+1/2}. */
        for (int64_t i = 0; i < Ndof; ++i) {
            u[(size_t)i] += dt * v[(size_t)i];
            if (fixed_dof[(size_t)i]) {
                u[(size_t)i] = 0.0;
                v[(size_t)i] = 0.0;
            }
        }
        /* Snapshot into Uhist column (step + 1). */
        for (int64_t i = 0; i < Ndof; ++i)
            Uhist->data[i * (nsteps + 1) + (step + 1)] = u[(size_t)i];
    }

    /* tlist column. */
    matlab_mat *tlist = mat_alloc(nsteps + 1, 1);
    for (int64_t i = 0; i <= nsteps; ++i) tlist->data[i] = (double)i * dt;

    /* Final-step displacement for legacy single-call sites. */
    matlab_mat *u_final = mat_alloc(Ndof, 1);
    memcpy(u_final->data, u.data(), sizeof(double) * (size_t)Ndof);

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Mesh", 4, (matlab_mat *)mesh);
    matlab_struct_set_mat(out, "u",    1, u_final);
    matlab_struct_set_mat(out, "Uhist", 5, Uhist);
    matlab_struct_set_mat(out, "tlist", 5, tlist);
    return out;
}

matlab_mat *matlab_pde_kernel_uhist(matlab_struct *r) {
    return matlab_struct_get_mat(r, "Uhist", 5);
}
matlab_mat *matlab_pde_kernel_tlist(matlab_struct *r) {
    return matlab_struct_get_mat(r, "tlist", 5);
}

/* --- structuralModal — generalised eigenvalue K φ = λ M φ -------- *
 *
 * For v1 we run UNCONSTRAINED modal analysis (no Dirichlet).  The
 * first 6 eigenvalues are near-zero rigid-body modes; physical
 * flexible modes start at index 7.  Matches the MathWorks doc
 * convention for the tuning-fork / wing-spar examples.
 *
 * Uses the existing pde_eigsmall inverse-iteration solver, which
 * expects DENSE K and M.  Practical ceiling is ~300 DOFs (the
 * dense LU inside inverse iteration costs O(N³) per mode).
 * Production-quality modal at scale needs Krylov-Schur with
 * shift-invert (roadmap §10.5 follow-up).
 *
 * Returns struct { Mesh, NaturalFrequencies (Hz, n×1), ModeShapes (3N×n) }.
 */

extern matlab_mat *matlab_pde_assemble_elast_3d(matlab_struct *mesh,
                                                 double E, double nu);
extern matlab_mat *matlab_pde_eigsmall(matlab_mat *K, matlab_mat *M,
                                        double nmodes_d);

matlab_struct *matlab_pde_set_num_modes(matlab_struct *model, double n) {
    matlab_struct_set_f64(model, "NumModes", 8, n);
    return model;
}

matlab_struct *matlab_pde_solve_structural_modal(matlab_struct *model) {
    matlab_struct *mesh = nullptr;
    if (field_holds_struct(model, "Mesh", 4)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Mesh", 4);
    } else if (field_holds_struct(model, "Geometry", 8)) {
        mesh = (matlab_struct *)matlab_struct_get_mat(model, "Geometry", 8);
    } else {
        return matlab_struct_new();
    }
    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    double E   = matlab_struct_get_f64(props, "YoungsModulus", 13);
    double nu  = matlab_struct_get_f64(props, "PoissonsRatio", 13);
    double rho = matlab_struct_get_f64(props, "MassDensity",   11);
    if (rho <= 0) rho = 1.0;
    int64_t nmodes = (int64_t)matlab_struct_get_f64(model, "NumModes", 8);
    if (nmodes <= 0) nmodes = 10;

    /* Dense K from the existing assembler. */
    matlab_mat *K = matlab_pde_assemble_elast_3d(mesh, E, nu);
    int64_t Ndof = K->rows;

    /* Build a DENSE lumped mass matrix from the per-DOF diagonal. */
    matlab_mat *Mdiag = nullptr;
    int64_t junk = 0;
    void *Ksp_unused = nullptr;
    elast_build_K_F_M_diag(mesh, E, nu, rho, &Ksp_unused, &Mdiag, &junk);
    matlab_mat *M = mat_alloc(Ndof, Ndof);
    for (int64_t i = 0; i < Ndof; ++i) M->data[i * Ndof + i] = Mdiag->data[i];

    matlab_mat *lams = matlab_pde_eigsmall(K, M, (double)nmodes);

    /* Convert λ = ω² (rad/s)² to natural frequencies in Hz. */
    matlab_mat *freqs = mat_alloc(nmodes, 1);
    for (int64_t i = 0; i < nmodes; ++i) {
        double l = lams->data[i];
        if (l < 0) l = 0;
        freqs->data[i] = sqrt(l) / (2.0 * M_PI);
    }

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Mesh", 4, (matlab_mat *)mesh);
    matlab_struct_set_mat(out, "NaturalFrequencies", 18, freqs);
    /* Eigenvectors are produced by pde_eigsmall but only the lambdas
     * are returned (the function discards the modes vector after the
     * deflation pass).  Mode shapes ship as a Tier-3 follow-up; v1
     * exposes frequencies only — adequate for the cantilever / tuning-
     * fork validation. */
    return out;
}

matlab_mat *matlab_pde_kernel_freqs(matlab_struct *r) {
    return matlab_struct_get_mat(r, "NaturalFrequencies", 18);
}

matlab_struct *matlab_pde_solve_electrostatic(matlab_struct *model) {
    matlab_struct *props = (matlab_struct *)
        matlab_struct_get_mat(model, "MaterialProperties", 18);
    /* For a pure Laplace problem with Dirichlet BCs and no body
     * charge, the solution V depends only on the BC ratio, not on
     * ε.  Using the raw ε = ε0·εr ≈ 1e-11 makes the assembled K
     * matrix ~1e-11 in magnitude; PCG with default tol=1e-6
     * converges at numerical noise and the free DOFs stay at zero.
     *
     * To preserve numerical conditioning we use the dimensionless εr
     * (or 1.0 by default) as the K coefficient.  The Dirichlet
     * voltage values pass through unchanged.  When body charge is
     * present, scale ρ by 1/ε0 so the source term has the right
     * units relative to the K-coefficient = εr we picked. */
    double eps_r = matlab_struct_get_f64(props, "RelativePermittivity", 20);
    if (eps_r <= 0) eps_r = 1.0;
    double rho_body_raw = matlab_struct_get_f64(model, "BodyCharge", 10);
    double eps0 = 8.8541878128e-12;
    double rho_body = rho_body_raw / eps0;
    return solve_scalar_poisson(model, eps_r, rho_body,
                                 "VoltageFaces", 12,
                                 "ChargeFaces", 11);
}

/* solve(model) — the unified entry point.  Reads AnalysisType +
 * dispatches to the appropriate kernel.
 *
 * v1 dispatches:
 *   'structuralStatic'    → 3-D linear elasticity (existing)
 *   'thermalSteadyState'  → 3-D Poisson with thermal conductivity
 *   'electrostatic'       → 3-D Poisson with permittivity
 *
 * The result is a matlab_struct whose fields match the relevant
 * Result class layout (.Mesh + .u + .vm for structural,
 * .Mesh + .u for thermal/electrostatic).  The MATLAB-side wrapper
 * then constructs the typed StaticStructuralResults / ThermalResults
 * / ElectrostaticResults instance via the kwarg-ctor sugar.
 */
matlab_struct *matlab_pde_solve(matlab_struct *model) {
    /* AnalysisType is stored as a matlab_string under kind=3 — read
     * its char data manually. */
    struct local_str { char *data; int64_t len; };
    matlab_mat *at_box = matlab_struct_get_mat(model, "AnalysisType", 12);
    /* kind=3 storage returns the matlab_string ptr verbatim. */
    if (at_box) {
        local_str *s = (local_str *)at_box;
        /* matlab_string layout: {char *data, int64_t len}.  We
         * pattern-match on prefix bytes. */
        if (s->data && s->len > 0) {
            if (s->len == 18 && memcmp(s->data, "thermalSteadyState", 18) == 0)
                return matlab_pde_solve_thermal_steady(model);
            if (s->len == 13 && memcmp(s->data, "electrostatic", 13) == 0)
                return matlab_pde_solve_electrostatic(model);
            if (s->len == 13 && memcmp(s->data, "magnetostatic", 13) == 0)
                return matlab_pde_solve_magnetostatic(model);
            if (s->len == 12 && memcmp(s->data, "dcConduction", 12) == 0)
                return matlab_pde_solve_dc_conduction(model);
            if (s->len == 19 &&
                memcmp(s->data, "structuralTransient", 19) == 0)
                return matlab_pde_solve_structural_transient(model);
            if (s->len == 15 &&
                memcmp(s->data, "structuralModal", 15) == 0)
                return matlab_pde_solve_structural_modal(model);
        }
    }
    return matlab_pde_solve_femodel(model);
}

/* Per-node von Mises stress.  Averages the per-tet vM over all tets
 * incident to each node — adequate for visualisation.  Returns
 * Nn × 1.  Bypass the Cauchy-stress reconstruction by reusing the
 * existing element-vM result and averaging.  Tracks an incidence
 * count per node so we divide by the right number.
 */
matlab_mat *matlab_pde_node_von_mises_3d(matlab_struct *mesh, matlab_mat *u_flat,
                                         double E, double nu) {
    matlab_mat *vm_tet = matlab_pde_von_mises_3d(mesh, u_flat, E, nu);
    matlab_mat *tets   = matlab_struct_get_mat(mesh, "Tets",  4);
    matlab_mat *nodes  = matlab_struct_get_mat(mesh, "Nodes", 5);
    int64_t Nn = nodes->rows;
    int64_t Nt = tets->rows;
    matlab_mat *vm_node = mat_alloc(Nn, 1);
    std::vector<int64_t> cnt((size_t)Nn, 0);
    for (int64_t e = 0; e < Nt; ++e) {
        double v = vm_tet->data[e];
        for (int k = 0; k < 4; ++k) {
            int64_t n = (int64_t)tets->data[e * 4 + k] - 1;
            if (n < 0 || n >= Nn) continue;
            vm_node->data[n] += v;
            cnt[(size_t)n] += 1;
        }
    }
    for (int64_t i = 0; i < Nn; ++i) {
        if (cnt[(size_t)i] > 0) vm_node->data[i] /= (double)cnt[(size_t)i];
    }
    return vm_node;
}

/* Convenience: max absolute displacement magnitude across all nodes.
 * u_flat is 3N×1. */
double matlab_pde_peak_disp_3d(matlab_mat *u_flat) {
    int64_t Ndof = u_flat->rows * u_flat->cols;
    int64_t Nn = Ndof / 3;
    double mx = 0.0;
    for (int64_t i = 0; i < Nn; ++i) {
        double ux = u_flat->data[i * 3 + 0];
        double uy = u_flat->data[i * 3 + 1];
        double uz = u_flat->data[i * 3 + 2];
        double m = sqrt(ux * ux + uy * uy + uz * uz);
        if (m > mx) mx = m;
    }
    return mx;
}

/* --- Tier-3: 2-D mass matrix + transient parabolic step ------------ *
 *
 * For the parabolic equation  d · ∂u/∂t − ∇·(c∇u) + au = f  on a 2D
 * triangulated domain with Dirichlet BC, we assemble the consistent
 * mass matrix M alongside K, then time-integrate
 *   M · u_{n+1} = (M − dt·K) · u_n  +  dt · F        (forward Euler)
 * for each step.  For the gating Tier-3 smoke test we use forward
 * Euler — it's stable when dt < 2 / λ_max(M^{-1} K).  The user can
 * pick a smaller dt or switch to Crank-Nicolson if needed.
 *
 * Returns a 2-field struct {M, K, F}.  c, a, f are scalar constants
 * (same shape as Tier-1's pde_assemble_poisson_2d).
 */
matlab_struct *matlab_pde_assemble_transient_2d(matlab_struct *mesh,
                                                double c, double a, double f) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes",     5);
    matlab_mat *tris  = matlab_struct_get_mat(mesh, "Triangles", 9);
    int64_t Nn = nodes->rows;
    int64_t Nt = tris->rows;

    matlab_mat *K = mat_alloc(Nn, Nn);
    matlab_mat *M = mat_alloc(Nn, Nn);
    matlab_mat *F = mat_alloc(Nn, 1);

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t i0 = (int64_t)tris->data[e * 3 + 0] - 1;
        int64_t i1 = (int64_t)tris->data[e * 3 + 1] - 1;
        int64_t i2 = (int64_t)tris->data[e * 3 + 2] - 1;
        double x0 = nodes->data[i0 * 2 + 0], y0 = nodes->data[i0 * 2 + 1];
        double x1 = nodes->data[i1 * 2 + 0], y1 = nodes->data[i1 * 2 + 1];
        double x2 = nodes->data[i2 * 2 + 0], y2 = nodes->data[i2 * 2 + 1];
        double twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        double area = 0.5 * twoA;
        if (area <= 0) area = -area;
        double b[3] = { (y1 - y2), (y2 - y0), (y0 - y1) };
        double cc[3] = { (x2 - x1), (x0 - x2), (x1 - x0) };
        double inv2A = 1.0 / twoA;
        for (int p = 0; p < 3; ++p) { b[p] *= inv2A; cc[p] *= inv2A; }
        int64_t loc[3] = { i0, i1, i2 };
        /* Consistent mass: Me_pq = area/12 if p≠q, area/6 if p==q. */
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) {
                double Ke = c * area * (b[p] * b[q] + cc[p] * cc[q]);
                if (p == q) Ke += a * area / 3.0;
                K->data[loc[p] * Nn + loc[q]] += Ke;
                double Me = (p == q) ? area / 6.0 : area / 12.0;
                M->data[loc[p] * Nn + loc[q]] += Me;
            }
            F->data[loc[p]] += f * area / 3.0;
        }
    }

    /* Mass-lumping: replace M with diag(row_sums(M)).  Forward Euler
     * needs a diagonal mass matrix to stay tractable + stable. */
    for (int64_t i = 0; i < Nn; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < Nn; ++j) s += M->data[i * Nn + j];
        for (int64_t j = 0; j < Nn; ++j) M->data[i * Nn + j] = 0.0;
        M->data[i * Nn + i] = s;
    }

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "M", 1, M);
    matlab_struct_set_mat(out, "K", 1, K);
    matlab_struct_set_mat(out, "F", 1, F);
    return out;
}

/* Generalised symmetric eigenvalue solver for K φ = λ M φ.  Implemented
 * via M^(1/2)-orthogonalisation: solve M w = K v iteratively (subspace
 * power iteration) for the smallest k modes — sufficient for the
 * Tier-3 modal smoke test on small (≤ 300 DOF) systems.
 *
 * For very small problems this just calls mldivide repeatedly; not
 * fast for production-scale modal analysis but matches the
 * structuralModal call shape and lets the test exercise the full
 * "Tier-3 modal" pipeline.
 *
 * Returns the k smallest eigenvalues as a column vector.  The
 * eigenvectors are recovered from the last iterate.
 *
 * Implementation: inverse-iteration with deflation.  At each step,
 *   1. Solve K w = M v_k via mldivide.
 *   2. Orthogonalise w against accumulated modes.
 *   3. Normalise w / sqrt(w' M w).
 *   4. λ_k = (w' K w) / (w' M w).
 * Iterates 30 times per mode — adequate for well-separated spectra.
 */
extern matlab_mat *matlab_mldivide_mm(matlab_mat *A, matlab_mat *B);

static double dot_mat(const double *a, const double *b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static void mat_vec(const double *A, const double *x, double *y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < n; ++j) s += A[i * n + j] * x[j];
        y[i] = s;
    }
}

matlab_mat *matlab_pde_eigsmall(matlab_mat *K, matlab_mat *M, double nmodes_d) {
    int64_t n = K->rows;
    int64_t nmodes = (int64_t)nmodes_d;
    if (nmodes < 1) nmodes = 1;
    if (nmodes > n) nmodes = n;

    matlab_mat *lams = mat_alloc(nmodes, 1);

    std::vector<std::vector<double>> modes;  /* accumulated mode vectors */
    std::vector<double> Mv(n), Kv(n), w(n), v(n);

    /* Seed RNG-free: deterministic start vectors with mode index as a phase. */
    for (int64_t k = 0; k < nmodes; ++k) {
        for (int64_t i = 0; i < n; ++i) {
            v[i] = sin((double)(i + 1) * (double)(k + 1) * 0.13);
        }
        /* Orthogonalise against accumulated modes (M-orthogonal). */
        for (auto &m : modes) {
            mat_vec(M->data, m.data(), Mv.data(), n);
            double a = dot_mat(v.data(), Mv.data(), n);
            for (int64_t i = 0; i < n; ++i) v[i] -= a * m[i];
        }
        for (int iter = 0; iter < 50; ++iter) {
            /* Inverse iteration: solve K w = M v. */
            mat_vec(M->data, v.data(), Mv.data(), n);
            matlab_mat *rhs = mat_alloc(n, 1);
            memcpy(rhs->data, Mv.data(), sizeof(double) * (size_t)n);
            matlab_mat *sol = matlab_mldivide_mm(K, rhs);
            if (sol->rows != n) {
                /* Singular K → bail with zero entry. */
                lams->data[k] = 0.0;
                break;
            }
            memcpy(w.data(), sol->data, sizeof(double) * (size_t)n);
            /* Deflate against previous modes. */
            for (auto &m : modes) {
                mat_vec(M->data, m.data(), Mv.data(), n);
                double a = dot_mat(w.data(), Mv.data(), n);
                for (int64_t i = 0; i < n; ++i) w[i] -= a * m[i];
            }
            /* Normalise w / sqrt(w' M w). */
            mat_vec(M->data, w.data(), Mv.data(), n);
            double nrm = sqrt(dot_mat(w.data(), Mv.data(), n));
            if (nrm > 0) for (int64_t i = 0; i < n; ++i) w[i] /= nrm;
            v.swap(w);
        }
        /* λ = v' K v / v' M v. */
        mat_vec(K->data, v.data(), Kv.data(), n);
        mat_vec(M->data, v.data(), Mv.data(), n);
        double num = dot_mat(v.data(), Kv.data(), n);
        double den = dot_mat(v.data(), Mv.data(), n);
        lams->data[k] = (den > 0) ? num / den : 0.0;
        modes.push_back(v);
    }
    return lams;
}

/* Forward Euler step for the parabolic equation:
 *   u_{n+1} = u_n + dt · M^{-1} · (F − K · u_n)
 *
 * For the Tier-3 smoke test we apply this directly with Dirichlet u=0
 * on the boundary nodes after each step.  Returns the u_{n+1} vector
 * (Nn × 1).  M, K, F are dense and come from
 * pde_assemble_transient_2d.
 */
matlab_mat *matlab_pde_step_forward_euler_2d(matlab_mat *M, matlab_mat *K,
                                             matlab_mat *F, matlab_mat *u,
                                             matlab_mat *bnd, double dt) {
    int64_t Nn = M->rows;
    matlab_mat *Ku = mat_alloc(Nn, 1);
    mat_vec(K->data, u->data, Ku->data, Nn);
    matlab_mat *rhs = mat_alloc(Nn, 1);
    for (int64_t i = 0; i < Nn; ++i) rhs->data[i] = F->data[i] - Ku->data[i];
    /* w = M \ rhs. */
    matlab_mat *w = matlab_mldivide_mm(M, rhs);
    matlab_mat *u_next = mat_alloc(Nn, 1);
    for (int64_t i = 0; i < Nn; ++i) u_next->data[i] = u->data[i] + dt * w->data[i];
    /* Apply Dirichlet u = 0 on boundary nodes. */
    int64_t Nb = bnd->rows * bnd->cols;
    for (int64_t k = 0; k < Nb; ++k) {
        int64_t n = (int64_t)bnd->data[k] - 1;
        if (n >= 0 && n < Nn) u_next->data[n] = 0.0;
    }
    return u_next;
}

/* Initial-condition vector: u = u_init at all nodes, 0 on Dirichlet
 * boundary.  Returns Nn × 1. */
matlab_mat *matlab_pde_init_uniform_2d(matlab_struct *mesh,
                                       double u_init, matlab_mat *bnd) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    int64_t Nn = nodes->rows;
    matlab_mat *u = mat_alloc(Nn, 1);
    for (int64_t i = 0; i < Nn; ++i) u->data[i] = u_init;
    int64_t Nb = bnd ? bnd->rows * bnd->cols : 0;
    for (int64_t k = 0; k < Nb; ++k) {
        int64_t n = (int64_t)bnd->data[k] - 1;
        if (n >= 0 && n < Nn) u->data[n] = 0.0;
    }
    return u;
}

/* --- Tier-4: Picard iteration for nonlinear c(u) -------------------- *
 *
 * Solves -∇·(c(u)∇u) = f with Dirichlet u=0 by repeatedly:
 *   1. Assemble K with c = c(u_k) at each element's centroid
 *   2. Apply Dirichlet
 *   3. Solve u_{k+1} = K \ F
 *   4. Stop when ||u_{k+1} - u_k||_inf < tol
 *
 * c_func selector:
 *   0  →  c(u) = c0 (linear baseline)
 *   1  →  c(u) = c0 * (1 + alpha * u^2)
 *
 * Returns a struct {Solution, NumIters, Resid}.
 */
matlab_struct *matlab_pde_solve_nonlinear_2d(matlab_struct *mesh,
                                             double c0, double alpha,
                                             double f, double c_func_d) {
    int c_func = (int)c_func_d;
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes",     5);
    matlab_mat *tris  = matlab_struct_get_mat(mesh, "Triangles", 9);
    int64_t Nn = nodes->rows;
    int64_t Nt = tris->rows;

    matlab_mat *bnd = matlab_pde_boundary_nodes_rect(mesh);
    matlab_mat *u    = mat_alloc(Nn, 1);  /* u_0 = 0 */
    double last_resid = 0.0;
    int64_t max_iter  = 30;
    int64_t iter      = 0;
    double tol        = 1e-7;

    for (iter = 0; iter < max_iter; ++iter) {
        /* Assemble K with c(u) at element centroids. */
        matlab_mat *K = mat_alloc(Nn, Nn);
        matlab_mat *F = mat_alloc(Nn, 1);

        for (int64_t e = 0; e < Nt; ++e) {
            int64_t i0 = (int64_t)tris->data[e * 3 + 0] - 1;
            int64_t i1 = (int64_t)tris->data[e * 3 + 1] - 1;
            int64_t i2 = (int64_t)tris->data[e * 3 + 2] - 1;
            double x0 = nodes->data[i0 * 2 + 0], y0 = nodes->data[i0 * 2 + 1];
            double x1 = nodes->data[i1 * 2 + 0], y1 = nodes->data[i1 * 2 + 1];
            double x2 = nodes->data[i2 * 2 + 0], y2 = nodes->data[i2 * 2 + 1];
            double twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            double area = 0.5 * twoA;
            if (area <= 0) area = -area;
            double b[3] = { (y1 - y2), (y2 - y0), (y0 - y1) };
            double cc[3] = { (x2 - x1), (x0 - x2), (x1 - x0) };
            double inv2A = 1.0 / twoA;
            for (int p = 0; p < 3; ++p) { b[p] *= inv2A; cc[p] *= inv2A; }
            /* Centroid u: average of 3 corner values. */
            double u_c = (u->data[i0] + u->data[i1] + u->data[i2]) / 3.0;
            double c_eff = c0;
            if (c_func == 1) c_eff = c0 * (1.0 + alpha * u_c * u_c);
            int64_t loc[3] = { i0, i1, i2 };
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    double Ke = c_eff * area * (b[p] * b[q] + cc[p] * cc[q]);
                    K->data[loc[p] * Nn + loc[q]] += Ke;
                }
                F->data[loc[p]] += f * area / 3.0;
            }
        }
        /* Apply Dirichlet u = 0. */
        int64_t Nb = bnd->rows * bnd->cols;
        std::vector<int8_t> fixed((size_t)Nn, 0);
        for (int64_t k = 0; k < Nb; ++k) {
            int64_t n = (int64_t)bnd->data[k] - 1;
            if (n >= 0 && n < Nn) fixed[(size_t)n] = 1;
        }
        for (int64_t n = 0; n < Nn; ++n) {
            if (!fixed[(size_t)n]) continue;
            for (int64_t c = 0; c < Nn; ++c) {
                K->data[n * Nn + c] = 0.0;
                K->data[c * Nn + n] = 0.0;
            }
            K->data[n * Nn + n] = 1.0;
            F->data[n] = 0.0;
        }
        matlab_mat *u_new = matlab_mldivide_mm(K, F);
        double max_diff = 0.0;
        for (int64_t i = 0; i < Nn; ++i) {
            double d = u_new->data[i] - u->data[i];
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
        }
        memcpy(u->data, u_new->data, sizeof(double) * (size_t)Nn);
        last_resid = max_diff;
        if (max_diff < tol) { iter++; break; }
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Solution", 8, u);
    matlab_struct_set_f64(out, "NumIters", 8, (double)iter);
    matlab_struct_set_f64(out, "Resid",    5, last_resid);
    return out;
}

/* Accessor for nonlinear-solve result Solution column. */
matlab_mat *matlab_pde_result_solution(matlab_struct *r) {
    return matlab_struct_get_mat(r, "Solution", 8);
}
double matlab_pde_result_num_iters(matlab_struct *r) {
    return matlab_struct_get_f64(r, "NumIters", 8);
}
double matlab_pde_result_resid(matlab_struct *r) {
    return matlab_struct_get_f64(r, "Resid", 5);
}

}  /* extern "C" */

/* ===================================================================
 * Geometry importers: STL (ASCII + binary), GLB (glTF 2.0 binary).
 *
 * Both produce a `surface fegeometry` struct compatible with the
 * pdeplot3D renderer:
 *     .Nodes      Nx3 doubles (x, y, z)
 *     .Faces      Fx4 doubles ([face_id n1 n2 n3] — face_id=1 since
 *                  STL/GLB have no region tags by default)
 *     .IsSurface  1 (no Tets — only the boundary triangulation)
 *
 * Vertex welding uses a hash-by-quantized-coordinate to merge
 * coincident vertices from per-triangle records.
 * =================================================================== */

#include <fstream>
#include <unordered_map>

namespace {

struct V3 {
    double x, y, z;
    bool operator==(const V3 &o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};
struct V3Hash {
    size_t operator()(const V3 &v) const {
        /* Round to 1e-9 precision to merge nearly-coincident vertices.
         * STL fields are float, so 1e-7 of unit scale is the resolution
         * floor.  GLB positions are usually f32 too. */
        int64_t ix = (int64_t)llrint(v.x * 1e9);
        int64_t iy = (int64_t)llrint(v.y * 1e9);
        int64_t iz = (int64_t)llrint(v.z * 1e9);
        size_t h = (size_t)ix;
        h ^= (size_t)iy + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= (size_t)iz + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};
struct V3EqQ {
    bool operator()(const V3 &a, const V3 &b) const {
        return llrint(a.x * 1e9) == llrint(b.x * 1e9) &&
               llrint(a.y * 1e9) == llrint(b.y * 1e9) &&
               llrint(a.z * 1e9) == llrint(b.z * 1e9);
    }
};

struct MeshBuilder {
    std::vector<double> nodes;            /* Nx3 row-major (xyz triples) */
    std::vector<int64_t> tris;            /* Tx3 1-based node ids */
    std::unordered_map<V3, int64_t, V3Hash, V3EqQ> dedupe;

    /* Add a triangle (a, b, c) — welds each vertex via the dedupe map. */
    void add_triangle(double ax, double ay, double az,
                      double bx, double by, double bz,
                      double cx, double cy, double cz) {
        auto add_vertex = [&](double x, double y, double z) -> int64_t {
            V3 key{x, y, z};
            auto it = dedupe.find(key);
            if (it != dedupe.end()) return it->second;
            int64_t id = (int64_t)(nodes.size() / 3) + 1;  /* 1-based */
            nodes.push_back(x);
            nodes.push_back(y);
            nodes.push_back(z);
            dedupe[key] = id;
            return id;
        };
        int64_t i0 = add_vertex(ax, ay, az);
        int64_t i1 = add_vertex(bx, by, bz);
        int64_t i2 = add_vertex(cx, cy, cz);
        /* Skip degenerate triangles. */
        if (i0 == i1 || i1 == i2 || i0 == i2) return;
        tris.push_back(i0);
        tris.push_back(i1);
        tris.push_back(i2);
    }
};

/* Build the final struct from a MeshBuilder. */
static matlab_struct *finalize_mesh(MeshBuilder &mb, const char *source_tag) {
    int64_t Nn = (int64_t)(mb.nodes.size() / 3);
    int64_t Nf = (int64_t)(mb.tris.size()  / 3);
    matlab_mat *nodes = mat_alloc(Nn, 3);
    matlab_mat *faces = mat_alloc(Nf, 4);
    for (int64_t i = 0; i < Nn; ++i) {
        nodes->data[i * 3 + 0] = mb.nodes[(size_t)(i * 3 + 0)];
        nodes->data[i * 3 + 1] = mb.nodes[(size_t)(i * 3 + 1)];
        nodes->data[i * 3 + 2] = mb.nodes[(size_t)(i * 3 + 2)];
    }
    for (int64_t i = 0; i < Nf; ++i) {
        faces->data[i * 4 + 0] = 1.0;                   /* face_id = 1 */
        faces->data[i * 4 + 1] = (double)mb.tris[(size_t)(i * 3 + 0)];
        faces->data[i * 4 + 2] = (double)mb.tris[(size_t)(i * 3 + 1)];
        faces->data[i * 4 + 3] = (double)mb.tris[(size_t)(i * 3 + 2)];
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Nodes",     5, nodes);
    matlab_struct_set_mat(out, "Faces",     5, faces);
    matlab_struct_set_f64(out, "IsSurface", 9, 1.0);
    matlab_struct_set_f64(out, "NumNodes",  8, (double)Nn);
    matlab_struct_set_f64(out, "NumFaces",  8, (double)Nf);
    (void)source_tag;  /* tag retained for future provenance use */
    return out;
}

static inline uint32_t rd_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline float rd_f32_le(const uint8_t *p) {
    uint32_t b = rd_u32_le(p);
    float v;
    memcpy(&v, &b, 4);
    return v;
}

static bool stl_is_binary(const std::vector<uint8_t> &buf) {
    /* Heuristic: binary STL has exactly 84 + 50*N bytes.  ASCII STL
     * starts with "solid" (case-insensitive) and contains "facet"
     * keyword as text.  Some binary STLs lie about starting with
     * "solid", so check the size match first. */
    if (buf.size() < 84) return false;
    uint32_t n = rd_u32_le(buf.data() + 80);
    if ((size_t)(84 + 50ULL * (uint64_t)n) == buf.size()) return true;
    /* Not exact size match — fall back to keyword sniff. */
    bool starts_solid = buf.size() >= 5 &&
        (buf[0] == 's' || buf[0] == 'S') &&
        (buf[1] == 'o' || buf[1] == 'O') &&
        (buf[2] == 'l' || buf[2] == 'L') &&
        (buf[3] == 'i' || buf[3] == 'I') &&
        (buf[4] == 'd' || buf[4] == 'D');
    return !starts_solid;
}

/* Strip leading whitespace from the iterator. */
static const char *skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' ||
                       *p == '\r' || *p == '\n')) ++p;
    return p;
}
/* Parse a double; advances p past the number.  Returns false on no-number. */
static bool parse_double(const char *&p, const char *end, double &out) {
    p = skip_ws(p, end);
    char *q = nullptr;
    out = strtod(p, &q);
    if (q == p) return false;
    p = q;
    if (p > end) p = end;
    return true;
}

}  /* anonymous namespace */

extern "C" {

/* matlab_string layout (mirrors matlab_runtime.cpp's matlab_string_s).
 * Forward-declared here so the loaders can accept the MATLAB string
 * ABI directly without depending on matlab_runtime.h's typedefs. */
struct matlab_string_local_s { char *data; int64_t len; };

/* matlab_pde_load_stl(filename) — read either ASCII or binary STL.
 * Auto-detects format.  Returns the standard surface fegeometry
 * struct, or NULL on failure.  Accepts a matlab_string* (the
 * standard MATLAB string ABI) and also the (const char *, int64_t)
 * shape via the wrapper below. */
matlab_struct *matlab_pde_load_stl_path(const char *path, int64_t plen) {
    std::string fn(path, (size_t)plen);
    std::ifstream f(fn, std::ios::binary);
    if (!f) return nullptr;
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    if (sz <= 0) return nullptr;
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf((size_t)sz);
    if (!f.read((char *)buf.data(), sz)) return nullptr;

    MeshBuilder mb;
    if (stl_is_binary(buf)) {
        uint32_t n = rd_u32_le(buf.data() + 80);
        if (84 + 50ULL * n > buf.size()) return nullptr;
        for (uint32_t i = 0; i < n; ++i) {
            const uint8_t *p = buf.data() + 84 + (size_t)i * 50;
            /* Skip 12-byte normal, then read 3 vertices x 3 floats. */
            double v[9];
            for (int k = 0; k < 9; ++k) v[k] = (double)rd_f32_le(p + 12 + k * 4);
            mb.add_triangle(v[0], v[1], v[2],
                            v[3], v[4], v[5],
                            v[6], v[7], v[8]);
        }
    } else {
        /* ASCII parser — scan for "vertex" keyword + 3 doubles. */
        const char *cur = (const char *)buf.data();
        const char *end = cur + buf.size();
        while (cur < end) {
            /* Locate next "vertex" token. */
            const char *p = cur;
            /* Skip until 'v' or 'V'. */
            while (p < end && *p != 'v' && *p != 'V') ++p;
            if (p + 6 > end) break;
            if ((p[1] == 'e' || p[1] == 'E') &&
                (p[2] == 'r' || p[2] == 'R') &&
                (p[3] == 't' || p[3] == 'T') &&
                (p[4] == 'e' || p[4] == 'E') &&
                (p[5] == 'x' || p[5] == 'X')) {
                /* Read 3 doubles for the first vertex. */
                p += 6;
                double v[9] = {0};
                if (!parse_double(p, end, v[0])) break;
                if (!parse_double(p, end, v[1])) break;
                if (!parse_double(p, end, v[2])) break;
                /* Skip to the next "vertex". */
                while (p < end) {
                    while (p < end && *p != 'v' && *p != 'V') ++p;
                    if (p + 6 > end) break;
                    if ((p[1] == 'e' || p[1] == 'E') &&
                        (p[2] == 'r' || p[2] == 'R') &&
                        (p[3] == 't' || p[3] == 'T') &&
                        (p[4] == 'e' || p[4] == 'E') &&
                        (p[5] == 'x' || p[5] == 'X')) break;
                    ++p;
                }
                if (p + 6 > end) break;
                p += 6;
                if (!parse_double(p, end, v[3])) break;
                if (!parse_double(p, end, v[4])) break;
                if (!parse_double(p, end, v[5])) break;
                while (p < end) {
                    while (p < end && *p != 'v' && *p != 'V') ++p;
                    if (p + 6 > end) break;
                    if ((p[1] == 'e' || p[1] == 'E') &&
                        (p[2] == 'r' || p[2] == 'R') &&
                        (p[3] == 't' || p[3] == 'T') &&
                        (p[4] == 'e' || p[4] == 'E') &&
                        (p[5] == 'x' || p[5] == 'X')) break;
                    ++p;
                }
                if (p + 6 > end) break;
                p += 6;
                if (!parse_double(p, end, v[6])) break;
                if (!parse_double(p, end, v[7])) break;
                if (!parse_double(p, end, v[8])) break;
                mb.add_triangle(v[0], v[1], v[2],
                                v[3], v[4], v[5],
                                v[6], v[7], v[8]);
                cur = p;
            } else {
                cur = p + 1;
            }
        }
    }
    return finalize_mesh(mb, "stl");
}

/* matlab_string ABI: wrapper that unpacks the descriptor and calls
 * the (path, plen) inner.  Exposed to MATLAB as `pde_load_stl(path)`. */
matlab_struct *matlab_pde_load_stl(void *s) {
    if (!s) return nullptr;
    auto *ms = (struct matlab_string_local_s *)s;
    return matlab_pde_load_stl_path(ms->data, ms->len);
}

/* Forward declaration — definition appears below. */
double matlab_pde_save_stl_binary(matlab_struct *mesh,
                                  const char *path, int64_t plen);

/* matlab_pde_save_stl(mesh, path) — MATLAB-callable wrapper around
 * matlab_pde_save_stl_binary using the matlab_string ABI for the
 * path.  Returns 1 on success, 0 on failure. */
double matlab_pde_save_stl(matlab_struct *mesh, void *s) {
    if (!s) return 0.0;
    auto *ms = (struct matlab_string_local_s *)s;
    return matlab_pde_save_stl_binary(mesh, ms->data, ms->len);
}

/* matlab_pde_save_stl_binary — companion writer for testing /
 * round-tripping.  Writes a minimal binary STL (zero normal vectors —
 * most viewers recompute from vertex order anyway).  Returns 1 on
 * success, 0 on failure. */
double matlab_pde_save_stl_binary(matlab_struct *mesh,
                                  const char *path, int64_t plen) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *faces = matlab_struct_get_mat(mesh, "Faces", 5);
    if (!nodes || !faces) return 0.0;
    std::string fn(path, (size_t)plen);
    std::ofstream f(fn, std::ios::binary | std::ios::trunc);
    if (!f) return 0.0;
    uint8_t header[80] = {0};
    memcpy(header, "matlab_llvm STL binary", 22);
    f.write((const char *)header, 80);
    uint32_t n = (uint32_t)faces->rows;
    f.write((const char *)&n, 4);
    for (int64_t i = 0; i < (int64_t)n; ++i) {
        int64_t a = (int64_t)faces->data[i * 4 + 1] - 1;
        int64_t b = (int64_t)faces->data[i * 4 + 2] - 1;
        int64_t c = (int64_t)faces->data[i * 4 + 3] - 1;
        float pf[12] = {0};  /* normal (3) + 3 vertices (9) */
        pf[3]  = (float)nodes->data[a * 3 + 0];
        pf[4]  = (float)nodes->data[a * 3 + 1];
        pf[5]  = (float)nodes->data[a * 3 + 2];
        pf[6]  = (float)nodes->data[b * 3 + 0];
        pf[7]  = (float)nodes->data[b * 3 + 1];
        pf[8]  = (float)nodes->data[b * 3 + 2];
        pf[9]  = (float)nodes->data[c * 3 + 0];
        pf[10] = (float)nodes->data[c * 3 + 1];
        pf[11] = (float)nodes->data[c * 3 + 2];
        f.write((const char *)pf, 48);
        uint16_t attr = 0;
        f.write((const char *)&attr, 2);
    }
    return f.good() ? 1.0 : 0.0;
}

double matlab_pde_num_nodes(matlab_struct *mesh) {
    return matlab_struct_get_f64(mesh, "NumNodes", 8);
}
double matlab_pde_num_faces(matlab_struct *mesh) {
    return matlab_struct_get_f64(mesh, "NumFaces", 8);
}

/* --- Sparse FEM assembly path ---------------------------------------
 *
 * These mirror the dense pde_assemble_* family but emit (I, J, V)
 * triplets instead of writing directly into a dense N×N array.  The
 * triplets are handed to matlab_sparse_from_triplets which sums
 * duplicates and compacts into CSR.
 *
 * Why this matters: dense K caps us at ~3 000 DOF.  Sparse K is
 * essentially N + (per-row fan-out) entries, which scales to
 * 100 000 DOF on commodity RAM.  Combined with PCG (matlab_sparse_pcg)
 * the iterative solve runs in O(N * #iter * avg fan-out) which is
 * typically 50–200× faster than dense LU at scale.
 *
 * The function-form API is structured as:
 *   1. matlab_pde_assemble_poisson_2d_sparse(mesh, c, a, f) -> struct
 *      with .K (sparse, m×m) and .F (dense, m×1).
 *   2. matlab_pde_assemble_elast_3d_sparse(mesh, E, nu) -> sparse K.
 *   3. matlab_pde_apply_dirichlet_sparse / matlab_pde_apply_fixed_3d_sparse
 *      zero rows/cols by rewriting (I, J, V) in place and re-summing.
 *
 * The MATLAB-side call sequence:
 *   sys  = pde_assemble_poisson_2d_sparse(mesh, 1, 0, 1);
 *   sys2 = pde_apply_dirichlet_sparse(sys, bnd, 0);
 *   res  = pcg(pde_sys_K_sparse(sys2), pde_sys_F(sys2), 1e-8, 1000);
 *   u    = pcg_x(res);
 */

void *matlab_sparse_from_triplets(matlab_mat *I, matlab_mat *J, matlab_mat *V,
                                  double m_d, double n_d);
double matlab_sparse_nnz(void *S);

/* 2-D Poisson sparse assembly. */
matlab_struct *matlab_pde_assemble_poisson_2d_sparse(matlab_struct *mesh,
                                                    double c, double a, double f) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes",     5);
    matlab_mat *tris  = matlab_struct_get_mat(mesh, "Triangles", 9);
    int64_t Nn = nodes->rows;
    int64_t Nt = tris->rows;

    /* Each triangle contributes 9 K entries + 3 F entries. */
    int64_t cap = Nt * 9;
    matlab_mat *I = mat_alloc(cap, 1);
    matlab_mat *J = mat_alloc(cap, 1);
    matlab_mat *V = mat_alloc(cap, 1);
    matlab_mat *F = mat_alloc(Nn, 1);
    int64_t pos = 0;

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t i0 = (int64_t)tris->data[e * 3 + 0] - 1;
        int64_t i1 = (int64_t)tris->data[e * 3 + 1] - 1;
        int64_t i2 = (int64_t)tris->data[e * 3 + 2] - 1;
        double x0 = nodes->data[i0 * 2 + 0], y0 = nodes->data[i0 * 2 + 1];
        double x1 = nodes->data[i1 * 2 + 0], y1 = nodes->data[i1 * 2 + 1];
        double x2 = nodes->data[i2 * 2 + 0], y2 = nodes->data[i2 * 2 + 1];
        double twoA = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        double area = 0.5 * twoA;
        if (area <= 0) area = -area;
        double b[3] = { (y1 - y2), (y2 - y0), (y0 - y1) };
        double cc[3] = { (x2 - x1), (x0 - x2), (x1 - x0) };
        double inv2A = 1.0 / twoA;
        for (int p = 0; p < 3; ++p) { b[p] *= inv2A; cc[p] *= inv2A; }
        int64_t loc[3] = { i0, i1, i2 };
        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) {
                double Ke = c * area * (b[p] * b[q] + cc[p] * cc[q]);
                if (p == q) Ke += a * area / 3.0;
                I->data[pos] = (double)(loc[p] + 1);  /* 1-based */
                J->data[pos] = (double)(loc[q] + 1);
                V->data[pos] = Ke;
                pos++;
            }
            F->data[loc[p]] += f * area / 3.0;
        }
    }
    /* Trim. */
    I->rows = pos; J->rows = pos; V->rows = pos;

    void *K = matlab_sparse_from_triplets(I, J, V, (double)Nn, (double)Nn);
    matlab_struct *out = matlab_struct_new();
    /* Store the sparse matrix pointer via the standard struct-set-mat
     * slot (the descriptor is sniffed via its 0xC0FFEE05 magic, so the
     * polymorphic disp / matvec callsites still work). */
    matlab_struct_set_mat(out, "K", 1, (matlab_mat *)K);
    matlab_struct_set_mat(out, "F", 1, F);
    return out;
}

/* matlab_pde_sys_K_sparse — accessor for the sparse-K field, returning
 * a void* so the caller can pass it directly to PCG / sparse_matvec
 * without going through the matlab_mat coercion. */
void *matlab_pde_sys_K_sparse(matlab_struct *sys) {
    return (void *)matlab_struct_get_mat(sys, "K", 1);
}

/* Dirichlet u = u_val on a set of node ids, sparse form.  Rebuilds
 * the sparse matrix by walking its CSR storage: for each constrained
 * row r, clear all entries; for each constrained column c, clear all
 * entries.  Insert a diagonal-1 for each constrained row.  F is
 * adjusted by subtracting K(:, fixed) * u_val (trivial for u_val=0).
 *
 * Uses a re-triplet pass to keep the code small.  At FEM scale this
 * is O(nnz) which is small compared to the solve.
 */
matlab_struct *matlab_pde_apply_dirichlet_sparse(matlab_struct *sys,
                                                  matlab_mat *node_ids,
                                                  double u_val) {
    /* Lift the sparse_mat back into triplets, filter, and rebuild. */
    extern matlab_mat *matlab_sparse_full(void *Sv);
    extern double matlab_sparse_rows(void *Sv);
    extern double matlab_sparse_cols(void *Sv);
    /* Pull the K + F. */
    void *K_raw  = (void *)matlab_struct_get_mat(sys, "K", 1);
    matlab_mat *F = matlab_struct_get_mat(sys, "F", 1);
    /* Treat the sparse_mat as opaque and read out its triplets. */
    struct sparse_view {
        uint32_t magic, _pad;
        int64_t *row_ptr;
        int64_t *col_idx;
        double  *vals;
        int64_t rows, cols, nnz;
    };
    sparse_view *S = (sparse_view *)K_raw;
    if (!S || S->magic != 0xC0FFEE05u) return nullptr;
    int64_t Nn = S->rows;
    int64_t Nd = node_ids->rows * node_ids->cols;
    std::vector<int8_t> fixed((size_t)Nn, 0);
    for (int64_t k = 0; k < Nd; ++k) {
        int64_t n = (int64_t)node_ids->data[k] - 1;
        if (n >= 0 && n < Nn) fixed[(size_t)n] = 1;
    }
    /* RHS adjustment for non-zero u_val. */
    matlab_mat *F2 = mat_alloc(Nn, 1);
    memcpy(F2->data, F->data, sizeof(double) * (size_t)Nn);
    if (u_val != 0.0) {
        for (int64_t r = 0; r < Nn; ++r) {
            int64_t lo = S->row_ptr[r];
            int64_t hi = S->row_ptr[r + 1];
            for (int64_t k = lo; k < hi; ++k) {
                int64_t c = S->col_idx[k];
                if (fixed[(size_t)c]) F2->data[r] -= S->vals[k] * u_val;
            }
        }
    }
    /* Rebuild triplets, filtering. */
    int64_t nnz_old = S->nnz;
    matlab_mat *I = mat_alloc(nnz_old + Nd, 1);
    matlab_mat *J = mat_alloc(nnz_old + Nd, 1);
    matlab_mat *V = mat_alloc(nnz_old + Nd, 1);
    int64_t pos = 0;
    for (int64_t r = 0; r < Nn; ++r) {
        int64_t lo = S->row_ptr[r];
        int64_t hi = S->row_ptr[r + 1];
        if (fixed[(size_t)r]) continue;  /* skip row */
        for (int64_t k = lo; k < hi; ++k) {
            int64_t c = S->col_idx[k];
            if (fixed[(size_t)c]) continue;  /* skip col */
            I->data[pos] = (double)(r + 1);
            J->data[pos] = (double)(c + 1);
            V->data[pos] = S->vals[k];
            pos++;
        }
    }
    /* Diagonal-1 + Dirichlet RHS for each fixed dof. */
    for (int64_t r = 0; r < Nn; ++r) {
        if (!fixed[(size_t)r]) continue;
        I->data[pos] = (double)(r + 1);
        J->data[pos] = (double)(r + 1);
        V->data[pos] = 1.0;
        pos++;
        F2->data[r] = u_val;
    }
    I->rows = pos; J->rows = pos; V->rows = pos;
    void *K2 = matlab_sparse_from_triplets(I, J, V, (double)Nn, (double)Nn);

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, (matlab_mat *)K2);
    matlab_struct_set_mat(out, "F", 1, F2);
    return out;
}

/* --- Sparse 3-D linear elasticity assembly ----------------------- */

/* Wire helpers — re-declarations of the static internals that the
 * dense path uses.  They're declared at file scope above; this is
 * just a local visibility marker. */
static void elast_compute_grad_extern(const double X[4][3], double dN[4][3],
                                      double *vol_out) {
    extern void elast_compute_grad(const double X[4][3], double dN[4][3],
                                   double *vol_out);
    elast_compute_grad(X, dN, vol_out);
}

void *matlab_pde_assemble_elast_3d_sparse(matlab_struct *mesh,
                                          double E, double nu) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *tets  = matlab_struct_get_mat(mesh, "Tets",  4);
    int64_t Nn = nodes->rows;
    int64_t Nt = tets->rows;
    int64_t Ndof = 3 * Nn;

    /* Each tet contributes 12*12 = 144 K entries.  At ~10k tets that's
     * ~1.4M triplets — fits in RAM comfortably. */
    int64_t cap = Nt * 144;
    matlab_mat *I = mat_alloc(cap, 1);
    matlab_mat *J = mat_alloc(cap, 1);
    matlab_mat *V = mat_alloc(cap, 1);
    int64_t pos = 0;

    /* Constitutive matrix — same as dense path. */
    double lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu  = E / (2.0 * (1.0 + nu));
    double D[6][6] = {0};
    D[0][0] = lam + 2.0 * mu;  D[0][1] = lam;            D[0][2] = lam;
    D[1][0] = lam;             D[1][1] = lam + 2.0 * mu; D[1][2] = lam;
    D[2][0] = lam;             D[2][1] = lam;            D[2][2] = lam + 2.0 * mu;
    D[3][3] = mu; D[4][4] = mu; D[5][5] = mu;

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t ids[4];
        double X[4][3];
        for (int i = 0; i < 4; ++i) {
            ids[i] = (int64_t)tets->data[e * 4 + i] - 1;
            X[i][0] = nodes->data[ids[i] * 3 + 0];
            X[i][1] = nodes->data[ids[i] * 3 + 1];
            X[i][2] = nodes->data[ids[i] * 3 + 2];
        }
        double dN[4][3];
        double Vol;
        elast_compute_grad_extern(X, dN, &Vol);
        /* B (6x12). */
        double B[6][12] = {0};
        for (int i = 0; i < 4; ++i) {
            double bx = dN[i][0], by = dN[i][1], bz = dN[i][2];
            int c = i * 3;
            B[0][c + 0] = bx;
            B[1][c + 1] = by;
            B[2][c + 2] = bz;
            B[3][c + 0] = by; B[3][c + 1] = bx;
            B[4][c + 1] = bz; B[4][c + 2] = by;
            B[5][c + 0] = bz; B[5][c + 2] = bx;
        }
        /* Ke = Vol * B^T * D * B. */
        double DB[6][12];
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 12; ++c) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += D[r][k] * B[k][c];
                DB[r][c] = s;
            }
        double Ke[12][12];
        for (int r = 0; r < 12; ++r)
            for (int c = 0; c < 12; ++c) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += B[k][r] * DB[k][c];
                Ke[r][c] = Vol * s;
            }
        /* Scatter into triplets (1-based global indices). */
        for (int p = 0; p < 4; ++p) {
            int64_t gp = ids[p] * 3;
            for (int q = 0; q < 4; ++q) {
                int64_t gq = ids[q] * 3;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        I->data[pos] = (double)(gp + a + 1);
                        J->data[pos] = (double)(gq + b + 1);
                        V->data[pos] = Ke[p * 3 + a][q * 3 + b];
                        pos++;
                    }
                }
            }
        }
    }
    I->rows = pos; J->rows = pos; V->rows = pos;
    return matlab_sparse_from_triplets(I, J, V, (double)Ndof, (double)Ndof);
}

}  /* extern "C" close before the template */

/* --- Geometry primitives (voxelize-AABB family) -------------------- *
 *
 * matlab_pde_multicylinder / matlab_pde_multisphere reuse the same
 * AABB-voxelize-then-6-tet-decompose pipeline as the STL/GLB volumetric
 * mesher (matlab_pde_voxelize_surface).  The only difference is the
 * inside-test: each primitive provides its own predicate.
 *
 * The Kuhn 6-tet decomposition + boundary-face recovery + face_id
 * assignment by dominant outward axis (1=-z, 2=+z, 3=-y, 4=+y, 5=-x,
 * 6=+x) match matlab_pde_mesh_cuboid_tet exactly so downstream
 * pde_face_nodes / pde_face_pressure_3d / pde_apply_fixed_3d_sparse
 * code paths plug in unchanged.
 */

namespace {

struct CylinderShape {
    double R;       /* outer radius */
    double R_in;    /* inner radius (0 = solid; > 0 = hollow) */
    double H;       /* axial extent */
    /* Axis-aligned along z.  Centroid at origin in XY; z spans [0, H]. */
    bool inside(double x, double y, double z) const {
        if (z < 0 || z > H) return false;
        double r2 = x * x + y * y;
        if (r2 > R * R) return false;
        if (R_in > 0 && r2 < R_in * R_in) return false;
        return true;
    }
};

struct SphereShape {
    double R;
    /* Centred at origin. */
    bool inside(double x, double y, double z) const {
        return x * x + y * y + z * z <= R * R;
    }
};

/* Common voxelize-decompose-collect routine, templated on the shape
 * predicate.  Returns a struct with Nodes / Tets / Faces / Nx / Ny /
 * Nz / W / D / H matching matlab_pde_mesh_cuboid_tet's output. */
template <typename Shape>
matlab_struct *voxelize_primitive(const Shape &shape,
                                  double xmin, double xmax,
                                  double ymin, double ymax,
                                  double zmin, double zmax,
                                  double voxel_size) {
    if (voxel_size <= 0) return nullptr;
    int64_t Nx = (int64_t)ceil((xmax - xmin) / voxel_size); if (Nx < 1) Nx = 1;
    int64_t Ny = (int64_t)ceil((ymax - ymin) / voxel_size); if (Ny < 1) Ny = 1;
    int64_t Nz = (int64_t)ceil((zmax - zmin) / voxel_size); if (Nz < 1) Nz = 1;
    double dx = (xmax - xmin) / (double)Nx;
    double dy = (ymax - ymin) / (double)Ny;
    double dz = (zmax - zmin) / (double)Nz;

    int64_t total_cells = Nx * Ny * Nz;
    std::vector<int8_t> inside((size_t)total_cells, 0);
    int64_t inside_count = 0;
    for (int64_t k = 0; k < Nz; ++k) {
        double cz = zmin + (k + 0.5) * dz;
        for (int64_t j = 0; j < Ny; ++j) {
            double cy = ymin + (j + 0.5) * dy;
            for (int64_t i = 0; i < Nx; ++i) {
                double cx = xmin + (i + 0.5) * dx;
                if (shape.inside(cx, cy, cz)) {
                    inside[(size_t)(k * Ny * Nx + j * Nx + i)] = 1;
                    inside_count++;
                }
            }
        }
    }
    if (inside_count == 0) return nullptr;

    int64_t Px = Nx + 1, Py = Ny + 1, Pz = Nz + 1;
    int64_t Pn = Px * Py * Pz;
    std::vector<int64_t> node_id((size_t)Pn, -1);
    std::vector<double> vol_nodes;
    vol_nodes.reserve((size_t)inside_count * 24);

    auto ensure_node = [&](int64_t i, int64_t j, int64_t k) -> int64_t {
        int64_t key = (k * Py + j) * Px + i;
        if (node_id[(size_t)key] >= 0) return node_id[(size_t)key];
        int64_t nid = (int64_t)(vol_nodes.size() / 3);
        vol_nodes.push_back(xmin + (double)i * dx);
        vol_nodes.push_back(ymin + (double)j * dy);
        vol_nodes.push_back(zmin + (double)k * dz);
        node_id[(size_t)key] = nid;
        return nid;
    };

    static const int Tdef[6][4] = {
        {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
        {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6},
    };

    std::vector<int64_t> tets_flat;
    tets_flat.reserve((size_t)inside_count * 24);

    auto inside_idx = [&](int64_t i, int64_t j, int64_t k) -> int8_t {
        if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) return 0;
        return inside[(size_t)(k * Ny * Nx + j * Nx + i)];
    };

    std::vector<int64_t> face_id, face_n1, face_n2, face_n3;
    auto add_tri = [&](int64_t fid, int64_t a, int64_t b, int64_t c) {
        face_id.push_back(fid);
        face_n1.push_back(a + 1);
        face_n2.push_back(b + 1);
        face_n3.push_back(c + 1);
    };

    for (int64_t k = 0; k < Nz; ++k) {
        for (int64_t j = 0; j < Ny; ++j) {
            for (int64_t i = 0; i < Nx; ++i) {
                if (!inside_idx(i, j, k)) continue;
                int64_t corners[8] = {
                    ensure_node(i,     j,     k    ),
                    ensure_node(i + 1, j,     k    ),
                    ensure_node(i + 1, j + 1, k    ),
                    ensure_node(i,     j + 1, k    ),
                    ensure_node(i,     j,     k + 1),
                    ensure_node(i + 1, j,     k + 1),
                    ensure_node(i + 1, j + 1, k + 1),
                    ensure_node(i,     j + 1, k + 1),
                };
                for (int t = 0; t < 6; ++t) {
                    tets_flat.push_back(corners[Tdef[t][0]] + 1);
                    tets_flat.push_back(corners[Tdef[t][1]] + 1);
                    tets_flat.push_back(corners[Tdef[t][2]] + 1);
                    tets_flat.push_back(corners[Tdef[t][3]] + 1);
                }
                if (!inside_idx(i, j, k - 1)) {
                    add_tri(1, corners[0], corners[2], corners[1]);
                    add_tri(1, corners[0], corners[3], corners[2]);
                }
                if (!inside_idx(i, j, k + 1)) {
                    add_tri(2, corners[4], corners[5], corners[6]);
                    add_tri(2, corners[4], corners[6], corners[7]);
                }
                if (!inside_idx(i, j - 1, k)) {
                    add_tri(3, corners[0], corners[1], corners[5]);
                    add_tri(3, corners[0], corners[5], corners[4]);
                }
                if (!inside_idx(i, j + 1, k)) {
                    add_tri(4, corners[3], corners[7], corners[6]);
                    add_tri(4, corners[3], corners[6], corners[2]);
                }
                if (!inside_idx(i - 1, j, k)) {
                    add_tri(5, corners[0], corners[4], corners[7]);
                    add_tri(5, corners[0], corners[7], corners[3]);
                }
                if (!inside_idx(i + 1, j, k)) {
                    add_tri(6, corners[1], corners[2], corners[6]);
                    add_tri(6, corners[1], corners[6], corners[5]);
                }
            }
        }
    }

    int64_t Nn = (int64_t)(vol_nodes.size() / 3);
    int64_t Nt = (int64_t)(tets_flat.size()  / 4);
    int64_t Nbnd = (int64_t)face_id.size();

    matlab_mat *Nodes = mat_alloc(Nn, 3);
    memcpy(Nodes->data, vol_nodes.data(), sizeof(double) * (size_t)(Nn * 3));
    matlab_mat *Tets = mat_alloc(Nt, 4);
    for (int64_t i = 0; i < Nt * 4; ++i) Tets->data[i] = (double)tets_flat[(size_t)i];
    matlab_mat *Faces = mat_alloc(Nbnd, 4);
    for (int64_t k = 0; k < Nbnd; ++k) {
        Faces->data[k * 4 + 0] = (double)face_id[(size_t)k];
        Faces->data[k * 4 + 1] = (double)face_n1[(size_t)k];
        Faces->data[k * 4 + 2] = (double)face_n2[(size_t)k];
        Faces->data[k * 4 + 3] = (double)face_n3[(size_t)k];
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Nodes", 5, Nodes);
    matlab_struct_set_mat(out, "Tets",  4, Tets);
    matlab_struct_set_mat(out, "Faces", 5, Faces);
    matlab_struct_set_f64(out, "Nx", 2, (double)Nx);
    matlab_struct_set_f64(out, "Ny", 2, (double)Ny);
    matlab_struct_set_f64(out, "Nz", 2, (double)Nz);
    matlab_struct_set_f64(out, "W",  1, xmax - xmin);
    matlab_struct_set_f64(out, "D",  1, ymax - ymin);
    matlab_struct_set_f64(out, "H",  1, zmax - zmin);
    matlab_struct_set_f64(out, "NumInsideCells", 14, (double)inside_count);
    return out;
}

}  /* anonymous namespace */

extern "C" {

/* multicylinder(R, H, voxel_size) — solid cylinder centred at origin
 * in XY with axis along z from 0 to H. */
matlab_struct *matlab_pde_multicylinder(double R, double H, double voxel_size) {
    CylinderShape S{R, 0.0, H};
    return voxelize_primitive(S, -R, R, -R, R, 0, H, voxel_size);
}

/* multicylinder_hollow(R_out, R_in, H, voxel_size) — annular cylinder. */
matlab_struct *matlab_pde_multicylinder_hollow(double R_out, double R_in,
                                                double H, double voxel_size) {
    CylinderShape S{R_out, R_in, H};
    return voxelize_primitive(S, -R_out, R_out, -R_out, R_out, 0, H, voxel_size);
}

/* multisphere(R, voxel_size) — solid sphere centred at origin. */
matlab_struct *matlab_pde_multisphere(double R, double voxel_size) {
    SphereShape S{R};
    return voxelize_primitive(S, -R, R, -R, R, -R, R, voxel_size);
}

/* --- Affine ops on fegeometry ---------------------------------- *
 *
 * Operate on the Nodes Nn×3 array.  Each op returns the SAME mesh
 * struct (mutated in place) so user code can chain `mesh =
 * pde_translate(mesh, ...)`.  The Tets / Faces tables don't change.
 */

matlab_struct *matlab_pde_translate(matlab_struct *mesh,
                                    double dx, double dy, double dz) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    if (!nodes) return mesh;
    int64_t Nn = nodes->rows;
    for (int64_t i = 0; i < Nn; ++i) {
        nodes->data[i * 3 + 0] += dx;
        nodes->data[i * 3 + 1] += dy;
        nodes->data[i * 3 + 2] += dz;
    }
    return mesh;
}

matlab_struct *matlab_pde_scale(matlab_struct *mesh,
                                double sx, double sy, double sz) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    if (!nodes) return mesh;
    int64_t Nn = nodes->rows;
    for (int64_t i = 0; i < Nn; ++i) {
        nodes->data[i * 3 + 0] *= sx;
        nodes->data[i * 3 + 1] *= sy;
        nodes->data[i * 3 + 2] *= sz;
    }
    return mesh;
}

/* matlab_pde_rotate(mesh, axis, angle_deg) — rotate every node by
 * `angle_deg` degrees around an axis.  axis: 1=x, 2=y, 3=z.
 *
 * 3-D rotation matrices:
 *   R_x = [1 0 0; 0 c -s; 0 s c]
 *   R_y = [c 0 s; 0 1 0; -s 0 c]
 *   R_z = [c -s 0; s c 0; 0 0 1]
 */
matlab_struct *matlab_pde_rotate(matlab_struct *mesh,
                                  double axis_d, double angle_deg) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    if (!nodes) return mesh;
    int64_t Nn = nodes->rows;
    int axis = (int)axis_d;
    double th = angle_deg * M_PI / 180.0;
    double c = cos(th);
    double s = sin(th);
    for (int64_t i = 0; i < Nn; ++i) {
        double x = nodes->data[i * 3 + 0];
        double y = nodes->data[i * 3 + 1];
        double z = nodes->data[i * 3 + 2];
        if (axis == 1) {  /* x-axis */
            nodes->data[i * 3 + 1] = c * y - s * z;
            nodes->data[i * 3 + 2] = s * y + c * z;
        } else if (axis == 2) {  /* y-axis */
            nodes->data[i * 3 + 0] = c * x + s * z;
            nodes->data[i * 3 + 2] = -s * x + c * z;
        } else {  /* z-axis (default) */
            nodes->data[i * 3 + 0] = c * x - s * y;
            nodes->data[i * 3 + 1] = s * x + c * y;
        }
    }
    return mesh;
}

}  /* extern "C" */

extern "C" {

/* --- Voxelize-AABB volumetric tet mesher --------------------------- *
 *
 * matlab_pde_voxelize_surface(surface, voxel_size) — take a surface
 * fegeometry (Nodes, Faces from STL/GLB importers) and build a
 * volumetric tet mesh by:
 *   1. Computing the AABB of the surface.
 *   2. Subdividing the AABB into a Nx × Ny × Nz grid of hex cells
 *      with edge length ~voxel_size.
 *   3. For each cell, ray-cast the centroid in +x and count
 *      intersections with the surface triangles (Möller-Trumbore).
 *      Odd count → inside.
 *   4. Keep inside cells.  Add the corner vertices of each kept cell
 *      to the volumetric node table (deduplicated via spatial hash).
 *   5. Split each kept hex into 6 tets via the same Kuhn 0-6
 *      diagonal decomposition used by multicuboid.
 *   6. Recover boundary triangles where an inside cell has an
 *      outside neighbour.  face_id assigned by dominant outward
 *      normal direction: 1=-z, 2=+z, 3=-y, 4=+y, 5=-x, 6=+x.
 *
 * Returns the same struct shape as pde_mesh_cuboid_tet so the FEM
 * assembler / pdeplot3D / etc. plug in unchanged.
 *
 * Trade-off: voxelization gives a step-stair boundary, not the
 * original surface triangulation.  Mesh quality is uniform (all tets
 * are identical shape) but the boundary fidelity scales with the
 * voxel_size.  Adequate for visualisation + qualitative stress
 * patterns; for high-fidelity boundary stress, a proper
 * constrained-Delaunay tetrahedralization is roadmap §10.3 follow-up.
 */

static inline bool ray_triangle_intersect(
    double ox, double oy, double oz,
    double dx, double dy, double dz,
    double v0x, double v0y, double v0z,
    double v1x, double v1y, double v1z,
    double v2x, double v2y, double v2z) {
    /* Möller-Trumbore.  Returns true if the ray (o + t*d, t>0) hits
     * the triangle.  Backface culling disabled (we want both sides). */
    double e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
    double e2x = v2x - v0x, e2y = v2y - v0y, e2z = v2z - v0z;
    double hx = dy * e2z - dz * e2y;
    double hy = dz * e2x - dx * e2z;
    double hz = dx * e2y - dy * e2x;
    double a = e1x * hx + e1y * hy + e1z * hz;
    if (a > -1e-12 && a < 1e-12) return false;  /* parallel */
    double f = 1.0 / a;
    double sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
    double u = f * (sx * hx + sy * hy + sz * hz);
    if (u < 0.0 || u > 1.0) return false;
    double qx = sy * e1z - sz * e1y;
    double qy = sz * e1x - sx * e1z;
    double qz = sx * e1y - sy * e1x;
    double v = f * (dx * qx + dy * qy + dz * qz);
    if (v < 0.0 || u + v > 1.0) return false;
    double t = f * (e2x * qx + e2y * qy + e2z * qz);
    return t > 1e-9;
}

matlab_struct *matlab_pde_voxelize_surface(matlab_struct *surface,
                                           double voxel_size) {
    matlab_mat *nodes = matlab_struct_get_mat(surface, "Nodes", 5);
    matlab_mat *faces = matlab_struct_get_mat(surface, "Faces", 5);
    if (!nodes || !faces) return nullptr;
    int64_t Nn = nodes->rows;
    int64_t Nf = faces->rows;
    if (Nn < 3 || Nf < 1 || voxel_size <= 0) return nullptr;

    /* AABB. */
    double xmin =  1e300, ymin =  1e300, zmin =  1e300;
    double xmax = -1e300, ymax = -1e300, zmax = -1e300;
    for (int64_t i = 0; i < Nn; ++i) {
        double x = nodes->data[i * 3 + 0];
        double y = nodes->data[i * 3 + 1];
        double z = nodes->data[i * 3 + 2];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
    }
    /* Inflate the AABB by a small padding so boundary cells stay
     * fully inside the grid even with floating-point rounding. */
    double pad = voxel_size * 0.05;
    xmin -= pad; ymin -= pad; zmin -= pad;
    xmax += pad; ymax += pad; zmax += pad;
    int64_t Nx = (int64_t)ceil((xmax - xmin) / voxel_size); if (Nx < 1) Nx = 1;
    int64_t Ny = (int64_t)ceil((ymax - ymin) / voxel_size); if (Ny < 1) Ny = 1;
    int64_t Nz = (int64_t)ceil((zmax - zmin) / voxel_size); if (Nz < 1) Nz = 1;
    double dx = (xmax - xmin) / (double)Nx;
    double dy = (ymax - ymin) / (double)Ny;
    double dz = (zmax - zmin) / (double)Nz;

    /* For each cell, test centroid inside-ness via ray-cast in +x. */
    int64_t total_cells = Nx * Ny * Nz;
    std::vector<int8_t> inside((size_t)total_cells, 0);

    /* Precompute triangle coords for fast access. */
    std::vector<double> tv((size_t)Nf * 9);
    for (int64_t k = 0; k < Nf; ++k) {
        int64_t i0 = (int64_t)faces->data[k * 4 + 1] - 1;
        int64_t i1 = (int64_t)faces->data[k * 4 + 2] - 1;
        int64_t i2 = (int64_t)faces->data[k * 4 + 3] - 1;
        for (int c = 0; c < 3; ++c) {
            tv[(size_t)k * 9 + 0 + c] = nodes->data[i0 * 3 + c];
            tv[(size_t)k * 9 + 3 + c] = nodes->data[i1 * 3 + c];
            tv[(size_t)k * 9 + 6 + c] = nodes->data[i2 * 3 + c];
        }
    }

    /* Direction = (1, 0, 0) — +x ray.  Use a slightly tilted ray so
     * we don't graze edges/vertices of axis-aligned triangles. */
    const double drx = 1.0, dry = 0.001, drz = 0.0007;

    int64_t inside_count = 0;
    for (int64_t k = 0; k < Nz; ++k) {
        double cz = zmin + (k + 0.5) * dz;
        for (int64_t j = 0; j < Ny; ++j) {
            double cy = ymin + (j + 0.5) * dy;
            for (int64_t i = 0; i < Nx; ++i) {
                double cx = xmin + (i + 0.5) * dx;
                int hits = 0;
                for (int64_t t = 0; t < Nf; ++t) {
                    const double *p = tv.data() + (size_t)t * 9;
                    if (ray_triangle_intersect(cx, cy, cz, drx, dry, drz,
                                                p[0], p[1], p[2],
                                                p[3], p[4], p[5],
                                                p[6], p[7], p[8])) {
                        hits++;
                    }
                }
                if (hits & 1) {
                    inside[(size_t)(k * Ny * Nx + j * Nx + i)] = 1;
                    inside_count++;
                }
            }
        }
    }

    if (inside_count == 0) return nullptr;

    /* Now build the tet mesh.  We allocate node ids for every corner
     * of every kept cell, deduplicating by grid index. */
    int64_t Px = Nx + 1, Py = Ny + 1, Pz = Nz + 1;
    int64_t Pn = Px * Py * Pz;
    std::vector<int64_t> node_id((size_t)Pn, -1);  /* -1 = not yet assigned */
    std::vector<double> vol_nodes;
    vol_nodes.reserve((size_t)inside_count * 24);

    auto ensure_node = [&](int64_t i, int64_t j, int64_t k) -> int64_t {
        int64_t key = (k * Py + j) * Px + i;
        if (node_id[(size_t)key] >= 0) return node_id[(size_t)key];
        int64_t nid = (int64_t)(vol_nodes.size() / 3);
        vol_nodes.push_back(xmin + (double)i * dx);
        vol_nodes.push_back(ymin + (double)j * dy);
        vol_nodes.push_back(zmin + (double)k * dz);
        node_id[(size_t)key] = nid;
        return nid;
    };

    /* Kuhn 6-tet decomposition (same as multicuboid). */
    static const int Tdef[6][4] = {
        {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
        {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6},
    };

    std::vector<int64_t> tets_flat;  /* (Nt, 4) row-major, 1-based */
    tets_flat.reserve((size_t)inside_count * 24);

    auto inside_idx = [&](int64_t i, int64_t j, int64_t k) -> int8_t {
        if (i < 0 || i >= Nx || j < 0 || j >= Ny || k < 0 || k >= Nz) return 0;
        return inside[(size_t)(k * Ny * Nx + j * Nx + i)];
    };

    /* Boundary face triangles. */
    std::vector<int64_t> face_id, face_n1, face_n2, face_n3;

    auto add_tri = [&](int64_t fid, int64_t a, int64_t b, int64_t c) {
        face_id.push_back(fid);
        face_n1.push_back(a + 1);
        face_n2.push_back(b + 1);
        face_n3.push_back(c + 1);
    };

    for (int64_t k = 0; k < Nz; ++k) {
        for (int64_t j = 0; j < Ny; ++j) {
            for (int64_t i = 0; i < Nx; ++i) {
                if (!inside_idx(i, j, k)) continue;
                int64_t corners[8] = {
                    ensure_node(i,     j,     k    ),
                    ensure_node(i + 1, j,     k    ),
                    ensure_node(i + 1, j + 1, k    ),
                    ensure_node(i,     j + 1, k    ),
                    ensure_node(i,     j,     k + 1),
                    ensure_node(i + 1, j,     k + 1),
                    ensure_node(i + 1, j + 1, k + 1),
                    ensure_node(i,     j + 1, k + 1),
                };
                for (int t = 0; t < 6; ++t) {
                    tets_flat.push_back(corners[Tdef[t][0]] + 1);
                    tets_flat.push_back(corners[Tdef[t][1]] + 1);
                    tets_flat.push_back(corners[Tdef[t][2]] + 1);
                    tets_flat.push_back(corners[Tdef[t][3]] + 1);
                }
                /* Boundary faces — emit when this cell's neighbour is
                 * NOT inside.  Same orientation as multicuboid. */
                if (!inside_idx(i, j, k - 1)) {  /* face 1: -z */
                    add_tri(1, corners[0], corners[2], corners[1]);
                    add_tri(1, corners[0], corners[3], corners[2]);
                }
                if (!inside_idx(i, j, k + 1)) {  /* face 2: +z */
                    add_tri(2, corners[4], corners[5], corners[6]);
                    add_tri(2, corners[4], corners[6], corners[7]);
                }
                if (!inside_idx(i, j - 1, k)) {  /* face 3: -y */
                    add_tri(3, corners[0], corners[1], corners[5]);
                    add_tri(3, corners[0], corners[5], corners[4]);
                }
                if (!inside_idx(i, j + 1, k)) {  /* face 4: +y */
                    add_tri(4, corners[3], corners[7], corners[6]);
                    add_tri(4, corners[3], corners[6], corners[2]);
                }
                if (!inside_idx(i - 1, j, k)) {  /* face 5: -x */
                    add_tri(5, corners[0], corners[4], corners[7]);
                    add_tri(5, corners[0], corners[7], corners[3]);
                }
                if (!inside_idx(i + 1, j, k)) {  /* face 6: +x */
                    add_tri(6, corners[1], corners[2], corners[6]);
                    add_tri(6, corners[1], corners[6], corners[5]);
                }
            }
        }
    }

    int64_t Nn_vol = (int64_t)(vol_nodes.size() / 3);
    int64_t Nt_vol = (int64_t)(tets_flat.size()  / 4);
    int64_t Nbnd   = (int64_t)face_id.size();

    matlab_mat *Nodes = mat_alloc(Nn_vol, 3);
    memcpy(Nodes->data, vol_nodes.data(), sizeof(double) * (size_t)(Nn_vol * 3));
    matlab_mat *Tets  = mat_alloc(Nt_vol, 4);
    for (int64_t i = 0; i < Nt_vol * 4; ++i) Tets->data[i] = (double)tets_flat[(size_t)i];
    matlab_mat *Faces = mat_alloc(Nbnd, 4);
    for (int64_t k = 0; k < Nbnd; ++k) {
        Faces->data[k * 4 + 0] = (double)face_id[(size_t)k];
        Faces->data[k * 4 + 1] = (double)face_n1[(size_t)k];
        Faces->data[k * 4 + 2] = (double)face_n2[(size_t)k];
        Faces->data[k * 4 + 3] = (double)face_n3[(size_t)k];
    }
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "Nodes", 5, Nodes);
    matlab_struct_set_mat(out, "Tets",  4, Tets);
    matlab_struct_set_mat(out, "Faces", 5, Faces);
    matlab_struct_set_f64(out, "Nx", 2, (double)Nx);
    matlab_struct_set_f64(out, "Ny", 2, (double)Ny);
    matlab_struct_set_f64(out, "Nz", 2, (double)Nz);
    matlab_struct_set_f64(out, "W",  1, xmax - xmin);
    matlab_struct_set_f64(out, "D",  1, ymax - ymin);
    matlab_struct_set_f64(out, "H",  1, zmax - zmin);
    matlab_struct_set_f64(out, "NumInsideCells", 14, (double)inside_count);
    return out;
}

/* --- 3-D scalar Poisson sparse assembly --------------------------- *
 *
 * Used by the Tier-3 thermal / electrostatic / dcConduction analysis
 * paths (all of which discretise as -∇·(c∇u) + au = f on the
 * volumetric tet mesh).  The same element-K formula as the 2-D
 * sparse path, except in 3-D with linear tet shape functions.
 *
 * Output struct: { K (sparse Nn x Nn), F (dense Nn x 1) }.
 */

void *matlab_pde_assemble_elast_3d_sparse(matlab_struct *mesh,
                                          double E, double nu);

matlab_struct *matlab_pde_assemble_poisson_3d_sparse(matlab_struct *mesh,
                                                     double c, double a,
                                                     double f) {
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *tets  = matlab_struct_get_mat(mesh, "Tets",  4);
    if (!nodes || !tets) return nullptr;
    int64_t Nn = nodes->rows;
    int64_t Nt = tets->rows;

    /* 16 K entries per tet + 4 F entries. */
    int64_t cap = Nt * 16;
    matlab_mat *I = mat_alloc(cap, 1);
    matlab_mat *J = mat_alloc(cap, 1);
    matlab_mat *V = mat_alloc(cap, 1);
    matlab_mat *F = mat_alloc(Nn, 1);
    int64_t pos = 0;

    for (int64_t e = 0; e < Nt; ++e) {
        int64_t ids[4];
        double X[4][3];
        for (int i = 0; i < 4; ++i) {
            ids[i] = (int64_t)tets->data[e * 4 + i] - 1;
            X[i][0] = nodes->data[ids[i] * 3 + 0];
            X[i][1] = nodes->data[ids[i] * 3 + 1];
            X[i][2] = nodes->data[ids[i] * 3 + 2];
        }
        double dN[4][3];
        double Vol;
        extern void elast_compute_grad(const double X[4][3], double dN[4][3],
                                        double *vol_out);
        elast_compute_grad(X, dN, &Vol);
        for (int p = 0; p < 4; ++p) {
            for (int q = 0; q < 4; ++q) {
                double Ke = c * Vol * (dN[p][0] * dN[q][0] +
                                        dN[p][1] * dN[q][1] +
                                        dN[p][2] * dN[q][2]);
                if (p == q) Ke += a * Vol / 4.0;
                I->data[pos] = (double)(ids[p] + 1);
                J->data[pos] = (double)(ids[q] + 1);
                V->data[pos] = Ke;
                pos++;
            }
            F->data[ids[p]] += f * Vol / 4.0;
        }
    }
    I->rows = pos; J->rows = pos; V->rows = pos;
    void *K = matlab_sparse_from_triplets(I, J, V, (double)Nn, (double)Nn);
    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, (matlab_mat *)K);
    matlab_struct_set_mat(out, "F", 1, F);
    return out;
}

/* Scalar Dirichlet — clamp `u = u_val` on a set of node ids.
 * Rebuilds the sparse triplets, dropping row r entries for fixed
 * r and inserting diagonal-1, then dropping col c entries for fixed
 * c (the column entries contribute F[r] -= K[r,c] * u_val).
 *
 * Returns { K (sparse), F (dense) } matching the structural pattern.
 */
matlab_struct *matlab_pde_apply_dirichlet_3d_sparse(void *K_sparse,
                                                     matlab_mat *F,
                                                     matlab_mat *node_ids,
                                                     double u_val) {
    struct sparse_view {
        uint32_t magic, _pad;
        int64_t *row_ptr;
        int64_t *col_idx;
        double  *vals;
        int64_t rows, cols, nnz;
    };
    sparse_view *S = (sparse_view *)K_sparse;
    if (!S || S->magic != 0xC0FFEE05u) return nullptr;
    int64_t Nn = S->rows;
    int64_t Nd = node_ids->rows * node_ids->cols;
    std::vector<int8_t> fixed((size_t)Nn, 0);
    for (int64_t k = 0; k < Nd; ++k) {
        int64_t n = (int64_t)node_ids->data[k] - 1;
        if (n >= 0 && n < Nn) fixed[(size_t)n] = 1;
    }

    matlab_mat *F2 = mat_alloc(Nn, 1);
    memcpy(F2->data, F->data, sizeof(double) * (size_t)Nn);
    if (u_val != 0.0) {
        for (int64_t r = 0; r < Nn; ++r) {
            int64_t lo = S->row_ptr[r];
            int64_t hi = S->row_ptr[r + 1];
            for (int64_t k = lo; k < hi; ++k) {
                int64_t c = S->col_idx[k];
                if (fixed[(size_t)c]) F2->data[r] -= S->vals[k] * u_val;
            }
        }
    }

    int64_t cap = S->nnz + Nn;
    matlab_mat *I = mat_alloc(cap, 1);
    matlab_mat *J = mat_alloc(cap, 1);
    matlab_mat *V = mat_alloc(cap, 1);
    int64_t pos = 0;
    for (int64_t r = 0; r < Nn; ++r) {
        int64_t lo = S->row_ptr[r];
        int64_t hi = S->row_ptr[r + 1];
        if (fixed[(size_t)r]) continue;
        for (int64_t k = lo; k < hi; ++k) {
            int64_t c = S->col_idx[k];
            if (fixed[(size_t)c]) continue;
            I->data[pos] = (double)(r + 1);
            J->data[pos] = (double)(c + 1);
            V->data[pos] = S->vals[k];
            pos++;
        }
    }
    for (int64_t r = 0; r < Nn; ++r) {
        if (!fixed[(size_t)r]) continue;
        I->data[pos] = (double)(r + 1);
        J->data[pos] = (double)(r + 1);
        V->data[pos] = 1.0;
        pos++;
        F2->data[r] = u_val;
    }
    I->rows = pos; J->rows = pos; V->rows = pos;
    void *K2 = matlab_sparse_from_triplets(I, J, V, (double)Nn, (double)Nn);

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, (matlab_mat *)K2);
    matlab_struct_set_mat(out, "F", 1, F2);
    return out;
}

/* Surface "scalar flux" load — adds heat/charge contributions to the
 * RHS F by integrating a constant value over each boundary triangle
 * of `face_id`.  Integral = q * area for piecewise-linear basis on a
 * triangle, distributed equally to its 3 corner nodes.
 */
matlab_mat *matlab_pde_face_scalar_load_3d(matlab_struct *mesh,
                                            double face_id_d, double q) {
    int64_t fid = (int64_t)face_id_d;
    matlab_mat *nodes = matlab_struct_get_mat(mesh, "Nodes", 5);
    matlab_mat *faces = matlab_struct_get_mat(mesh, "Faces", 5);
    if (!nodes || !faces) return mat_alloc(0, 0);
    int64_t Nn = nodes->rows;
    int64_t Nf = faces->rows;
    matlab_mat *F = mat_alloc(Nn, 1);
    for (int64_t k = 0; k < Nf; ++k) {
        if ((int64_t)faces->data[k * 4 + 0] != fid) continue;
        int64_t i0 = (int64_t)faces->data[k * 4 + 1] - 1;
        int64_t i1 = (int64_t)faces->data[k * 4 + 2] - 1;
        int64_t i2 = (int64_t)faces->data[k * 4 + 3] - 1;
        double *p0 = nodes->data + i0 * 3;
        double *p1 = nodes->data + i1 * 3;
        double *p2 = nodes->data + i2 * 3;
        double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        double e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        double n[3] = {
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]
        };
        double mag = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        double area = 0.5 * mag;
        double s = q * area / 3.0;
        F->data[i0] += s;
        F->data[i1] += s;
        F->data[i2] += s;
    }
    return F;
}

/* Fixed-DOF Dirichlet on a sparse 3-D elasticity K, F pair.  Same
 * shape as matlab_pde_apply_fixed_3d but operates on triplets. */
matlab_struct *matlab_pde_apply_fixed_3d_sparse(void *K_sparse,
                                                 matlab_mat *F,
                                                 matlab_mat *node_ids) {
    struct sparse_view {
        uint32_t magic, _pad;
        int64_t *row_ptr;
        int64_t *col_idx;
        double  *vals;
        int64_t rows, cols, nnz;
    };
    sparse_view *S = (sparse_view *)K_sparse;
    if (!S || S->magic != 0xC0FFEE05u) return nullptr;
    int64_t Ndof = S->rows;
    int64_t Nn = Ndof / 3;
    int64_t Nd = node_ids->rows * node_ids->cols;

    std::vector<int8_t> fixed((size_t)Ndof, 0);
    for (int64_t k = 0; k < Nd; ++k) {
        int64_t n = (int64_t)node_ids->data[k] - 1;
        if (n < 0 || n >= Nn) continue;
        fixed[(size_t)(n * 3 + 0)] = 1;
        fixed[(size_t)(n * 3 + 1)] = 1;
        fixed[(size_t)(n * 3 + 2)] = 1;
    }

    matlab_mat *F2 = mat_alloc(Ndof, 1);
    memcpy(F2->data, F->data, sizeof(double) * (size_t)Ndof);

    /* Rebuild triplets, filtering. */
    int64_t cap = S->nnz + Ndof;
    matlab_mat *I = mat_alloc(cap, 1);
    matlab_mat *J = mat_alloc(cap, 1);
    matlab_mat *V = mat_alloc(cap, 1);
    int64_t pos = 0;
    for (int64_t r = 0; r < Ndof; ++r) {
        int64_t lo = S->row_ptr[r];
        int64_t hi = S->row_ptr[r + 1];
        if (fixed[(size_t)r]) continue;
        for (int64_t k = lo; k < hi; ++k) {
            int64_t c = S->col_idx[k];
            if (fixed[(size_t)c]) continue;
            I->data[pos] = (double)(r + 1);
            J->data[pos] = (double)(c + 1);
            V->data[pos] = S->vals[k];
            pos++;
        }
    }
    for (int64_t r = 0; r < Ndof; ++r) {
        if (!fixed[(size_t)r]) continue;
        I->data[pos] = (double)(r + 1);
        J->data[pos] = (double)(r + 1);
        V->data[pos] = 1.0;
        pos++;
        F2->data[r] = 0.0;
    }
    I->rows = pos; J->rows = pos; V->rows = pos;
    void *K2 = matlab_sparse_from_triplets(I, J, V, (double)Ndof, (double)Ndof);

    matlab_struct *out = matlab_struct_new();
    matlab_struct_set_mat(out, "K", 1, (matlab_mat *)K2);
    matlab_struct_set_mat(out, "F", 1, F2);
    return out;
}

}  /* extern "C" */

/* ===================================================================
 * GLB (glTF 2.0 binary) importer.
 *
 * Container layout:
 *   12-byte header: magic "glTF" + uint32 version (2) + uint32 length
 *   Chunks: uint32 length + uint32 type + data
 *     - type 0x4E4F534A "JSON" — scene description
 *     - type 0x004E4942 "BIN\0" — packed vertex/index data
 *
 * v1 extracts the first mesh's first primitive:
 *   meshes[0].primitives[0].attributes.POSITION  → Nx3 floats
 *   meshes[0].primitives[0].indices              → triangle indices
 *
 * Scene-graph node transforms are ignored (a real glTF scene applies
 * a hierarchy of TRS transforms to each mesh instance; for surface
 * visualisation of a single asset that's adequate).  No texture /
 * material / animation / morph-target parsing.
 * =================================================================== */

namespace {

/* Minimal JSON value tree.  Numbers are stored as doubles, strings as
 * std::string, arrays as vector<JsonV>, objects as a flat vector of
 * (key, JsonV) pairs (so we walk in source order — adequate at the
 * sizes glTF uses). */
struct JsonV {
    enum Type { TNull, TBool, TNum, TStr, TArr, TObj } type = TNull;
    bool   b = false;
    double n = 0;
    std::string s;
    std::vector<JsonV> a;
    std::vector<std::pair<std::string, JsonV>> o;

    const JsonV *find(const char *key) const {
        if (type != TObj) return nullptr;
        for (auto &kv : o) if (kv.first == key) return &kv.second;
        return nullptr;
    }
    const JsonV *at(size_t i) const {
        if (type != TArr || i >= a.size()) return nullptr;
        return &a[i];
    }
    double as_num(double def = 0) const { return type == TNum ? n : def; }
};

struct JsonParser {
    const char *p;
    const char *end;
    bool ok = true;
    void skipWS() {
        while (p < end && (*p == ' ' || *p == '\t' ||
                            *p == '\r' || *p == '\n')) ++p;
    }
    bool consume(char c) {
        skipWS();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }
    JsonV parseValue() {
        skipWS();
        if (p >= end) { ok = false; return {}; }
        char c = *p;
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        return parseNumber();
    }
    JsonV parseObject() {
        JsonV v; v.type = JsonV::TObj;
        ++p;  /* skip { */
        skipWS();
        if (consume('}')) return v;
        while (p < end) {
            skipWS();
            if (*p != '"') { ok = false; return v; }
            JsonV key = parseString();
            if (!ok) return v;
            if (!consume(':')) { ok = false; return v; }
            JsonV val = parseValue();
            if (!ok) return v;
            v.o.emplace_back(key.s, std::move(val));
            skipWS();
            if (consume('}')) return v;
            if (!consume(',')) { ok = false; return v; }
        }
        ok = false; return v;
    }
    JsonV parseArray() {
        JsonV v; v.type = JsonV::TArr;
        ++p;  /* skip [ */
        skipWS();
        if (consume(']')) return v;
        while (p < end) {
            v.a.push_back(parseValue());
            if (!ok) return v;
            skipWS();
            if (consume(']')) return v;
            if (!consume(',')) { ok = false; return v; }
        }
        ok = false; return v;
    }
    JsonV parseString() {
        JsonV v; v.type = JsonV::TStr;
        ++p;  /* skip " */
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                char e2 = p[1];
                if (e2 == '"') v.s += '"';
                else if (e2 == '\\') v.s += '\\';
                else if (e2 == '/') v.s += '/';
                else if (e2 == 'n') v.s += '\n';
                else if (e2 == 't') v.s += '\t';
                else if (e2 == 'r') v.s += '\r';
                else if (e2 == 'b') v.s += '\b';
                else if (e2 == 'f') v.s += '\f';
                else v.s += e2;  /* \uXXXX / others — pass through best-effort */
                p += 2;
            } else {
                v.s += *p++;
            }
        }
        if (p < end && *p == '"') ++p;
        else ok = false;
        return v;
    }
    JsonV parseBool() {
        JsonV v; v.type = JsonV::TBool;
        if (p + 4 <= end && memcmp(p, "true", 4) == 0) {
            v.b = true; p += 4; return v;
        }
        if (p + 5 <= end && memcmp(p, "false", 5) == 0) {
            v.b = false; p += 5; return v;
        }
        ok = false; return v;
    }
    JsonV parseNull() {
        JsonV v; v.type = JsonV::TNull;
        if (p + 4 <= end && memcmp(p, "null", 4) == 0) {
            p += 4; return v;
        }
        ok = false; return v;
    }
    JsonV parseNumber() {
        JsonV v; v.type = JsonV::TNum;
        char *q = nullptr;
        v.n = strtod(p, &q);
        if (q == p) { ok = false; return v; }
        p = q;
        return v;
    }
};

static JsonV parse_json(const char *data, size_t n) {
    JsonParser jp{data, data + n, true};
    JsonV r = jp.parseValue();
    if (!jp.ok) r = {};
    return r;
}

/* Read an index of an unspecified width (u8 / u16 / u32) from the BIN
 * chunk.  glTF component types: 5121=uint8, 5123=uint16, 5125=uint32. */
static uint32_t read_index(const uint8_t *p, int component_type) {
    switch (component_type) {
        case 5121: return (uint32_t)p[0];
        case 5123: return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
        case 5125: return rd_u32_le(p);
        default:   return rd_u32_le(p);
    }
}
static int component_size(int component_type) {
    switch (component_type) {
        case 5120: case 5121: return 1;
        case 5122: case 5123: return 2;
        case 5125: case 5126: return 4;
        default: return 4;
    }
}

}  /* anonymous namespace */

extern "C" {

matlab_struct *matlab_pde_load_glb_path(const char *path, int64_t plen) {
    std::string fn(path, (size_t)plen);
    std::ifstream f(fn, std::ios::binary);
    if (!f) return nullptr;
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    if (sz <= 12) return nullptr;
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf((size_t)sz);
    if (!f.read((char *)buf.data(), sz)) return nullptr;

    /* Header: "glTF" + version + total length. */
    if (memcmp(buf.data(), "glTF", 4) != 0) return nullptr;
    uint32_t version = rd_u32_le(buf.data() + 4);
    if (version != 2) return nullptr;
    uint32_t total = rd_u32_le(buf.data() + 8);
    if (total > buf.size()) total = (uint32_t)buf.size();

    const uint8_t *json_data = nullptr;
    uint32_t       json_len  = 0;
    const uint8_t *bin_data  = nullptr;
    uint32_t       bin_len   = 0;

    /* Walk chunks. */
    size_t off = 12;
    while (off + 8 <= total) {
        uint32_t clen = rd_u32_le(buf.data() + off);
        uint32_t ctype = rd_u32_le(buf.data() + off + 4);
        if (off + 8 + clen > total) break;
        const uint8_t *cdata = buf.data() + off + 8;
        if (ctype == 0x4E4F534Au) {     /* "JSON" */
            json_data = cdata;
            json_len  = clen;
        } else if (ctype == 0x004E4942u) { /* "BIN\0" */
            bin_data = cdata;
            bin_len  = clen;
        }
        off += 8 + clen;
    }
    if (!json_data || !bin_data) return nullptr;
    (void)bin_len;

    /* Parse JSON. */
    JsonV root = parse_json((const char *)json_data, (size_t)json_len);
    if (root.type != JsonV::TObj) return nullptr;
    const JsonV *meshes      = root.find("meshes");
    const JsonV *accessors   = root.find("accessors");
    const JsonV *bufferViews = root.find("bufferViews");
    if (!meshes || !accessors || !bufferViews) return nullptr;

    /* Helper to fetch a bufferView byteOffset + length + stride for
     * a given accessor index. */
    auto get_accessor_view = [&](int acc_idx,
                                 int &component_type,
                                 int &count,
                                 int &byte_offset,
                                 int &byte_stride) -> bool {
        const JsonV *acc = accessors->at((size_t)acc_idx);
        if (!acc) return false;
        const JsonV *bv_idx_v = acc->find("bufferView");
        const JsonV *ct_v     = acc->find("componentType");
        const JsonV *cnt_v    = acc->find("count");
        if (!bv_idx_v || !ct_v || !cnt_v) return false;
        int bv_idx = (int)bv_idx_v->as_num();
        const JsonV *bv = bufferViews->at((size_t)bv_idx);
        if (!bv) return false;
        const JsonV *bv_off_v = bv->find("byteOffset");
        const JsonV *bv_str_v = bv->find("byteStride");
        int bv_off = bv_off_v ? (int)bv_off_v->as_num() : 0;
        int bv_str = bv_str_v ? (int)bv_str_v->as_num() : 0;
        const JsonV *acc_off_v = acc->find("byteOffset");
        int acc_off = acc_off_v ? (int)acc_off_v->as_num() : 0;
        component_type = (int)ct_v->as_num();
        count          = (int)cnt_v->as_num();
        byte_offset    = bv_off + acc_off;
        byte_stride    = bv_str;
        return true;
    };

    /* Take meshes[0].primitives[0]. */
    const JsonV *m0 = meshes->at(0);
    if (!m0) return nullptr;
    const JsonV *prims = m0->find("primitives");
    if (!prims || !prims->at(0)) return nullptr;
    const JsonV *p0    = prims->at(0);
    const JsonV *attrs = p0->find("attributes");
    if (!attrs) return nullptr;
    const JsonV *pos_idx_v = attrs->find("POSITION");
    if (!pos_idx_v) return nullptr;
    int pos_acc_idx = (int)pos_idx_v->as_num();

    /* POSITION attribute: VEC3 float (componentType 5126). */
    int pos_ct = 0, pos_count = 0, pos_off = 0, pos_str = 0;
    if (!get_accessor_view(pos_acc_idx, pos_ct, pos_count, pos_off, pos_str))
        return nullptr;
    if (pos_ct != 5126) return nullptr;  /* must be float */
    int pos_step = pos_str ? pos_str : 12;  /* 3 × 4-byte floats */

    std::vector<double> positions((size_t)pos_count * 3);
    for (int i = 0; i < pos_count; ++i) {
        const uint8_t *q = bin_data + pos_off + (size_t)i * pos_step;
        positions[(size_t)i * 3 + 0] = (double)rd_f32_le(q + 0);
        positions[(size_t)i * 3 + 1] = (double)rd_f32_le(q + 4);
        positions[(size_t)i * 3 + 2] = (double)rd_f32_le(q + 8);
    }

    /* Indices: optional.  If absent, draw as a triangle strip of the
     * positions in order (then count must be a multiple of 3 for a
     * TRIANGLES primitive). */
    std::vector<uint32_t> indices;
    const JsonV *idx_v = p0->find("indices");
    if (idx_v) {
        int idx_ct = 0, idx_count = 0, idx_off = 0, idx_str = 0;
        if (!get_accessor_view((int)idx_v->as_num(),
                               idx_ct, idx_count, idx_off, idx_str))
            return nullptr;
        int sz_per = component_size(idx_ct);
        if (!idx_str) idx_str = sz_per;
        indices.resize((size_t)idx_count);
        for (int i = 0; i < idx_count; ++i) {
            indices[(size_t)i] = read_index(
                bin_data + idx_off + (size_t)i * idx_str, idx_ct);
        }
    } else {
        indices.resize((size_t)pos_count);
        for (int i = 0; i < pos_count; ++i)
            indices[(size_t)i] = (uint32_t)i;
    }

    /* Default primitive mode is 4 (TRIANGLES).  Modes 5 (TRIANGLE_STRIP)
     * and 6 (TRIANGLE_FAN) are not handled in v1. */
    const JsonV *mode_v = p0->find("mode");
    int mode = mode_v ? (int)mode_v->as_num() : 4;
    if (mode != 4) return nullptr;
    if (indices.size() % 3 != 0) return nullptr;

    /* Push triangles through the welding hash so coincident verts
     * across primitives (or after f32 → f64 precision loss) collapse
     * to single nodes. */
    MeshBuilder mb;
    size_t ntri = indices.size() / 3;
    for (size_t t = 0; t < ntri; ++t) {
        uint32_t a = indices[t * 3 + 0];
        uint32_t b = indices[t * 3 + 1];
        uint32_t c = indices[t * 3 + 2];
        if ((int)a >= pos_count || (int)b >= pos_count || (int)c >= pos_count)
            continue;
        mb.add_triangle(positions[a * 3 + 0], positions[a * 3 + 1], positions[a * 3 + 2],
                        positions[b * 3 + 0], positions[b * 3 + 1], positions[b * 3 + 2],
                        positions[c * 3 + 0], positions[c * 3 + 1], positions[c * 3 + 2]);
    }
    return finalize_mesh(mb, "glb");
}

matlab_struct *matlab_pde_load_glb(void *s) {
    if (!s) return nullptr;
    auto *ms = (struct matlab_string_local_s *)s;
    return matlab_pde_load_glb_path(ms->data, ms->len);
}

}  /* extern "C" */
