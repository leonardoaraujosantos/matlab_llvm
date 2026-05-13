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
