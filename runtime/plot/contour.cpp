#include "contour.h"

#include <algorithm>
#include <cmath>

namespace matlab_plot {

namespace {

inline double lerp(double v1, double v2, double L) {
    /* Caller guarantees v1 != v2 and L lies between them. */
    double t = (L - v1) / (v2 - v1);
    if (t < 0) t = 0; else if (t > 1) t = 1;
    return t;
}

/* Edges of the cell (data coords with row r, col c, 1-based axis):
 *   A = z[r,   c  ] at (c+1, r+1)
 *   B = z[r,   c+1] at (c+2, r+1)
 *   C = z[r+1, c+1] at (c+2, r+2)
 *   D = z[r+1, c  ] at (c+1, r+2)
 * AB along top, BC along right, CD along bottom, DA along left. */
struct Pt { double x, y; };

inline Pt edge_AB(double r, double c, double A, double B, double L) {
    return { (c + 1) + lerp(A, B, L), r + 1 };
}
inline Pt edge_BC(double r, double c, double B, double C, double L) {
    return { c + 2, (r + 1) + lerp(B, C, L) };
}
inline Pt edge_CD(double r, double c, double C, double D, double L) {
    return { (c + 2) - lerp(C, D, L), r + 2 };
}
inline Pt edge_DA(double r, double c, double D, double A, double L) {
    return { c + 1, (r + 2) - lerp(D, A, L) };
}

inline void emit(std::vector<double> &x, std::vector<double> &y,
                 Pt p, Pt q) {
    x.push_back(p.x); y.push_back(p.y);
    x.push_back(q.x); y.push_back(q.y);
}

}  // namespace

void marching_squares(const double *Z, int rows, int cols,
                      const std::vector<double> &levels,
                      std::vector<double> &out_x,
                      std::vector<double> &out_y) {
    if (!Z || rows < 2 || cols < 2) return;

    for (double L : levels) {
        for (int r = 0; r < rows - 1; ++r) {
            for (int c = 0; c < cols - 1; ++c) {
                double A = Z[ r      * cols +  c     ];
                double B = Z[ r      * cols + (c + 1)];
                double C = Z[(r + 1) * cols + (c + 1)];
                double D = Z[(r + 1) * cols +  c     ];

                int idx = ((A >= L) ? 8 : 0)
                        | ((B >= L) ? 4 : 0)
                        | ((C >= L) ? 2 : 0)
                        | ((D >= L) ? 1 : 0);

                /* Skip empty cases without computing edges. */
                if (idx == 0 || idx == 15) continue;

                /* Pre-compute edges that the case actually uses. */
                switch (idx) {
                    case 1:  emit(out_x, out_y, edge_DA(r, c, D, A, L), edge_CD(r, c, C, D, L)); break;
                    case 2:  emit(out_x, out_y, edge_BC(r, c, B, C, L), edge_CD(r, c, C, D, L)); break;
                    case 3:  emit(out_x, out_y, edge_BC(r, c, B, C, L), edge_DA(r, c, D, A, L)); break;
                    case 4:  emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_BC(r, c, B, C, L)); break;
                    case 5:
                        /* Saddle: B & D above. Separated topology — top-left
                         * corner connects up-left, bottom-right connects down-right. */
                        emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_DA(r, c, D, A, L));
                        emit(out_x, out_y, edge_BC(r, c, B, C, L), edge_CD(r, c, C, D, L));
                        break;
                    case 6:  emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_CD(r, c, C, D, L)); break;
                    case 7:  emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_DA(r, c, D, A, L)); break;
                    case 8:  emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_DA(r, c, D, A, L)); break;
                    case 9:  emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_CD(r, c, C, D, L)); break;
                    case 10:
                        /* Saddle: A & C above. Separated topology. */
                        emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_BC(r, c, B, C, L));
                        emit(out_x, out_y, edge_CD(r, c, C, D, L), edge_DA(r, c, D, A, L));
                        break;
                    case 11: emit(out_x, out_y, edge_AB(r, c, A, B, L), edge_BC(r, c, B, C, L)); break;
                    case 12: emit(out_x, out_y, edge_BC(r, c, B, C, L), edge_DA(r, c, D, A, L)); break;
                    case 13: emit(out_x, out_y, edge_BC(r, c, B, C, L), edge_CD(r, c, C, D, L)); break;
                    case 14: emit(out_x, out_y, edge_DA(r, c, D, A, L), edge_CD(r, c, C, D, L)); break;
                    default: break;
                }
            }
        }
    }
}

}  // namespace matlab_plot
