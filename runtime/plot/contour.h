#ifndef MATLAB_PLOT_CONTOUR_H
#define MATLAB_PLOT_CONTOUR_H

#include <vector>

namespace matlab_plot {

/* Marching-squares contour generator. Z is row-major, rows*cols doubles.
 * For each level in `levels`, append the resulting line segments to
 * `out_x` / `out_y` as endpoint pairs:
 *   segment k goes from (out_x[2k], out_y[2k]) to (out_x[2k+1], out_y[2k+1]).
 *
 * Coordinate convention: x = column index in [1, cols], y = row index in
 * [1, rows]. Ambiguous saddle cases (5 and 10) are resolved by the
 * "separated" topology — each crossing connects to its nearest neighbor
 * around the cell. */
void marching_squares(const double *Z, int rows, int cols,
                      const std::vector<double> &levels,
                      std::vector<double> &out_x,
                      std::vector<double> &out_y);

}  // namespace matlab_plot

#endif
