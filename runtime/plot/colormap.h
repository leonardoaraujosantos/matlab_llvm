#ifndef MATLAB_PLOT_COLORMAP_H
#define MATLAB_PLOT_COLORMAP_H

#include <cstdint>
#include <string_view>

namespace matlab_plot {

enum class Colormap {
    Gray, Parula, Jet, Viridis, Hot, Cool
};

/* Map t ∈ [0, 1] to (r, g, b) ∈ [0, 1]^3. Out-of-range t is clamped. */
void cmap_eval(Colormap cm, double t, float &r, float &g, float &b);

/* Parse "gray", "parula", "jet", "viridis", "hot", "cool" (case-sensitive,
 * to match MATLAB). Unknown name returns Parula (the MATLAB default). */
Colormap cmap_from_name(std::string_view name);

}  // namespace matlab_plot

#endif
