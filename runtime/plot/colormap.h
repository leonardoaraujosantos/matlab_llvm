#ifndef MATLAB_PLOT_COLORMAP_H
#define MATLAB_PLOT_COLORMAP_H

#include <cstdint>
#include <string_view>

namespace matlab_plot {

enum class Colormap {
    Gray, Parula, Jet, Viridis, Hot, Cool,
    /* matplotlib-parity additions (2026-05 plotting Tier-3): closes
     * the gap for users porting matplotlib scripts. All four
     * perceptually-uniform sequential maps come from the
     * matplotlib source (BSD-licensed); turbo is the Google
     * perceptually-uniform rainbow. */
    Magma, Inferno, Plasma, Cividis, Turbo
};

/* Map t ∈ [0, 1] to (r, g, b) ∈ [0, 1]^3. Out-of-range t is clamped. */
void cmap_eval(Colormap cm, double t, float &r, float &g, float &b);

/* Parse "gray", "parula", "jet", "viridis", "hot", "cool", plus the
 * matplotlib-parity additions "magma" / "inferno" / "plasma" /
 * "cividis" / "turbo" (case-sensitive, to match MATLAB). Unknown
 * name returns Parula (the MATLAB default). */
Colormap cmap_from_name(std::string_view name);

}  // namespace matlab_plot

#endif
