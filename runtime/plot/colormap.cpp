#include "colormap.h"

#include <algorithm>
#include <cmath>

namespace matlab_plot {

namespace {

inline float clamp01(float v) { return v < 0.f ? 0.f : v > 1.f ? 1.f : v; }

/* Computed maps (cheap; no LUT). */
void eval_gray(double t, float &r, float &g, float &b) {
    r = g = b = static_cast<float>(t);
}
void eval_jet(double t, float &r, float &g, float &b) {
    r = clamp01(static_cast<float>(1.5 - std::fabs(4.0 * t - 3.0)));
    g = clamp01(static_cast<float>(1.5 - std::fabs(4.0 * t - 2.0)));
    b = clamp01(static_cast<float>(1.5 - std::fabs(4.0 * t - 1.0)));
}
void eval_hot(double t, float &r, float &g, float &b) {
    r = clamp01(static_cast<float>(3.0 * t));
    g = clamp01(static_cast<float>(3.0 * t - 1.0));
    b = clamp01(static_cast<float>(3.0 * t - 2.0));
}
void eval_cool(double t, float &r, float &g, float &b) {
    r = static_cast<float>(t);
    g = 1.f - static_cast<float>(t);
    b = 1.f;
}

/* Tabulated maps — 16 sampled control points, linearly interpolated. The
 * sample values are abbreviated from the published 256-entry maps;
 * sufficient for typical figure resolutions, no perceptible banding at
 * <1000px wide outputs. */
constexpr int LUT_N = 16;

constexpr float parula_lut[LUT_N][3] = {
    {0.2422f, 0.1504f, 0.6603f}, {0.2510f, 0.2151f, 0.7773f},
    {0.2459f, 0.2870f, 0.8769f}, {0.2168f, 0.3675f, 0.9352f},
    {0.1480f, 0.4595f, 0.9404f}, {0.0938f, 0.5353f, 0.9197f},
    {0.0779f, 0.5904f, 0.8845f}, {0.0509f, 0.6308f, 0.8429f},
    {0.0335f, 0.6665f, 0.7944f}, {0.1136f, 0.6905f, 0.7287f},
    {0.2799f, 0.6957f, 0.6315f}, {0.4694f, 0.6809f, 0.4970f},
    {0.6510f, 0.6644f, 0.3543f}, {0.8298f, 0.6651f, 0.2228f},
    {0.9788f, 0.7085f, 0.1118f}, {0.9763f, 0.9831f, 0.0538f},
};

constexpr float viridis_lut[LUT_N][3] = {
    {0.2670f, 0.0049f, 0.3294f}, {0.2832f, 0.1313f, 0.4495f},
    {0.2773f, 0.2335f, 0.4974f}, {0.2576f, 0.3271f, 0.5132f},
    {0.2336f, 0.4163f, 0.5181f}, {0.2080f, 0.5034f, 0.5169f},
    {0.1830f, 0.5895f, 0.5114f}, {0.1644f, 0.6755f, 0.5020f},
    {0.1530f, 0.7596f, 0.4824f}, {0.1934f, 0.8400f, 0.4435f},
    {0.3047f, 0.9151f, 0.3849f}, {0.4775f, 0.9656f, 0.3170f},
    {0.6800f, 0.9882f, 0.2564f}, {0.8521f, 0.9925f, 0.2253f},
    {0.9614f, 0.9785f, 0.2278f}, {0.9932f, 0.9062f, 0.1439f},
};

void eval_lut(const float lut[LUT_N][3], double t,
              float &r, float &g, float &b) {
    if (t <= 0) { r = lut[0][0];          g = lut[0][1];          b = lut[0][2];          return; }
    if (t >= 1) { r = lut[LUT_N-1][0];    g = lut[LUT_N-1][1];    b = lut[LUT_N-1][2];    return; }
    double pos = t * (LUT_N - 1);
    int    i0  = static_cast<int>(pos);
    double f   = pos - i0;
    int    i1  = std::min(i0 + 1, LUT_N - 1);
    r = static_cast<float>(lut[i0][0] * (1 - f) + lut[i1][0] * f);
    g = static_cast<float>(lut[i0][1] * (1 - f) + lut[i1][1] * f);
    b = static_cast<float>(lut[i0][2] * (1 - f) + lut[i1][2] * f);
}

}  // namespace

void cmap_eval(Colormap cm, double t, float &r, float &g, float &b) {
    if (t < 0) t = 0; else if (t > 1) t = 1;
    switch (cm) {
        case Colormap::Gray:    eval_gray(t, r, g, b);              return;
        case Colormap::Parula:  eval_lut(parula_lut,  t, r, g, b);  return;
        case Colormap::Jet:     eval_jet(t, r, g, b);               return;
        case Colormap::Viridis: eval_lut(viridis_lut, t, r, g, b);  return;
        case Colormap::Hot:     eval_hot(t, r, g, b);               return;
        case Colormap::Cool:    eval_cool(t, r, g, b);              return;
    }
    eval_lut(parula_lut, t, r, g, b);
}

Colormap cmap_from_name(std::string_view name) {
    if (name == "gray")    return Colormap::Gray;
    if (name == "parula")  return Colormap::Parula;
    if (name == "jet")     return Colormap::Jet;
    if (name == "viridis") return Colormap::Viridis;
    if (name == "hot")     return Colormap::Hot;
    if (name == "cool")    return Colormap::Cool;
    return Colormap::Parula;
}

}  // namespace matlab_plot
