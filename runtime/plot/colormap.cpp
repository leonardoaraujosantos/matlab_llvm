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

/* matplotlib-parity sequential colormaps. Sampled at 16 evenly-
 * spaced points from the matplotlib reference source (BSD-3). */
constexpr float magma_lut[LUT_N][3] = {
    {0.0015f, 0.0005f, 0.0139f}, {0.0510f, 0.0298f, 0.1556f},
    {0.1376f, 0.0598f, 0.3061f}, {0.2580f, 0.0676f, 0.4179f},
    {0.3863f, 0.0941f, 0.4789f}, {0.5060f, 0.1346f, 0.5051f},
    {0.6228f, 0.1740f, 0.5093f}, {0.7406f, 0.2148f, 0.4956f},
    {0.8553f, 0.2667f, 0.4636f}, {0.9466f, 0.3576f, 0.4109f},
    {0.9879f, 0.4824f, 0.3713f}, {0.9970f, 0.6196f, 0.3742f},
    {0.9947f, 0.7515f, 0.4136f}, {0.9876f, 0.8769f, 0.4814f},
    {0.9871f, 0.9911f, 0.6010f}, {0.9871f, 0.9911f, 0.7491f},
};
constexpr float inferno_lut[LUT_N][3] = {
    {0.0015f, 0.0005f, 0.0139f}, {0.0560f, 0.0359f, 0.1715f},
    {0.1538f, 0.0464f, 0.3387f}, {0.2796f, 0.0577f, 0.4421f},
    {0.4076f, 0.0911f, 0.4731f}, {0.5219f, 0.1273f, 0.4711f},
    {0.6322f, 0.1665f, 0.4533f}, {0.7445f, 0.2168f, 0.4174f},
    {0.8460f, 0.2810f, 0.3608f}, {0.9311f, 0.3724f, 0.2855f},
    {0.9810f, 0.4951f, 0.1995f}, {0.9961f, 0.6391f, 0.1284f},
    {0.9839f, 0.7793f, 0.0833f}, {0.9586f, 0.9094f, 0.1453f},
    {0.9882f, 0.9981f, 0.6446f}, {0.9882f, 0.9981f, 0.6446f},
};
constexpr float plasma_lut[LUT_N][3] = {
    {0.0506f, 0.0297f, 0.5279f}, {0.2052f, 0.0185f, 0.6066f},
    {0.3367f, 0.0090f, 0.6395f}, {0.4521f, 0.0073f, 0.6437f},
    {0.5566f, 0.0344f, 0.6321f}, {0.6517f, 0.0840f, 0.6107f},
    {0.7374f, 0.1397f, 0.5853f}, {0.8170f, 0.1985f, 0.5526f},
    {0.8888f, 0.2613f, 0.5118f}, {0.9477f, 0.3340f, 0.4546f},
    {0.9854f, 0.4163f, 0.3886f}, {0.9920f, 0.5168f, 0.3196f},
    {0.9778f, 0.6244f, 0.2533f}, {0.9514f, 0.7344f, 0.1907f},
    {0.9279f, 0.8454f, 0.1239f}, {0.9402f, 0.9755f, 0.1314f},
};
constexpr float cividis_lut[LUT_N][3] = {
    {0.0000f, 0.1262f, 0.3015f}, {0.0000f, 0.1697f, 0.3759f},
    {0.0892f, 0.2168f, 0.4123f}, {0.1838f, 0.2587f, 0.4220f},
    {0.2528f, 0.3036f, 0.4172f}, {0.3120f, 0.3499f, 0.4154f},
    {0.3678f, 0.3957f, 0.4180f}, {0.4242f, 0.4419f, 0.4234f},
    {0.4831f, 0.4884f, 0.4276f}, {0.5450f, 0.5358f, 0.4290f},
    {0.6105f, 0.5841f, 0.4275f}, {0.6814f, 0.6336f, 0.4218f},
    {0.7556f, 0.6849f, 0.4129f}, {0.8324f, 0.7378f, 0.3996f},
    {0.9136f, 0.7929f, 0.3801f}, {0.9954f, 0.8492f, 0.3534f},
};
constexpr float turbo_lut[LUT_N][3] = {
    {0.1900f, 0.0718f, 0.2322f}, {0.2700f, 0.2147f, 0.5043f},
    {0.2811f, 0.4060f, 0.7641f}, {0.2363f, 0.5860f, 0.9166f},
    {0.1696f, 0.7421f, 0.9583f}, {0.1326f, 0.8541f, 0.8830f},
    {0.1786f, 0.9255f, 0.7345f}, {0.3098f, 0.9722f, 0.5404f},
    {0.4881f, 0.9888f, 0.3725f}, {0.6739f, 0.9701f, 0.2659f},
    {0.8329f, 0.9023f, 0.2266f}, {0.9395f, 0.7886f, 0.2298f},
    {0.9810f, 0.6394f, 0.2125f}, {0.9514f, 0.4514f, 0.1574f},
    {0.8526f, 0.2737f, 0.0827f}, {0.7300f, 0.1500f, 0.0750f},
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
        case Colormap::Magma:   eval_lut(magma_lut,   t, r, g, b);  return;
        case Colormap::Inferno: eval_lut(inferno_lut, t, r, g, b);  return;
        case Colormap::Plasma:  eval_lut(plasma_lut,  t, r, g, b);  return;
        case Colormap::Cividis: eval_lut(cividis_lut, t, r, g, b);  return;
        case Colormap::Turbo:   eval_lut(turbo_lut,   t, r, g, b);  return;
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
    if (name == "magma")   return Colormap::Magma;
    if (name == "inferno") return Colormap::Inferno;
    if (name == "plasma")  return Colormap::Plasma;
    if (name == "cividis") return Colormap::Cividis;
    if (name == "turbo")   return Colormap::Turbo;
    return Colormap::Parula;
}

}  // namespace matlab_plot
