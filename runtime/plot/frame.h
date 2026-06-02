#ifndef MATLAB_PLOT_FRAME_H
#define MATLAB_PLOT_FRAME_H

#include <cstdint>
#include <vector>

/* Internal definition of the opaque matlab_frame handle (see matlab_plot.h).
 * Holds one captured raster of a figure: premultiplied ARGB32 pixels in
 * native byte order (B,G,R,A on little-endian), tightly packed at
 * stride == width*4. Shared between c_api.cpp (getframe / registry) and
 * videowriter.cpp (the encoder reads these pixels). */
struct matlab_frame {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int stride = 0;   // bytes per row; width*4 for the packed copy
};

#endif  // MATLAB_PLOT_FRAME_H
