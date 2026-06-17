# Plotting Spec

## Purpose
Documents the observed behavior of the headless plotting runtime: the Cairo rendering backend, supported 2-D/3-D plot types, figure/axes/decoration handling, color maps and line/marker styles, output to PNG/SVG/PDF, and animation capture via getframe + VideoWriter to MP4/AVI through libav. The runtime is headless by design with no interactive figures (src: runtime/matlab_plot.h, runtime/plot/; doc: docs/plotting.md).

## Requirements

### Requirement: Headless Cairo backend
The system SHALL render all plots through Cairo with no display server, subprocess, or temp files, using a per-thread current figure and offering no interactive (pan/zoom/rotate) capability.

#### Scenario: Render without a display
- **WHEN** a program produces a plot in an environment with no X11/GUI
- **THEN** the system SHALL render in-memory via Cairo image/SVG/PDF surfaces (default 800×600) without requiring a display server (src: runtime/matlab_plot.h header comment; runtime/plot/cairo_render.cpp; doc: docs/plotting.md)

### Requirement: 2-D plot types
The system SHALL provide 2-D plot types including line, scatter, bar, stem, stairs, area, errorbar, histogram, image (imshow/imagesc/pcolor), contour/contourf, and quiver.

#### Scenario: Draw a line and bar chart
- **WHEN** a program calls `plot`, `scatter`, `bar`, `stem`, `stairs`, `area`, `errorbar`, `histogram`, `imagesc`, `contour`, or `quiver`
- **THEN** the system SHALL add the corresponding series to the current axes and render it (src: runtime/plot/c_api.cpp matlab_plot2/matlab_scatter/matlab_bar1/matlab_contour/matlab_quiver; runtime/plot/figure.h SeriesKind)

### Requirement: 3-D plot types
The system SHALL provide 3-D plot types plot3, surf/surfc, and mesh/meshc using orthographic projection with a configurable view and painter's-algorithm depth sorting.

#### Scenario: Render a surface
- **WHEN** a program calls `plot3`, `surf`, `surfc`, `mesh`, or `meshc`, optionally with `view(az,el)`
- **THEN** the system SHALL project to 2-D, depth-sort faces, and shade surfaces via the active colormap (default view az -37.5°, el 30°) (src: runtime/plot/c_api.cpp matlab_plot3/matlab_surf1/matlab_mesh1/matlab_view; doc: docs/plotting.md)

### Requirement: Figure, axes, and decoration
The system SHALL manage figure lifecycle, subplots, and axes decoration including title, axis labels, text, legend, grid, hold, limits, ticks, and log scales.

#### Scenario: Lay out and annotate a figure
- **WHEN** a program calls `figure`, `subplot`, `gcf`, `close`, `title`, `xlabel`/`ylabel`/`zlabel`, `legend`, `grid`, `hold`, `xlim`/`ylim`, `xticks`, or `loglog`/`semilogx`/`semilogy`
- **THEN** the system SHALL apply the operation to the current figure/axes and render accordingly (src: runtime/plot/c_api.cpp matlab_figure_new/matlab_gcf/matlab_close_all/matlab_title/matlab_legend; runtime/matlab_plot.h matlab_subplot/matlab_xlim/matlab_loglog)

### Requirement: Color maps and styles
The system SHALL provide a set of colormaps and SHALL honor line-style, marker, color, width, and display-name styling via linespec strings and Name/Value pairs.

#### Scenario: Apply colormap and styling
- **WHEN** a program calls `colormap('jet')` or styles a series via a linespec (e.g. `'r--o'`) or Name/Value pairs (`'LineWidth'`, `'Color'`, `'LineStyle'`, `'Marker'`, `'MarkerSize'`, `'DisplayName'`)
- **THEN** the system SHALL apply the requested colormap (gray, parula, jet, viridis, hot, cool; unknown names fall back to parula) and series styling, auto-cycling the MATLAB R2014b+ 7-color palette when no color is given (src: runtime/plot/colormap.cpp; runtime/plot/c_api.cpp matlab_colormap; runtime/matlab_plot.h matlab_set_series_linewidth/matlab_set_series_marker)

### Requirement: Static output formats
The system SHALL export figures to PNG, SVG, and PDF, both to a file (saveas/print) and as in-memory byte buffers, selecting the format from the filename extension.

#### Scenario: Save and render to bytes
- **WHEN** a program calls `saveas`/`print` with a `.png`/`.svg`/`.pdf` path, or calls the in-memory render entry points
- **THEN** the system SHALL emit the correct format (PNG magic, SVG `<`, PDF `%PDF`), returning a status code for files or a caller-freed buffer for in-memory renders (src: runtime/matlab_plot.h matlab_savefig/matlab_render_png/matlab_render_svg/matlab_render_pdf/matlab_plot_buffer_free; runtime/plot/cairo_render.h Format; test: test/Runtime/test_plot_basic.cpp)

### Requirement: Animation capture and video export
The system SHALL capture rendered frames via getframe and encode them to MP4 (H.264) or AVI (Motion JPEG) through libav, when built with FFmpeg support.

#### Scenario: Write a video
- **WHEN** a program calls `getframe`, then `VideoWriter` (selecting profile by extension or name), sets frame rate/quality, and calls `open`/`writeVideo`/`close`, in a build defining `MATLAB_LLVM_WITH_PLOT_FFMPEG`
- **THEN** the system SHALL encode the ARGB32 frames to an `.mp4` (H.264, default 30 fps, quality 0-100) or `.avi` (MJPEG) container via libav, and SHALL otherwise return an error producing no file (src: runtime/matlab_plot.h matlab_getframe/matlab_videowriter_new/matlab_videowriter_set_framerate/matlab_videowriter_close; runtime/plot/videowriter.cpp; doc: docs/plotting.md)
