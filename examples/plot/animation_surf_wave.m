% animation_surf_wave.m — an animated 3-D surface (radial ripple), saved as
% Motion JPEG AVI.
%
% Exercises the second VideoWriter profile ('Motion JPEG AVI' -> MJPEG/AVI)
% and the Quality property, re-drawing a surf() each frame as a travelling
% wave sweeps outward.
%
% Run:
%   cat examples/plot/animation_surf_wave.m | build/matlabc -repl /dev/stdin

[xx, yy] = meshgrid(linspace(-4, 4, 24), linspace(-4, 4, 24));
r = sqrt(xx.^2 + yy.^2) + 0.4;     % +0.4 keeps the centre finite

v = VideoWriter('/tmp/plot_wave.avi', 'Motion JPEG AVI');
v.FrameRate = 20;
v.Quality   = 90;
open(v);

figure(1);
colormap('viridis');
for k = 1:48
    z = sin(r - k * 0.3) ./ r;
    surf(xx, yy, z);
    title('Radial ripple');
    writeVideo(v, getframe(gcf));
end

close(v);
disp('wrote /tmp/plot_wave.avi');
