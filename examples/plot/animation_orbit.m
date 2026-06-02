% animation_orbit.m — a marker travelling along a Lissajous curve, saved to MP4.
%
% The canonical "dynamic property update" animation pattern (MATLAB's
% animation-techniques doc): fixed axis limits set once, the static path drawn
% behind a moving marker, one getframe per step fed to VideoWriter.
%
% Build with video support for a real file:
%   cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON -DMATLAB_LLVM_WITH_PLOT_FFMPEG=ON
% Run:
%   cat examples/plot/animation_orbit.m | build/matlabc -repl /dev/stdin

t = linspace(0, 2*pi, 90);
x = sin(2 * t);
y = cos(3 * t);

v = VideoWriter('/tmp/plot_orbit.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

figure(1);
for k = 1:numel(t)
    plot(x, y, 'Color', [0.8 0.8 0.8], 'LineWidth', 1);    % static path
    hold on;
    plot(x(k), y(k), 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    hold off;
    xlim([-1.2 1.2]); ylim([-1.2 1.2]);
    axis('equal');
    title('Lissajous orbit');
    writeVideo(v, getframe(gcf));
end

close(v);
disp('wrote /tmp/plot_orbit.mp4');
