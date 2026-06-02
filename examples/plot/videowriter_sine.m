% videowriter_sine.m — animate a travelling sine wave and save it to MP4.
%
% Demonstrates the animation/video surface (docs/plotting.md §4 Tier A/B):
%   getframe     — capture the current figure as a frame
%   VideoWriter  — open an encoder for a container file
%   writeVideo   — append each captured frame
%   close        — flush + finalise the file
%
% Build matlabc with video support for this to produce a real file:
%   cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON -DMATLAB_LLVM_WITH_PLOT_FFMPEG=ON
% (needs the FFmpeg dev libraries). Without it, VideoWriter reports that
% video support is disabled.
%
% Run:
%   cat examples/plot/videowriter_sine.m | build/matlabc -repl /dev/stdin

x = linspace(0, 2*pi, 200);

v = VideoWriter('/tmp/sine_wave.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

figure(1);
for k = 1:60
    y = sin(x + k * 0.1);
    plot(x, y, 'LineWidth', 2, 'Color', 'b');
    xlim([0 2*pi]);          % fixed axes so the frame size stays constant
    ylim([-1.2 1.2]);
    title('Travelling sine wave');
    writeVideo(v, getframe(gcf));
end

close(v);
disp('wrote /tmp/sine_wave.mp4');
