% animation_fourbar.m — a four-bar linkage (crank-rocker) animated to MP4.
%
% A mechanism animation: the crank rotates a full turn while the coupler and
% rocker follow. The joint B is solved each frame from the vector-loop closure
% as the intersection of circle(A, r3) and circle(O4, r4). Fixed axes keep the
% frame size constant so VideoWriter gets uniform frames.
%
% Run:
%   cat examples/plot/animation_fourbar.m | build/matlabc -repl /dev/stdin

r1 = 4.0;            % ground link  (O2 -> O4)
r2 = 1.0;            % crank        (O2 -> A), shortest -> fully rotates
r3 = 3.5;            % coupler      (A  -> B)
r4 = 3.0;            % rocker       (O4 -> B)
O2 = [0 0];
O4 = [r1 0];

n = 90;
ang = linspace(0, 2*pi, n);

v = VideoWriter('/tmp/plot_fourbar.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

figure(1);
for k = 1:n
    th2 = ang(k);
    ax = r2 * cos(th2);
    ay = r2 * sin(th2);

    % Joint B = intersection of circle(A, r3) and circle(O4, r4).
    dx = O4(1) - ax;
    dy = O4(2) - ay;
    d  = sqrt(dx * dx + dy * dy);
    a  = (r3 * r3 - r4 * r4 + d * d) / (2 * d);
    h2 = r3 * r3 - a * a;            % >= 0 for a Grashof crank-rocker
    if h2 < 0
        h2 = 0;
    end
    h  = sqrt(h2);
    mx = ax + a * dx / d;
    my = ay + a * dy / d;
    bx = mx - h * dy / d;        % upper assembly branch
    by = my + h * dx / d;

    plot([O2(1) O4(1)], [O2(2) O4(2)], 'Color', [0.7 0.7 0.7], 'LineWidth', 2);
    hold on;
    plot([O2(1) ax], [O2(2) ay], 'b-', 'LineWidth', 3);     % crank
    plot([ax bx],    [ay by],    'r-', 'LineWidth', 3);     % coupler
    plot([O4(1) bx], [O4(2) by], 'g-', 'LineWidth', 3);     % rocker
    plot([O2(1) O4(1) ax bx], [O2(2) O4(2) ay by], ...
         'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');    % joints
    hold off;
    xlim([-2 6]); ylim([-4 4]);
    axis('equal');
    title('Four-bar linkage (crank-rocker)');
    writeVideo(v, getframe(gcf));
end

close(v);
disp('wrote /tmp/plot_fourbar.mp4');
