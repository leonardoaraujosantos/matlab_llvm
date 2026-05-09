% Compare default vs equal aspect ratio on a unit circle.
t = linspace(0, 2*pi, 200);
x = cos(t);
y = sin(t);

figure;
subplot(1, 2, 1);
plot(x, y, 'b-');
title('default aspect');
grid on;

subplot(1, 2, 2);
plot(x, y, 'b-');
axis equal;
title('axis equal');
grid on;

saveas(gcf, '/tmp/plot_axis.png');
