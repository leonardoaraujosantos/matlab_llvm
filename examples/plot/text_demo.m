% text — annotations placed at data coordinates.
x = 0:0.05:10;
y = sin(x);

figure;
plot(x, y, 'b-');
title('text annotations');
text(pi/2,           0.85, 'first peak');
text(3*pi/2 - 0.4,  -0.85, 'first trough');
text(5*pi/2 - 0.4,   0.85, 'second peak');
grid on;
saveas(gcf, '/tmp/plot_text.png');
