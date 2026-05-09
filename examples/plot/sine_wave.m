% Sine wave — the simplest plot.
x = 0:0.05:10;
y = sin(x);

figure;
plot(x, y, 'r--');
title('sin(x) on [0, 10]');
xlabel('x');
ylabel('sin(x)');
grid on;
saveas(gcf, '/tmp/plot_sine.png');
