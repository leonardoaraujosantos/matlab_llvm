% errorbar — sin wave with symmetric noise bars per point.
x = linspace(0.5, 10, 20);
y = sin(x);
e = 0.05 + 0.15 * abs(sin(7.3 * x + 1.1));   % deterministic noise envelope

figure;
errorbar(x, y, e);
title('errorbar(sin) ± noise');
xlabel('x');
ylabel('y');
grid on;
saveas(gcf, '/tmp/plot_errorbar.png');
