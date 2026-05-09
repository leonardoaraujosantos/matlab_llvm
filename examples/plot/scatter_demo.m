% Scatter of a noisy linear relationship.
n = 80;
x = linspace(0, 10, n);
y = 0.5 * x + 0.3 * randn(1, n);

figure;
scatter(x, y);
title('Noisy linear scatter');
xlabel('x');
ylabel('0.5 x + noise');
grid on;
saveas(gcf, '/tmp/plot_scatter.png');
