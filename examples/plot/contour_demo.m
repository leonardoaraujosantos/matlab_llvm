% Contour plot of the MATLAB peaks-like surface via marching squares.
[xx, yy] = meshgrid(linspace(-3, 3, 50), linspace(-3, 3, 50));
z = 3 * (1 - xx).^2 .* exp(-xx.^2 - (yy + 1).^2) ...
  - 10 * (xx/5 - xx.^3 - yy.^5) .* exp(-xx.^2 - yy.^2) ...
  - (1/3) * exp(-(xx + 1).^2 - yy.^2);

figure;
contour(z);
title('contour(peaks)');
grid on;
saveas(gcf, '/tmp/plot_contour.png');
