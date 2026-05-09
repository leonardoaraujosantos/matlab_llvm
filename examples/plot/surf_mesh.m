% mesh and surf — wireframe vs filled 3-D surface of a peaks-like function.
[xx, yy] = meshgrid(linspace(-3, 3, 30), linspace(-3, 3, 30));
z = 3 * (1 - xx).^2 .* exp(-xx.^2 - (yy + 1).^2) ...
  - 10 * (xx/5 - xx.^3 - yy.^5) .* exp(-xx.^2 - yy.^2) ...
  - (1/3) * exp(-(xx + 1).^2 - yy.^2);

figure;
subplot(1, 2, 1); mesh(z); title('mesh(peaks)');
subplot(1, 2, 2); colormap('viridis'); surf(z); title('surf(peaks)');
saveas(gcf, '/tmp/plot_3d_surface.png');
