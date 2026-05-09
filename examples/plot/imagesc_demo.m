% imagesc — pseudocolor heatmap with axis ticks and a colorbar.
[xx, yy] = meshgrid(linspace(-3, 3, 60), linspace(-2, 2, 40));
z = sin(3 * xx) .* cos(2 * yy);

figure;
imagesc(z);
colormap('viridis');
colorbar;
title('imagesc + colorbar');
xlabel('column');
ylabel('row');
saveas(gcf, '/tmp/plot_imagesc.png');
