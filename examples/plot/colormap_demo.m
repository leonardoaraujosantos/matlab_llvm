% Same Gaussian bump rendered with four colormaps for visual comparison.
[xx, yy] = meshgrid(linspace(-2, 2, 64), linspace(-2, 2, 64));
img = exp(-(xx.^2 + yy.^2));

figure;
subplot(2, 2, 1); imshow(img); colormap('parula');  title('parula');
subplot(2, 2, 2); imshow(img); colormap('viridis'); title('viridis');
subplot(2, 2, 3); imshow(img); colormap('jet');     title('jet');
subplot(2, 2, 4); imshow(img); colormap('hot');     title('hot');
saveas(gcf, '/tmp/plot_colormaps.png');
