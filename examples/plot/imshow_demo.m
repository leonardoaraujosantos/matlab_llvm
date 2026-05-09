% Display a 2-D Gaussian bump as a grayscale image.
N = 64;
[xx, yy] = meshgrid(linspace(-2, 2, N), linspace(-2, 2, N));
img = exp(-(xx.^2 + yy.^2));

figure;
imshow(img);
title('imshow: Gaussian bump');
saveas(gcf, '/tmp/plot_imshow.png');
