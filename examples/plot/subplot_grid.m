% 2x2 subplot grid mixing line, scatter, bar, and image.
x  = 0:0.1:10;
y1 = sin(x);
y2 = cos(x);
bx = 1:5;
by = [4 7 2 8 5];
[xx, yy] = meshgrid(linspace(-2, 2, 32), linspace(-2, 2, 32));
img = exp(-(xx.^2 + yy.^2));

figure;
subplot(2, 2, 1);
plot(x, y1, 'b-');
title('sin');
xlabel('x'); ylabel('sin(x)');
grid on;

subplot(2, 2, 2);
scatter(x, y2);
title('scatter cos');
grid on;

subplot(2, 2, 3);
bar(bx, by);
title('bars');
grid on;

subplot(2, 2, 4);
imshow(img);
title('imshow');

saveas(gcf, '/tmp/plot_subplot.png');
