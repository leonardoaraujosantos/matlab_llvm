% 3-D helix via plot3 with the default view.
t = 0:0.1:20;
x = cos(t);
y = sin(t);
z = t * 0.2;

figure;
plot3(x, y, z);
title('plot3: helix');
saveas(gcf, '/tmp/plot_3d_helix.png');
