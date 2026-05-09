% Discrete signal display: stem (DSP convention) and stairs (sample-and-hold).
n = 0:29;
y = sin(0.4 * n) .* exp(-0.05 * n);

figure;
subplot(2, 1, 1); stem(n, y);   title('stem(n, y)');   grid on;
subplot(2, 1, 2); stairs(n, y); title('stairs(n, y)'); grid on;
saveas(gcf, '/tmp/plot_stem_stairs.png');
