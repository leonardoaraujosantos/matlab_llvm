% Histogram of synthetic samples + area chart of a damped sin wave.
n = 1000;
samples = randn(1, n);            % standard-normal-ish samples

t = 0:0.1:6;
y = sin(t) .* exp(-0.05 * t);

figure;
subplot(1, 2, 1); histogram(samples, 30); title('histogram(N=1000, 30 bins)'); grid on;
subplot(1, 2, 2); area(t, y);             title('area(damped sin)');           grid on;
saveas(gcf, '/tmp/plot_hist_area.png');
