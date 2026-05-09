% Log-scale axes: exponential decay on semilogy, power law on loglog.
x  = linspace(0.1, 100, 80);
y_exp = exp(-x * 0.05) * 1000;     % straight on semilogy
y_pow = x .^ 2.5;                  % straight on loglog

figure;
subplot(2, 2, 1); plot(x, y_exp, 'b-'); title('linear: exp decay'); grid on;
subplot(2, 2, 2); plot(x, y_exp, 'b-'); semilogy;
                  title('semilogy: exp decay'); grid on;
subplot(2, 2, 3); plot(x, y_pow, 'b-'); title('linear: power law'); grid on;
subplot(2, 2, 4); plot(x, y_pow, 'b-'); loglog;
                  title('loglog: power law'); grid on;
saveas(gcf, '/tmp/plot_log.png');
