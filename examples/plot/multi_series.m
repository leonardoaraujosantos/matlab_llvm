% Two curves on the same axes via hold on.
x  = 0:0.05:10;
y1 = sin(x);
y2 = cos(x);

figure;
plot(x, y1, 'b-');
hold on;
plot(x, y2, 'r--');
title('sin and cos');
xlabel('x');
ylabel('y');
legend({'sin(x)', 'cos(x)'});
grid on;
saveas(gcf, '/tmp/plot_multi.png');
