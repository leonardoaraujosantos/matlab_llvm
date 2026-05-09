% Bar chart of arbitrary categorical heights.
x = 1:6;
y = [3 7 2 8 5 4];

figure;
bar(x, y);
title('Quarterly counts');
xlabel('Quarter');
ylabel('Count');
grid on;
saveas(gcf, '/tmp/plot_bar.png');
