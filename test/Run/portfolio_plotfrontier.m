% Financial Toolbox Tier-7 — plotFrontier returns the frontier points;
% render them with the ordinary plot builtin.
m = [0.10; 0.12; 0.08];
C = [ 0.04 0.01 0.005; 0.01 0.05 0.008; 0.005 0.008 0.03 ];
p = Portfolio();
p = setAssetMoments(p, m, C);
p = setDefaultConstraints(p);

pts = plotFrontier(p, 20);          % Kx2 [risk, return]
risk = pts(:, 1);
ret  = pts(:, 2);
plot(risk, ret, 'b');
xlabel('Risk (std)');
ylabel('Expected return');
title('Efficient frontier');
saveas(gcf, '/tmp/portfolio_frontier.png');

fprintf('frontier points returned: %.0f\n', size(pts, 1));
fprintf('rendered: /tmp/portfolio_frontier.png\n');
