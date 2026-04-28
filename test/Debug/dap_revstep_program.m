% Fixture for the reverse-stepping (stepBack) scenario. Three
% straight-line assignments with no loop / branches; the DAP
% client breaks at line 4, walks back one statement at a time,
% and verifies the workspace rewinds in lock-step.
a = 100;
b = 200;
c = 300;
disp(c);
