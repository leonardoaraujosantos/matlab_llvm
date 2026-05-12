% Smith chart numeric grid: matched-load r=1 circle + |Γ|=1 unit circle.
% 8 points around each circle; verify the radii via complex-aware
% subscript (which returns the real part of a complex column entry).

g = smithGrid(1.0, 8.0);
R = smithRCircle(g);
U = smithUnitCircle(g);

disp(R);              % 8 points around r=1 circle (centered 0.5+0i, radius 0.5)
disp(U);              % 8 points around |Γ|=1 circle (centered 0+0i, radius 1)
