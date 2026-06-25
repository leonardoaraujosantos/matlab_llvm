% sim3d handle semantics + scalar-scale broadcast.
% a2 = a aliases the same underlying actor (handle class), so a mutation
% through a2 is visible through a. A scalar Scale broadcasts to [s s s].
a = sim3d.Actor('cube', 'box');
a2 = a;
a2.Translation = [7 8 9];
disp(a.Translation);     % 7 8 9 — shared handle
a.Scale = 3;
disp(a.Scale);           % 3 3 3 — scalar broadcast
