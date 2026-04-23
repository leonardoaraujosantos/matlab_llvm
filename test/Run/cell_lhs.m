C = {0, 0, 0};
C{1} = 10;
C{2} = 20;
C{3} = 30;
disp(C{1});
disp(C{2});
disp(C{3});
% Overwrite existing
C{2} = 999;
disp(C{2});
% Write past end auto-grows
C{5} = 555;
disp(C{5});
