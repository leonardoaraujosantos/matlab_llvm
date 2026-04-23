s.x = 5;
n = 10;
% isstruct
disp(isstruct(s));
disp(isstruct(n));
% isfield
disp(isfield(s, 'x'));
disp(isfield(s, 'y'));
s.nested.val = 1;
disp(isfield(s, 'nested'));
