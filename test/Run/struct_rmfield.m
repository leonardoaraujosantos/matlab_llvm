s.x = 1;
s.y = 2;
s.z = 3;
disp(isfield(s, 'y'));
s = rmfield(s, 'y');
disp(isfield(s, 'y'));
disp(isfield(s, 'x'));
disp(isfield(s, 'z'));
disp(s.x);
disp(s.z);

% Remove non-existent field is a no-op
s = rmfield(s, 'absent');
disp(isfield(s, 'x'));
