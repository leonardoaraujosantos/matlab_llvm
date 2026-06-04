% strjoin(C, delim): join a cell of strings with delim; 1-arg uses a space.
% Was unrecognised ("undefined name 'strjoin'").
c = {'a','b','c'};
fprintf('[%s]\n', strjoin(c, '-'));
fprintf('[%s]\n', strjoin(c));
d = {'foo','bar'};
fprintf('[%s]\n', strjoin(d, ', '));
e = {'only'};
fprintf('[%s]\n', strjoin(e, '-'));
