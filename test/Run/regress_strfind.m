% strfind(s, pat): 1-based positions of every (overlapping) occurrence; [] if
% none / pat empty / pat longer than s. Was unrecognised ("undefined name").
k = strfind('abcabc', 'bc');
fprintf('basic n=%.0f at %.0f %.0f\n', numel(k), k(1), k(2));
m = strfind('aaaa', 'aa');
fprintf('overlap n=%.0f %.0f %.0f %.0f\n', numel(m), m(1), m(2), m(3));
e = strfind('abc', 'xyz');
fprintf('none n=%.0f\n', numel(e));
s = 'hello world';
q = strfind(s, 'o');
fprintf('var n=%.0f at %.0f %.0f\n', numel(q), q(1), q(2));
