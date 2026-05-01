% Phase 4 — containers.Map / dictionary. Three constructor surfaces:
%   1. m = containers.Map(); m(k) = v; v = m(k);
%   2. d = dictionary(k1, v1, k2, v2, ...);
%   3. n = dictionary(); n(num) = v;  -- numeric keys

% containers.Map with string keys.
m = containers.Map();
m('alpha') = 1;
m('beta') = 2;
m('gamma') = 3;
disp(m('alpha'));            % 1
disp(m('beta'));             % 2
disp(m('gamma'));            % 3
disp(length(m));             % 3
disp(isKey(m, 'beta'));      % 1
disp(isKey(m, 'delta'));     % 0

% Update an existing key.
m('beta') = 200;
disp(m('beta'));             % 200

% dictionary(...) inline init.
d = dictionary("a", 10, "b", 20, "c", 30);
disp(d("a"));                % 10
disp(d("b"));                % 20
disp(d("c"));                % 30

% Numeric-keyed dictionary.
n = dictionary();
n(1.5) = 100;
n(2.5) = 200;
n(3.5) = 300;
disp(n(2.5));                % 200
disp(length(n));             % 3

% Remove.
remove(m, 'beta');
disp(isKey(m, 'beta'));      % 0
disp(length(m));             % 2
