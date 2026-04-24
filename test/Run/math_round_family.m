% round / floor / ceil / fix — scalar + matrix, plus mod / rem.
% All shipped after the pi/e/Inf/NaN constants work as a tail of
% general math builtins the compiler can lower.
disp(round(3.7));
disp(floor(3.7));
disp(ceil(3.2));
disp(fix(-3.7));
disp(mod(10, 3));
disp(rem(-10, 3));
disp(floor([1.2 3.7; -0.5 2.9]));
