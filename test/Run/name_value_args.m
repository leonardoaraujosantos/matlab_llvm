% Name=Value call arguments (R2021a+): `f(Name=val)` lowers to the classic
% `f('Name', val)` pair. Exercised through plot (a name-value consumer).
% Also guards that a single `=` after an identifier triggers name-value but
% `==` stays a comparison.
x = 1:5;
plot(x, x, LineWidth=2, Color=[0 0 1]);
title('name=value');

a = 3;
if a == 3                 % `==` must NOT be parsed as a name-value argument
    disp('eq works');
end
disp('name=value ok');
