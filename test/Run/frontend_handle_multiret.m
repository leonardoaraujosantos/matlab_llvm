% #80: multi-return through a function handle (variable and struct field).
h = @twoout;
[a,b] = h(3);
fprintf('var a=%g b=%g\n', a, b);
s.f = @twoout;
[c,d] = s.f(4);
fprintf('field c=%g d=%g\n', c, d);
function [x,y] = twoout(z)
x = z + 1;
y = z * 2;
end
