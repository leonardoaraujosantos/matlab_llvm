% #81: invoke a function handle held in a struct field.
s.h = @inc;
v = s.h(5);
fprintf('direct=%g\n', v);
h = @inc;
s.g = h;
fprintf('viavar=%g\n', s.g(10));
function y = inc(z)
y = z + 1;
end
