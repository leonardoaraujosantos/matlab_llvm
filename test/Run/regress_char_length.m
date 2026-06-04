% length/numel/ndims/isempty on a CHAR value used to hit "unsupported call
% shape": a char literal lowers to matlab.const_char (tensor<1xNxi8>), which
% the matrix-based runtime size queries don't accept. The result is now folded
% from the literal's known shape. (Runtime-built matlab_string* is a separate
% type-tracking gap.)
fprintf('len %.0f\n', length('hello'));
fprintf('numel %.0f\n', numel('hello'));
fprintf('ndims %.0f\n', ndims('hello'));
fprintf('empty %.0f %.0f\n', isempty(''), isempty('x'));
s = 'world';
fprintf('var %.0f\n', length(s));
