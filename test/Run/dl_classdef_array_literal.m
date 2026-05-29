% dl_classdef_array_literal — `[obj1; obj2; obj3]` literal syntax for
% classdef object arrays.  The lowering pass detects all-classdef
% vertcat literals and routes through matlab_dlnet_oa_new +
% matlab_dlnet_oa_append (the runtime carrier shipped in F), producing
% the same object-array as the explicit `objArrayNew/Append/Get` API
% but with the MATLAB-idiomatic `[A; B; C]` literal.

dlreset();
A = dlarray(zeros(2, 2));
B = dlarray(ones(2, 2));
C = dlarray([1 2; 3 4]);

% Build an object array via the literal — H item.
arr = [A; B; C];

n = objArrayLen(arr);
fprintf('dl_classdef_array_literal: length = %.0f\n', n);

% Extract each obj back and verify the original Data is preserved.
obj1 = objArrayGet(arr, 1);
obj2 = objArrayGet(arr, 2);
obj3 = objArrayGet(arr, 3);
D1 = extractdata(obj1);
D2 = extractdata(obj2);
D3 = extractdata(obj3);
fprintf('dl_classdef_array_literal: arr(1) = [%.0f %.0f; %.0f %.0f]\n', ...
        D1(1, 1), D1(1, 2), D1(2, 1), D1(2, 2));
fprintf('dl_classdef_array_literal: arr(2) = [%.0f %.0f; %.0f %.0f]\n', ...
        D2(1, 1), D2(1, 2), D2(2, 1), D2(2, 2));
fprintf('dl_classdef_array_literal: arr(3) = [%.0f %.0f; %.0f %.0f]\n', ...
        D3(1, 1), D3(1, 2), D3(2, 1), D3(2, 2));

if n == 3 && D1(1, 1) == 0 && D2(1, 1) == 1 && D3(1, 1) == 1 && D3(2, 2) == 4
    fprintf('dl_classdef_array_literal: PASS\n');
else
    fprintf('dl_classdef_array_literal: FAIL\n');
end
