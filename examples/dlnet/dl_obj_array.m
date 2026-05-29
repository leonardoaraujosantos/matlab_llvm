% dl_obj_array — generic object-array carrier (F item, pragmatic
% alternative to the `[obj1; obj2; obj3]` classdef array literal
% which is gated on a multi-week Sema implementation).
%
% Provides the same capability surface — build a list of classdef
% instances, index into them, iterate — via an explicit constructor +
% append API rather than parser sugar.
%
% Demo: build an array of 3 dlarray instances with distinct payloads,
% verify length + per-index access return the expected matrices.

dlreset();
A = dlarray(zeros(2, 2));
B = dlarray(ones(2, 2));
C = dlarray([1 2; 3 4]);

arr = objArrayNew();
arr = objArrayAppend(arr, A);
arr = objArrayAppend(arr, B);
arr = objArrayAppend(arr, C);

n = objArrayLen(arr);
fprintf('dl_obj_array: length = %.0f\n', n);

% Round-trip the 3 stored objs back out, extract their data, verify.
obj1 = objArrayGet(arr, 1);
obj2 = objArrayGet(arr, 2);
obj3 = objArrayGet(arr, 3);
D1 = extractdata(obj1);
D2 = extractdata(obj2);
D3 = extractdata(obj3);
fprintf('dl_obj_array: arr(1) = [%.0f %.0f; %.0f %.0f]\n', ...
        D1(1, 1), D1(1, 2), D1(2, 1), D1(2, 2));
fprintf('dl_obj_array: arr(2) = [%.0f %.0f; %.0f %.0f]\n', ...
        D2(1, 1), D2(1, 2), D2(2, 1), D2(2, 2));
fprintf('dl_obj_array: arr(3) = [%.0f %.0f; %.0f %.0f]\n', ...
        D3(1, 1), D3(1, 2), D3(2, 1), D3(2, 2));

ok_len = (n == 3);
ok_a = (D1(1, 1) == 0) && (D1(2, 2) == 0);
ok_b = (D2(1, 1) == 1) && (D2(2, 2) == 1);
ok_c = (D3(1, 1) == 1) && (D3(2, 2) == 4);

if ok_len && ok_a && ok_b && ok_c
    fprintf('dl_obj_array: PASS\n');
else
    fprintf('dl_obj_array: FAIL\n');
end
