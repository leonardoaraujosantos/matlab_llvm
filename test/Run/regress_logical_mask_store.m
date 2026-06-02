% regress_logical_mask_store.m — regression test for logical-mask indexed
% assignment, `v(mask) = x` (#144). Before the fix, the mask matrix (0/1
% values, same shape as v) was passed to matlab_slice_store1[_scalar] which
% interpreted its entries as literal 1-based linear indices: a mask like
% [0 0 1 1] became indices {-1,-1,0,0}, so the whole assignment collapsed
% onto element 1 (e.g. v(v>2)=0 gave [0 2 3 4] instead of [1 2 0 0]). The
% store now mirrors the read path's same-shape-as-A heuristic and treats a
% same-shape index as a logical mask. Reduced to scalars via sum() so the
% output is backend-formatting-independent.

% --- scalar RHS, comparison mask -----------------------------------
v = [1 2 3 4];
v(v > 2) = 0;
disp(sum(v));         % 1+2+0+0 = 3   (was 0+2+3+4 = 9)

% --- scalar RHS, other comparison ----------------------------------
w = [1 2 3 4];
w(w < 3) = 9;
disp(sum(w));         % 9+9+3+4 = 25  (was 9+2+3+4 = 18)

% --- column vector --------------------------------------------------
c = [1; 2; 3; 4];
c(c > 2) = 0;
disp(sum(c));         % 3

% --- vector RHS spread across mask positions -----------------------
u = [1 2 3 4];
u([1 0 1 0] > 0) = [10 30];
disp(sum(u));         % 10+2+30+4 = 46

% --- numeric (non-mask) index assignment is unaffected -------------
n = [1 2 3 4];
n([2 3]) = 0;
disp(sum(n));         % 1+0+0+4 = 5

% --- mask that selects nothing leaves v intact ---------------------
z = [1 2 3 4];
z(z > 99) = 0;
disp(sum(z));         % 10
