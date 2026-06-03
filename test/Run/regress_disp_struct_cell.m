% regress_disp_struct_cell.m — regression test for disp of a struct or cell
% (#156). Before the fix, disp routed the struct/cell pointer to the
% polymorphic matlab_disp_mat path, which read it as a matrix descriptor
% (garbage rows/cols/data) and SIGSEGV'd (rc 139) on every backend. disp now
% dispatches a struct- or cell-bound argument to matlab_disp_struct /
% matlab_disp_cell, which print a field / element listing. The output format
% is not byte-exact to MATLAB, but it is deterministic and crash-free.

% --- struct with scalar fields -------------------------------------
s.a = 1;
s.b = 2;
disp(s);

% --- struct with a matrix field (shown as a size summary) ----------
t.v = [1 2 3];
t.n = 5;
disp(t);

% --- cell of scalars -----------------------------------------------
c = {1, 2, 3};
disp(c);

% --- cell with a matrix element ------------------------------------
d = {1, [1 2 3]};
disp(d);

% --- plain numeric disp is unaffected ------------------------------
disp(42);
disp([1 2 3]);
