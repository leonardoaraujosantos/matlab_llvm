% Regression for #332: a script that calls helpers defined in *separate
% sibling .m files* must resolve on the AOT -emit-* path (IDE Compile/Run),
% the same way the -dap Debug launch already does. Before the fix the emit
% lanes failed with "undefined name" for every sibling helper.
%
% sib332_a is called directly; it in turn calls sib332_b, which the entry
% script never names — so this also exercises the transitive (fixpoint)
% sibling merge, not just first-level references.
r = sib332_a(6);
disp(r)
