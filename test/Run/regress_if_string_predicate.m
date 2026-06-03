% regress_if_string_predicate.m — regression test for using an
% F64-returning string predicate (contains / startsWith / endsWith /
% strcmp / ...) directly as an if/while condition (#148). These builtins
% are `none`-typed at MIR-to-MLIR lowering, so fixupIfCond left a
% verifier-placeholder unrealized_conversion_cast on the scf.if. Their
% result only refines to f64 in the LowerTensorOps loop, and no pass
% afterwards resolved the placeholder on the AOT / JIT lowering paths
% (only the SV-emit path ran RefineIfConds), so the cast survived to
% translateModuleToLLVMIR and failed. RefineIfConds now runs on those
% paths too.

% --- contains as a direct if-condition (true) ----------------------
if contains("hello world", "world")
  disp(1);                 % 1
end

% --- startsWith true / false branches ------------------------------
if startsWith("abcdef", "abc")
  disp(2);                 % 2
end
if startsWith("abcdef", "xyz")
  disp(99);
else
  disp(3);                 % 3
end

% --- endsWith ------------------------------------------------------
if endsWith("report.txt", ".txt")
  disp(4);                 % 4
end

% --- predicate result stored then used as a condition --------------
hit = contains("needle in haystack", "needle");
if hit
  disp(5);                 % 5
end

% --- predicate inside a while guard --------------------------------
s = "aaaa";
n = 0;
while contains(s, "a") && n < 3
  n = n + 1;
  s = "";   % drop the match so the loop also tests the false path
  if n == 1
    s = "a";
  end
end
disp(n);                   % 2
