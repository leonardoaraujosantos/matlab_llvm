% Function-only sibling; the multi-file pre-load picks this up so
% calls to helper_fn() from sibling .m files compile and run, and
% breakpoints set on this file's lines fire correctly.
function r = helper_fn(x)
    intermediate = x * 3;
    r = intermediate + 1;
end
