% Issue #23 regression: `for n = M` where M is a 1-D static-shape
% vector must lower both at script top-level and inside user function
% bodies.  Pre-fix, LowerSeqLoops's extractRange returned false (M
% isn't a matlab.range), the matlab.for survived, and the LLVM lowering
% rejected it with "additional unconverted matlab.* ops left in IR".

% --- Script-top-level form -------------------------------------------------
sizes_top = [10, 20, 30];
s_top = 0;
for n = sizes_top
    s_top = s_top + n;
end
fprintf('top  s=%.0f\n', s_top);

% --- Function-body form (the documented blocker) --------------------------
fprintf('fn   s=%.0f\n', accVector());

% --- Function-body form using literal-matrix iterator ---------------------
fprintf('lit  s=%.0f\n', accLiteral());

function r = accVector()
    sizes = [128, 256, 512];
    r = 0;
    for n = sizes
        r = r + n;
    end
end

function r = accLiteral()
    r = 0;
    for n = [1, 2, 3, 4, 5]
        r = r + n;
    end
end
