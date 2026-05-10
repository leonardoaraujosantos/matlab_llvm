% Control System Toolbox stdlib classdefs (§3.1).
%
% Auto-prepended by matlabc when the user input mentions any of
% `tf` / `ss` / `zpk` / `pid` / `frd` as a call target or LHS — see
% `findCstPrelude` and `userMentionsCstClass` in
% `tools/matlabc/main.cpp`.
%
% Slice 2 surface: tf-vs-tf operator overloads (`G + H`, `G - H`,
% `G * H`, `G / H`, `-G`). Scalar mixing (`s + 2`-style, `tf('s')`
% builder) needs Sema-level CST property type tracking and is the
% next slice; today scalar mixing falls through to the existing
% generic operator dispatch which doesn't know about tf semantics.

classdef tf
    properties
        Numerator
        Denominator
    end
    methods
        function obj = tf(num, den)
            if nargin == 2
                obj.Numerator = num;
                obj.Denominator = den;
            end
        end

        function r = plus(a, b)
            % Parallel: (a.num*b.den + b.num*a.den) / (a.den*b.den).
            cross_a = conv(a.Numerator, b.Denominator);
            cross_b = conv(b.Numerator, a.Denominator);
            new_num = cst_polyadd(cross_a, cross_b);
            new_den = conv(a.Denominator, b.Denominator);
            r = tf(new_num, new_den);
        end

        function r = minus(a, b)
            cross_a = conv(a.Numerator, b.Denominator);
            cross_b = conv(b.Numerator, a.Denominator);
            new_num = cst_polysub(cross_a, cross_b);
            new_den = conv(a.Denominator, b.Denominator);
            r = tf(new_num, new_den);
        end

        function r = uminus(a)
            r = tf(-a.Numerator, a.Denominator);
        end

        function r = mtimes(a, b)
            % Series cascade.
            new_num = conv(a.Numerator, b.Numerator);
            new_den = conv(a.Denominator, b.Denominator);
            r = tf(new_num, new_den);
        end

        function r = mrdivide(a, b)
            new_num = conv(a.Numerator, b.Denominator);
            new_den = conv(a.Denominator, b.Numerator);
            r = tf(new_num, new_den);
        end
    end
end

function r = cst_polyadd(p, q)
    np = length(p);
    nq = length(q);
    if np >= nq
        n = np;
    else
        n = nq;
    end
    r = zeros(1, n);
    op = n - np;
    oq = n - nq;
    for i = 1:np
        r(i + op) = r(i + op) + p(i);
    end
    for i = 1:nq
        r(i + oq) = r(i + oq) + q(i);
    end
    % `reshape` is on PtrRet so the function's return type settles to
    % matlab_mat *, which the constructor's `tf(num, den)` call expects
    % as an operand. Without this nudge Sema infers the return as
    % `tensor<1x?xf64>` (from `r = zeros(1, n)`) and the `r =
    % tf(new_num, ...)` matlab.call inside the operator method bodies
    % silently fails to convert — slot retyping doesn't propagate
    % through the inner-loop iterations of LowerUserCalls.
    r = reshape(r, 1, n);
end

function r = cst_polysub(p, q)
    np = length(p);
    nq = length(q);
    if np >= nq
        n = np;
    else
        n = nq;
    end
    r = zeros(1, n);
    op = n - np;
    oq = n - nq;
    for i = 1:np
        r(i + op) = r(i + op) + p(i);
    end
    for i = 1:nq
        r(i + oq) = r(i + oq) - q(i);
    end
    r = reshape(r, 1, n);
end
