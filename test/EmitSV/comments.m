% Phase 5.6.2b SV — source `% ...` comments are forwarded to the
% emitted SV as `// ...` lines.
function y = comments(a, b)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(y, fi, signed, 16, 0)

    % Sum the two inputs to produce the result.
    y = a + b;
end
