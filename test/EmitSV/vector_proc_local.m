% Phase 4.5.5 (workaround) — combine scalar-args (4.5.5 limitation)
% with local fi-arrays (4.5.4 v1) so the function body operates on
% array-shaped data while the function boundary stays scalar. This
% pattern lets the user keep the array-style code shape inside
% the body even though vector function args aren't supported yet.
T = numerictype(1, 16, 8);
[m, d] = vector_proc_local(fi(1, T), fi(2, T), fi(3, T), fi(4, T), fi(5, T), fi(6, T));
disp(m);

function [mag_sq, dot_prod] = vector_proc_local(a1, a2, a3, b1, b2, b3)
    %#codegen
    a = fi(zeros(1, 3), 1, 16, 8);
    b = fi(zeros(1, 3), 1, 16, 8);
    a(1) = a1; a(2) = a2; a(3) = a3;
    b(1) = b1; b(2) = b2; b(3) = b3;

    dot_prod = a(1) * b(1) + a(2) * b(2) + a(3) * b(3);
    mag_sq   = a(1) * a(1) + a(2) * a(2) + a(3) * a(3);
end
