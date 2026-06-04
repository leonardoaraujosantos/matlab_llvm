% #216: matrix * complex-scalar must stay complex (the product used to drop the
% imaginary part because matmul_mm read the matlab_mat_c as a real matrix).
x = [1 2 3];
y = x * complex(0.5, 0.5);          % [0.5+0.5i 1+1i 1.5+1.5i]
fprintf('mxs: re=%.1f im=%.1f\n', sum(real(y)), sum(imag(y)));
z = complex(2, -1) * x;             % scalar * matrix -> [2-1i 4-2i 6-3i]
fprintf('sxm: re=%.1f im=%.1f\n', sum(real(z)), sum(imag(z)));
r = [1 2 3] * [1; 1; 1];            % real matmul unaffected
fprintf('real: %.1f\n', r);
