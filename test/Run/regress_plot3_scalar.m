% Regression for #237: plot3 with scalar coordinates must lower instead of
% failing MLIR verification with 'f64' != '!llvm.ptr'. The scalar f64
% operands are boxed into 1x1 matrices (matlab_mat_from_scalar), matching
% MATLAB — plot3(0.5,0.5,0.5,'ko') draws a single 3-D marker — and matching
% the 2-D plot family, which already accepts scalar coordinates. Before the
% fix this aborted the JIT/REPL and could segfault inside a larger script.
figure(1);
plot3(0.5, 0.5, 0.5, 'ko');        % 4-arg scalar form with linespec
disp('PLOT3_FMT_OK')
plot3(1, 2, 3);                    % 3-arg scalar form, no linespec
disp('PLOT3_NOFMT_OK')
plot3([0 1 2], [0 1 0], [0 0.5 1]); % vector form still lowers unchanged
disp('PLOT3_VEC_OK')
saveas(gcf, '/tmp/regress_plot3_scalar.png');
disp('SAVEAS_OK')
