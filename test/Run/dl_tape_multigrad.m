% dl_tape_multigrad.m — multiple dlgradient calls on one tape, across
% dlreset() iterations, must stay correct.  Guards the #82 tape-memory
% fix: grad() now hands each prior adjoint set to the tape's free list
% before re-seeding (orphaned-adjoint leak), and dlreset() frees the
% adjoints / backward temporaries.  Gradients must be invariant across
% resets and unaffected by the second grad call reusing the tape.
for k = 1:3
    dlreset();
    A = dlarray([1 2; 3 4]);
    B = dlarray([1 1; 1 1]);
    Y = A .* B;
    L = sum(Y);
    gA = dlgradient(L, A);   % dL/dA = B
    gB = dlgradient(L, B);   % dL/dB = A  (second grad on the same tape)
    fprintf('iter%d gA=%.0f,%.0f gB=%.0f,%.0f\n', ...
            k, gA(1,1), gA(2,2), gB(1,1), gB(2,2));
end
