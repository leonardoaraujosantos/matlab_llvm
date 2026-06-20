% #335 Tier A: gpuArray is a routable device-resident value (a magic-tagged
% carrier), not an identity builtin. Each op on it dispatches through a hook
% that, with no device backend linked, falls back to the host CPU and stays
% numerically correct. gather() unwraps to host. The tag round-trips through
% the workspace and is recognized per-op (verified interpreted via -repl too).
A = [1 2; 3 4];
B = [5 6; 7 8];
Ag = gpuArray(A);
Bg = gpuArray(B);

disp(gather(Ag * Bg))      % mtimes:      [19 22; 43 50]
disp(gather(Ag .* Bg))     % elementwise: [5 12; 21 32]
disp(gather(Ag + Bg))      % add:         [6 8; 10 12]
disp(gather(Ag - Bg))      % sub:         [-4 -4; -4 -4]
disp(gather(Ag .* 3))      % gpuArray .* scalar -> gpuArray: [3 6; 9 12]

fprintf('sum_all = %g\n', gather(sum(Ag, 'all')));   % 10
fprintf('numel   = %g\n', numel(Ag));                % 4 (inspection sees through)
