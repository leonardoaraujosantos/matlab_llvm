% N-port S→Y conversion check.  Use the 2-port test_amp fixture and
% verify Y_11 = 0.01962 (= 0.9812/50 from the closed-form matrix
% algebra Y = (1/z0)·(I+S)⁻¹·(I−S)).

data = touchstoneRead("/Users/leonardoaraujo/work/matlab_llvm/test/Run/fixtures/rf/test_amp.s2p");
y = sparamS2yN(data);
disp(y.NumPorts);             % 2
disp(tsYij(y, 1, 1));         % 0.01962 (freq 1), 0.01636 (freq 2)
disp(tsYij(y, 2, 1));         % -0.07547 / -0.06061

% Same target via S→Z.  Z_11 = z0·[(I−S)⁻¹·(I+S)]_11 ≈ ?
z = sparamS2zN(data);
disp(tsZij(z, 1, 1));
