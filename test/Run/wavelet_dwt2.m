% 2-D wavelet transform round-trip on an image
A = magic(16);
[cA, cH, cV, cD] = dwt2(A, 'db2');
Ar = idwt2(cA, cH, cV, cD, 'db2');
fprintf('dwt2 subband: %.0f x %.0f\n', size(cA,1), size(cA,2));
fprintf('dwt2 PR ok: %.0f\n', round(max(max(abs(A - Ar))) * 1e6));
[C, S] = wavedec2(A, 2, 'db2');
Ar2 = waverec2(C, S, 'db2');
fprintf('wavedec2 PR ok: %.0f\n', round(max(max(abs(A - Ar2))) * 1e6));
