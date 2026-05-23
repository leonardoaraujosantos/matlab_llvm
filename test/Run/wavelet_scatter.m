% scattering features + EMD additivity + matching pursuit
x = sin(2*pi*(0:255)/8) + 0.5*sin(2*pi*(0:255)/32);
feat = waveletScattering(x);
fprintf('scatter feat len: %.0f\n', length(feat));
imf = emd(x, 4);
s = sum(imf, 1);
fprintf('emd additive: %.0f\n', round(max(abs(x - s)) * 1e6));
D = [1 0 0.6; 0 1 0.6; 0 0 0.5];
y = 2*D(:,1) + 3*D(:,2);
c = matchingPursuit(D, y, 3);
fprintf('omp recovers a1: %.0f\n', round(c(1)));
