% Plain scalar accumulation + conditional print.
acc = 0.0;
for i = 1:50
    acc = acc + 1.0 / i;
    if mod(i, 10) == 0
        fprintf('i=%d harmonic=%.6f\n', i, acc);
    end
end
