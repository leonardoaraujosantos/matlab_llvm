% Constant-range loop whose counter is printed through fprintf.
s = 0;
for k = 1:10
    s = s + k;
    fprintf('k=%d partial=%d\n', k, s);
end
