bits = [0 1 1 0 1 1 1];
reset = false;

for i = 1:length(bits)
    detectado = mealy_fsm(bits(i), reset);
    fprintf('Bit: %d | Saída Mealy: %d\n', bits(i), detectado);
end

