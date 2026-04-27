% Simulação de fluxo de bits
bits = [1 0 1 1 0 0 1];
reset = false;

for i = 1:length(bits)
    [detectado, estado] = moore_fsm(bits(i), reset);
    fprintf('Entrada: %d | Estado: %d | Saída: %d\n', bits(i), estado, detectado);
end

