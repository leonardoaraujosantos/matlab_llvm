reset = false;
fprintf('Iniciando contagem:\n');

for i = 1:15
    if i == 12
        reset = true; % Testa o reset no meio do caminho
    else
        reset = false;
    end
    
    out = counter_0_to_10(reset);
    fprintf('Ciclo %d -> Valor: %d\n', i, int8(out));
end

