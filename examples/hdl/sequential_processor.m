function [y, ovfl] = sequential_processor(x, gain, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % hdl: port(reset, bool)
    % cocotb: stimulus(x, impulse, 1.0)
    % cocotb: stimulus(gain, constant, 0.25)
    % cocotb: stimulus(reset, constant, 0)
    % cocotb: latency(4)

    % --- Configurações de Ponto Fixo (Explícitas) ---
    % Entradas: x (16 bits, 14 frac), gain (16 bits, 12 frac)
    % Coeficientes internos: 16 bits, 15 frac
    h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15);
    
    % --- Registradores (Memória do Circuito) ---
    persistent delay_line;
    if isempty(delay_line) || reset
        delay_line = fi(zeros(1, 4), 1, 16, 14);
    end
    
    % --- Lógica Sequencial: Deslocamento ---
    % No hardware, isso vira uma cadeia de 4 Flip-Flops de 16 bits
    delay_line = [fi(x, 1, 16, 14), delay_line(1:3)];
    
    % --- Lógica Combinacional: MAC (Multiply-Accumulate) ---
    % Acumulador de 36 bits para garantir precisão e evitar overflow intermediário
    acc = fi(0, 1, 36, 29); 
    
    for i = 1:4
        % Multiplicação: (16 bits, 14 frac) * (16 bits, 15 frac) = (32 bits, 29 frac)
        prod = delay_line(i) * h(i);
        acc(:) = acc + prod;
    end
    
    % --- Estágio Final: Ganho e Saturação ---
    % Multiplica o resultado do filtro pelo ganho externo
    % (36 bits) * (16 bits) = 52 bits internos
    full_res = acc * fi(gain, 1, 16, 12);
    
    % Cast final para saída de 16 bits com proteção de overflow
    % O HDL Coder gerará lógica de saturação se configurado
    y = fi(full_res, 1, 16, 12, 'OverflowAction', 'Saturate');
    ovfl = (full_res > 32767) || (full_res < -32768);
end

