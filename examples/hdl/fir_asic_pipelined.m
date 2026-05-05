function [y, ovfl] = fir_asic_pipelined(x, gain, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % hdl: port(reset, bool)
    % cocotb: stimulus(gain, constant, 0.25)
    % cocotb: stimulus(reset, constant, 0)
    % cocotb: latency(4)

    % Tipos de dados constantes
    h = fi([0.1, 0.2, 0.3, 0.4], 1, 16, 15);
    
    % --- REGISTRADORES (PERSISTENT) ---
    persistent delay_line;   % Estágio 0: Entradas
    persistent reg_products; % Estágio 1: Após Multiplicações
    persistent reg_acc;      % Estágio 2: Após Soma (Filtro)
    persistent reg_output;   % Estágio 3: Após Ganho e Saturação
    
    if isempty(delay_line) || reset
        delay_line = fi(zeros(1, 4), 1, 16, 14);
        reg_products = fi(zeros(1, 4), 1, 32, 29);
        reg_acc = fi(0, 1, 36, 29);
        reg_output = fi(0, 1, 16, 12);
    end
    
    % --- ESTÁGIO 1: Shift Register e Multiplicação ---
    delay_line = [fi(x, 1, 16, 14), delay_line(1:3)];
    for i = 1:4
        reg_products(i) = delay_line(i) * h(i); 
    end
    
    % --- ESTÁGIO 2: Acumulação (Soma) ---
    % O valor de 'acc' é guardado em um registrador para o próximo ciclo
    acc_temp = fi(0, 1, 36, 29);
    for i = 1:4
        acc_temp(:) = acc_temp + reg_products(i);
    end
    reg_acc = acc_temp;
    
    % --- ESTÁGIO 3: Ganho e Saturação ---
    full_res = reg_acc * fi(gain, 1, 16, 12);
    reg_output = fi(full_res, 1, 16, 12, 'OverflowAction', 'Saturate');
    
    % Saídas
    y = reg_output;
    ovfl = (full_res > 32767) || (full_res < -32768);
end

