function [out_signal, state_display] = moore_fsm(input_bit, reset)
    % #codegen
    
    % Definição dos estados usando tipos enumerados ou inteiros
    % Usar uint8 é eficiente para o HDL Coder
    S0 = uint8(0); % Estado Inicial
    S1 = uint8(1); % Estado Processando
    S2 = uint8(2); % Estado Concluído
    
    % Variável persistente para armazenar o estado atual (Registrador)
    persistent current_state;
    
    if isempty(current_state) || reset
        current_state = S0;
    end
    
    % --- Lógica de Transição de Estado (Combinacional) ---
    switch current_state
        case S0
            if input_bit == 1
                current_state = S1;
            end
        case S1
            if input_bit == 0
                current_state = S2;
            else
                current_state = S0;
            end
        case S2
            if input_bit == 1
                current_state = S1;
            else
                current_state = S0;
            end
        otherwise
            current_state = S0;
    end
    
    % --- Lógica de Saída de Moore (Depende APENAS do estado) ---
    % Em Moore, a saída é decodificada diretamente do registrador de estado
    if current_state == S2
        out_signal = true;
    else
        out_signal = false;
    end
    
    state_display = current_state;
end

