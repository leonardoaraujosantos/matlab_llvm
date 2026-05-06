function out_signal = mealy_fsm(input_bit, reset)
    %#codegen
    % hdl: port(input_bit, bool)
    % hdl: port(reset, bool)
    
    % Definição dos estados
    S0 = uint8(0); 
    S1 = uint8(1);
    
    persistent current_state;
    
    if isempty(current_state) || reset
        current_state = S0;
    end

    % Inicializa a saída
    out_signal = false;

    % Lógica de Transição e Saída (Mealy)
    switch current_state
        case S0
            if input_bit == 1
                current_state = S1;
                out_signal = false; % Saída depende do estado E da entrada
            else
                current_state = S0;
                out_signal = false;
            end
            
        case S1
            if input_bit == 1
                current_state = S1;
                out_signal = true;  % Detectou a sequência "11" no mesmo ciclo!
            else
                current_state = S0;
                out_signal = false;
            end
            
        otherwise
            current_state = S0;
            out_signal = false;
    end
end

