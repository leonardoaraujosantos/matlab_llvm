function y = mux_4to1_16bit(in0, in1, in2, in3, sel)
    % #codegen
    % Define que o código é para geração de HDL
    
    % Inicializa a saída com o mesmo tipo das entradas
    y = fi(0, 1, 16, 0);

    % Lógica do Multiplexador
    switch sel
        case 0
            y = in0;
        case 1
            y = in1;
        case 2
            y = in2;
        case 3
            y = in3;
        otherwise
            y = in0; % Valor padrão (Safe State)
    end
end

