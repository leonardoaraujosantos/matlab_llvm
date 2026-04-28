function [data_out, overflow] = alu_16bit(a, b, sel)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(sel, fi, unsigned, 8, 0)

    data_out = fi(0, 1, 16, 0);
    overflow = false;

    switch sel
        case 0 % Soma
            data_out = a + b;
            % Overflow soma: sinais iguais geram resultado com sinal diferente
            % (A e B positivos geram negativo OU A e B negativos geram positivo)
            if (a > 0 && b > 0 && data_out <= 0) || (a < 0 && b < 0 && data_out >= 0)
                overflow = true;
            end

        case 1 % Subtração
            data_out = a - b;
            % Overflow subtração: sinais opostos geram resultado com sinal inesperado
            if (a > 0 && b < 0 && data_out <= 0) || (a < 0 && b > 0 && data_out >= 0)
                overflow = true;
            end

        case 2 % AND
            data_out = bitand(a, b);
        case 3 % OR
            data_out = bitor(a, b);
        case 4 % XOR
            data_out = bitxor(a, b);
        case 5 % Not A
            data_out = bitcmp(a);
        otherwise
            data_out = fi(0, 1, 16, 0);
    end
end
