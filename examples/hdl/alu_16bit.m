function [data_out, overflow] = alu_16bit(a, b, sel)
    % #codegen
    % A diretiva #codegen é essencial para indicar compatibilidade com HDL
    
    % Inicializa a saída com o tipo de dado correto (16-bit assinado)
    data_out = fi(0, 1, 16, 0); 
    overflow = false;

    % Switch case para selecionar a operação
    switch sel
        case 0 % Soma
            data_out = a + b;
        case 1 % Subtração
            data_out = a - b;
        case 2 % AND bit a bit
            data_out = bitand(a, b);
        case 3 % OR bit a bit
            data_out = bitor(a, b);
        case 4 % XOR bit a bit
            data_out = bitxor(a, b);
        case 5 % Not A
            data_out = bitcmp(a);
        otherwise
            data_out = fi(0, 1, 16, 0);
    end
    
    % Verificação simples de overflow (opcional)
    % Nota: O objeto 'fi' gerencia o truncation/wrap automaticamente
end

