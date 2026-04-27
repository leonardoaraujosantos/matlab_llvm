function count = counter_0_to_10(reset)
    % #codegen
    
    % 'persistent' define variáveis que mantêm o valor entre chamadas (Registradores)
    persistent count_reg;
    
    % Inicialização do registrador (ocorre no reset do hardware)
    if isempty(count_reg)
        count_reg = fi(0, 0, 4, 0); % 4 bits são suficientes para contar até 10
    end
    
    % Lógica de Reset e Contagem
    if reset
        count_reg = fi(0, 0, 4, 0);
    else
        if count_reg >= 10
            count_reg = fi(0, 0, 4, 0); % Reinicia ao chegar em 10
        else
            count_reg = count_reg + 1;  % Incrementa
        end
    end
    
    % Atribui o valor do registrador à saída
    count = count_reg;
end

