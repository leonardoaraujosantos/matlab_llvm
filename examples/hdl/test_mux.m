% Entradas de teste
i0 = fi(10, 1, 16, 0);
i1 = fi(20, 1, 16, 0);
i2 = fi(30, 1, 16, 0);
i3 = fi(40, 1, 16, 0);
s  = uint8(2); % Seleciona a entrada 'in2'

% Execução
resultado = mux_4to1_16bit(i0, i1, i2, i3, s);

disp(['Saída do MUX: ', char(resultado.getval)]);

