% Phase 5.6 closure SV — full pipelined FIR ASIC processor.
%
% Verbatim copy of `examples/hdl/fir_asic_pipelined.m` with
% `% hdl: port(...)` pragmas at the top so the function compiles
% standalone (no driver). Closes the last remaining
% Stage F sub-pattern: `subscript_store(persistent_arr, i, val)`
% — i.e. `reg_products(i) = delay_line(i) * h(i)` writing back
% INTO a persistent fi-array slot from inside an unrolled
% for-loop body, without an enclosing `_persistent_set_ptr`
% follow-up. Lowers to a per-element `_global_set_f64(idx_k,
% val)` directly on the synthetic per-element register.
%
% Module shape:
%   - delay_line:   1×4 i16 persistent shift register
%   - reg_products: 1×4 i32 persistent (Stage F sub-stage write)
%   - reg_acc:      i64 (36-bit fi) scalar persistent
%   - reg_output:   i16 scalar persistent
%
% The synthetic per-element ids use a 1000-base offset so they
% don't collide with the user's original scalar persistent
% indices (`reg_acc`/`reg_output` at idx 2/3 would otherwise
% alias with delay_line[2]/delay_line[3] under the old
% `Idx*100+k` scheme).
function [y, ovfl] = fir_asic_pipelined(x, gain, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 14)
    % hdl: port(gain, fi, signed, 16, 12)
    % hdl: port(reset, bool)
    
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

