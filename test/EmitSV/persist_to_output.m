% B1/B2 fix — output port assigned the value of a persistent
% register. The runtime ABI returns f64 from `matlab_global_get_*`,
% but the slot's element type is the register's typed width
% (uint8 here), so LowerScalarSlots used to skip on type mismatch,
% leaving the body without an assignment to the output and the
% prelude `'0` as the only driver. After the fix the body assigns
% the typed register read directly to the output port.
function display = persist_to_output(reset)
    %#codegen
    % hdl: port(reset, bool)
    persistent counter;
    if isempty(counter)
        counter = uint8(0);
    end
    if reset
        counter = uint8(0);
    else
        counter = counter + uint8(1);
    end
    display = counter;
end
