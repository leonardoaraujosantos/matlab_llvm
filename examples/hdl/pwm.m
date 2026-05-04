function pwm_out = pwm(duty, reset)
    %#codegen
    % hdl: port(duty, fi, unsigned, 8, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 8-bit PWM generator. A persistent counter wraps around 0..255;
    % the output is high whenever the counter is below `duty`. So
    % duty=0 → always low, duty=255 → high 255/256 of the time.
    %
    % Tests:
    %   - persistent counter with natural wraparound (relies on i8
    %     unsigned wrap semantics)
    %   - register-vs-input compare driving a combinational output
    %     bit
    %   - the pattern: "register read, compare against input,
    %     output gated, register update for next cycle"

    persistent counter;
    if isempty(counter) || reset
        counter = uint8(0);
    end

    pwm_out = counter < duty;

    counter = counter + uint8(1);
end
