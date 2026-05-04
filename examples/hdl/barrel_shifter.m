function y = barrel_shifter(x, amt)
    %#codegen
    % hdl: port(x, fi, unsigned, 16, 0)
    % hdl: port(amt, fi, unsigned, 8, 0)
    %
    % Logical-left barrel shifter via case decode on `amt`.
    % Synthesizes to a 16:1 mux tree. Exercises:
    %   - bitshift with constant left-shift amounts
    %   - case discriminator on an 8-bit port covering 16 of
    %     its 256 possible values (the rest pass through)
    %   - mutually-exclusive output assignments under a single
    %     switch

    switch amt
        case 0;  y = x;
        case 1;  y = bitshift(x, 1);
        case 2;  y = bitshift(x, 2);
        case 3;  y = bitshift(x, 3);
        case 4;  y = bitshift(x, 4);
        case 5;  y = bitshift(x, 5);
        case 6;  y = bitshift(x, 6);
        case 7;  y = bitshift(x, 7);
        case 8;  y = bitshift(x, 8);
        case 9;  y = bitshift(x, 9);
        case 10; y = bitshift(x, 10);
        case 11; y = bitshift(x, 11);
        case 12; y = bitshift(x, 12);
        case 13; y = bitshift(x, 13);
        case 14; y = bitshift(x, 14);
        case 15; y = bitshift(x, 15);
        otherwise; y = x;
    end
end
