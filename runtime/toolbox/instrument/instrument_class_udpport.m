% instrument_class_udpport.m — UDP datagram networking object.
%
% Auto-prepended when the user input mentions `udpport` (a bare class name).
% One class per file (see instrument_class_tcpclient.m for the rationale).
% Connectionless: `u.write(data, address, port)` sends a datagram to an
% explicit destination, `u.read(n)` receives available bytes. Call methods with
% DOT syntax; socket state lives in the C++ runtime. Tier-1 payloads are raw
% float64 (one matrix element per 8 bytes); one handle per thread.

classdef udpport < handle
    methods
        function obj = udpport(localPort)
            matlab_udpport_new(obj, localPort);
        end
        function write(obj, data, address, port)
            matlab_udp_write_to(obj, data, address, port);
        end
        function data = read(obj, count)
            data = matlab_net_read(obj, count);
        end
        function writeline(obj, str)
            matlab_net_writeline(obj, str);
        end
        function str = readline(obj)
            str = matlab_net_readline(obj);
        end
        function flush(obj)
            matlab_net_flush(obj);
        end
    end
end
