% instrument_class_tcpclient.m — TCP client networking object.
%
% Auto-prepended when the user input mentions `tcpclient` (a bare class name,
% like `tf`; no parser fold needed). One class per file so a program that uses
% only one networking class does not drag in the others' identically-named
% methods (which would otherwise mis-type and break the C++ emit lane).
%
% Call the methods with DOT syntax — `c.write(data)`, `r = c.read(n)` — the
% same convention as the sim3d objects; the bare form `read(c, n)` is not
% dispatched to a classdef method by this frontend. Socket state lives in the
% C++ runtime (runtime/toolbox/instrument/runtime_instrument.cpp); no MATLAB-
% side properties are stored (the runtime owns address/port/socket). Tier-1
% semantics: non-blocking with a bounded read timeout; payloads are raw float64
% (one matrix element per 8 bytes); one handle per thread.

classdef tcpclient < handle
    methods
        function obj = tcpclient(address, port)
            matlab_tcpclient_new(obj, address, port);
        end
        function write(obj, data)
            matlab_net_write(obj, data);
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
