% instrument_class_tcpserver.m — TCP server networking object.
%
% Auto-prepended when the user input mentions `tcpserver` (a bare class name).
% One class per file (see instrument_class_tcpclient.m for the rationale).
% Binds + listens on construction; accepts a single client lazily on the first
% read/write, so constructing a server does not block waiting for a peer. Call
% methods with DOT syntax (`s.read(n)`); socket state lives in the C++ runtime.

classdef tcpserver < handle
    methods
        function obj = tcpserver(address, port)
            matlab_tcpserver_new(obj, address, port);
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
