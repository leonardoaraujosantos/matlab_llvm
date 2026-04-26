% Fixture for the error()-backtrace scenario. The script calls fail()
% which calls deeper() which raises error(). When DebugMode is on, the
% runtime snapshots the frame stack inside matlab_set_error_msg before
% the implicit unwind pops frames off, then prints `error: <msg>` plus
% one `at <fn> (<file>:<line>)` line per frame to stderr.
%
% Line numbers are referenced from dap_scenarios.py.
disp('before');
fail();
disp('after');

function fail()
    deeper();
end

function deeper()
    error('boom');
end
