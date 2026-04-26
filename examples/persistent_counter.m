% `persistent` keeps a function-local variable alive across calls —
% it's MATLAB's equivalent of C's `static` locals. The variable is
% private to the declaring function (each function gets its own
% storage even if two functions both declare `persistent n`) and
% survives until process exit (or `clear` of the function).
disp(count());   % 1
disp(count());   % 2
disp(count());   % 3

function y = count()
    persistent n;
    n = n + 1;
    y = n;
end
