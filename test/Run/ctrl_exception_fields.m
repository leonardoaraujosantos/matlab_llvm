% Caught MException carries identifier + formatted message (#405).
% error(id, fmt, args) sets ME.identifier and a printf-formatted ME.message;
% plain error(msg) sets the message with an empty identifier.
try
    error('myPkg:bad', 'value was %d', 42);
catch ME
    fprintf('id=%s\n', ME.identifier);
    fprintf('msg=%s\n', ME.message);
end

try
    error('plain boom');
catch ME
    fprintf('id2=[%s]\n', ME.identifier);
    fprintf('msg2=%s\n', ME.message);
end

fprintf('done\n');
