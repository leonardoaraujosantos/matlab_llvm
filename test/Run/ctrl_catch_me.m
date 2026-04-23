% catch ME binds ME.message to the last error() argument.
try
    error('bad input');
catch ME
    disp(ME.message);
end

% No error -> catch body skipped, ME.message preserved from prior
% session isn't an issue here since the flag gates entry.
x = 0;
try
    x = 1;
catch ME
    disp(ME.message);
end
disp(x);

% Second error() overwrites the stored message.
try
    error('second failure');
catch ME
    disp(ME.message);
end
