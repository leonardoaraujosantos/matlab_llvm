% Program with parfor — disp substitutions must keep the runtime call
% so the mutex still coordinates interleaved output between workers.
parfor i = 1:3
    disp(i);
end
