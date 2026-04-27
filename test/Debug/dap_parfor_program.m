% Fixture for parfor multi-thread DAP enumeration. Three pthread
% workers run the body concurrently; on first hook entry each one
% registers itself with the runtime's thread table. The DAP scenario
% verifies `threads` reports >1 entry after the worker dispatcher
% has fanned out.
parfor i = 1:3
    x = i + 1;
    disp(x);
end
disp('done');
