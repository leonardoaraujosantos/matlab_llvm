% Multi-file fixture entry point. Calls helper_fn() defined in
% dap_helper.m sitting alongside this script. Line numbers
% referenced from dap_scenarios.py.
disp('main start');
result = helper_fn(7);
disp(result);
