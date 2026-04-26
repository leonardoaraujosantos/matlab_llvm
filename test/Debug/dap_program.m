% DAP scenario fixture. Line numbers are referenced by absolute index
% from dap_scenarios.py — do not reflow this file without updating
% the EXPECTED_* constants there.
x = 10;
y = 20;
z = x + y;

for i = 1:3
    z = z + i;
end

disp(z);
