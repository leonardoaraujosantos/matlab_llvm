% csv_stats.m — load a CSV alongside this script as a heterogeneous
% MATLAB table and run summary stats on each numeric column.
%
% The dataset (csv_stats.csv) is 16 weather observations across four
% European cities sampled on four April Wednesdays in 2026. Columns:
%
%   date           datetime
%   city           string
%   temperature_c  numeric  (°C)
%   humidity_pct   numeric  (%)
%   rainfall_mm    numeric  (mm)
%   wind_kmh       numeric  (km/h)
%
% readtable auto-detects the delimiter (',') and infers a kind per
% column. The numeric columns flow through the regular matrix path,
% so mean / std / median / min / max work without conversion.
%
% Path convention: this script reads `csv_stats.csv` from the current
% working directory. The IDE's Run button sets cwd to the script's
% folder, so the relative path resolves there. From the repo root,
% invoke as:  cd examples && matlabc -repl < csv_stats.m

T = readtable("csv_stats.csv");

disp("=== Dataset shape =====================================");
disp("rows =");
disp(height(T));
disp("cols =");
disp(width(T));

disp("T =");
disp(T);

% --- Per-column summaries ------------------------------------------
%
% Pull each numeric column off the table once into a plain vector,
% then run the five canonical descriptive stats on it. Doing it
% column-by-column keeps the example readable and makes it obvious
% that table columns drop straight into the same reductions you'd
% use on a raw matrix.

temp = T.temperature_c;
hum  = T.humidity_pct;
rain = T.rainfall_mm;
wind = T.wind_kmh;

disp("=== temperature_c (°C) ================================");
disp("mean   ="); disp(mean(temp));
disp("std    ="); disp(std(temp));
disp("median ="); disp(median(temp));
disp("min    ="); disp(min(temp));
disp("max    ="); disp(max(temp));

disp("=== humidity_pct (%) ==================================");
disp("mean   ="); disp(mean(hum));
disp("std    ="); disp(std(hum));
disp("median ="); disp(median(hum));
disp("min    ="); disp(min(hum));
disp("max    ="); disp(max(hum));

disp("=== rainfall_mm (mm) ==================================");
disp("mean   ="); disp(mean(rain));
disp("std    ="); disp(std(rain));
disp("median ="); disp(median(rain));
disp("min    ="); disp(min(rain));
disp("max    ="); disp(max(rain));

disp("=== wind_kmh (km/h) ===================================");
disp("mean   ="); disp(mean(wind));
disp("std    ="); disp(std(wind));
disp("median ="); disp(median(wind));
disp("min    ="); disp(min(wind));
disp("max    ="); disp(max(wind));

% --- Aggregates ----------------------------------------------------
%
% Total rainfall over the sampling window plus the single hottest /
% coldest readings — useful as a "did the table load correctly" sanity
% check the user can eyeball against the source CSV.

disp("=== Aggregates =========================================");
disp("total rainfall (all rows) =");
disp(sum(rain));
disp("hottest reading =");
disp(max(temp));
disp("coldest reading =");
disp(min(temp));
disp("variance of temperature =");
disp(var(temp));
