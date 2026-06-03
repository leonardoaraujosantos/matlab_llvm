% Regression: fprintf must collapse %% to a literal % even with no value args
% (MATLAB). Previously the no-arg path used fputs and printed %% verbatim. (#208)
fprintf('rate: 50%%\n');         % rate: 50%
fprintf('a%%b%%c\n');            % a%b%c
fprintf('100%%%%\n');            % 100%%  (two literal percents)
fprintf('%d done (%%)\n', 5);    % 5 done (%)   (value path still correct)
fprintf('no percent here\n');    % unchanged
