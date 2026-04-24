% sub2ind / ind2sub / assert.
%
% Linear indexing here follows MATLAB's column-major convention
% even though our matlab_mat storage is row-major — the indexing
% model stays MATLAB-compatible at the API boundary.

shp = [3 4];

disp(sub2ind(shp, 1, 1));    % 1
disp(sub2ind(shp, 2, 1));    % 2
disp(sub2ind(shp, 1, 2));    % 4  — column-major
disp(sub2ind(shp, 3, 4));    % 12

disp(ind2sub(shp, 1));       % [1 1]
disp(ind2sub(shp, 4));       % [1 2]
disp(ind2sub(shp, 12));      % [3 4]

% assert: true silently succeeds, false sets the error flag.
assert(1 == 1);
disp(1);                     % 1

try
    assert(1 == 2, "expected 1 to equal 2");
catch ME
    disp(ME.message);        % expected 1 to equal 2
end
