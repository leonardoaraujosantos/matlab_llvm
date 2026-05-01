% Phase 5.2 — categorical. Auto-deduplication of category names with
% alphabetical sort, length / iscategory / disp dispatch through the
% typed runtime.

colors = categorical(["red", "green", "blue", "red", "green"]);
disp(colors);            % 5 lines: one per element
disp(length(colors));    % 5
disp(iscategory(colors, "red"));     % 1
disp(iscategory(colors, "purple"));  % 0

shapes = categorical(["circle", "square", "triangle", "circle"]);
disp(shapes);
disp(length(shapes));    % 4
