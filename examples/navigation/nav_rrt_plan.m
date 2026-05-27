% Navigation Toolbox — "Plan Mobile Robot Paths Using RRT" (HEADLINE).
% Mirrors https://www.mathworks.com/help/nav/ug/plan-mobile-robot-paths-using-rrt.html
%
% The full Tier-1 -> Tier-2 path-planning arc end to end:
%   occupancyMap  ->  inflate  ->  stateSpaceSE2  ->  validatorOccupancyMap
%                 ->  plannerRRT  ->  plan  ->  shortenpath
%
% (The MathWorks page loads `office_area_gridmap.mat` and uses a
% `stateSpaceDubins`; we build the obstacle map programmatically and use the
% holonomic `stateSpaceSE2` so the example is self-contained.  The validator
% `.Map` property form is replaced by the 2-arg `validatorOccupancyMap(ss,map)`
% idiom — both documented in docs/navigation_toolbox_roadmap.md.)

rng(0);

% --- Build a 25 x 25 m office-like map at 1 cell/m with two wall segments.
W = 25; H = 25;
map = occupancyMap(W, H, 1);
for y = 5:18
    setOccupancy(map, [10 y], 1.0);     % vertical wall, gap at the top
end
for x = 12:22
    setOccupancy(map, [x 14] , 1.0);    % horizontal wall, gap at the left
end
inflate(map, 0.5);                       % robot-radius clearance

% --- State space + validator over the inflated map.
ss = stateSpaceSE2([0 W; 0 H; -pi pi]);
sv = validatorOccupancyMap(ss, map);
sv.ValidationDistance = 0.1;

% --- RRT planner.
planner = plannerRRT(ss, sv);
planner.MaxConnectionDistance = 3.0;
planner.MaxIterations = 20000;
planner.GoalBias = 0.1;

start = [2 2 0];
goal  = [23 23 0];

result = plan(planner, start, goal);
found  = result(1, 2);
nstate = result(1, 1);

fprintf('RRT planning: %d x %d m map, start (%.0f,%.0f) -> goal (%.0f,%.0f)\n', ...
        W, H, start(1), start(2), goal(1), goal(2));

if found < 0.5
    fprintf('No path found within %d iterations.\n', planner.MaxIterations);
else
    % Path states live in rows 2..N+1 (row 1 is [numStates exitflag numIters]).
    states = result(2:end, :);

    % Raw RRT path length.
    len = 0.0;
    for i = 1:(nstate - 1)
        dx = states(i+1, 1) - states(i, 1);
        dy = states(i+1, 2) - states(i, 2);
        len = len + sqrt(dx*dx + dy*dy);
    end

    % Greedy shortcut smoothing.
    np = navPath(states);
    short = shortenpath(np, sv);
    slen = 0.0;
    for i = 1:(size(short, 1) - 1)
        dx = short(i+1, 1) - short(i, 1);
        dy = short(i+1, 2) - short(i, 2);
        slen = slen + sqrt(dx*dx + dy*dy);
    end

    fprintf('Path found: %d states, length %.2f m\n', nstate, len);
    fprintf('Shortened : %d waypoints, length %.2f m\n', size(short, 1), slen);
    fprintf('Straight-line lower bound = %.2f m\n', ...
            sqrt((goal(1)-start(1))^2 + (goal(2)-start(2))^2));
end
