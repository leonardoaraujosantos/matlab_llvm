% Navigation Tier-4 — poseGraph square loop + optimizePoseGraph.
% Four [1 0 pi/2] edges form a unit square; the 4th closes the loop back to
% node 1.  The graph is already consistent, so optimization reproduces the
% exact square.  Print abs-based quantities so the golden is sign-stable
% across libstdc++/libc++ (near-zero coords can carry either sign).
pg = poseGraph();
addRelativePose(pg, [1 0 pi/2]);
addRelativePose(pg, [1 0 pi/2]);
addRelativePose(pg, [1 0 pi/2]);
addRelativePose(pg, [1 0 pi/2], 4, 1);   % loop closure back to node 1
nodes = optimizePoseGraph(pg);
fprintf('nodes=%.0f\n', size(nodes,1));
fprintf('n2=(%.2f,%.2f) n3=(%.2f,%.2f)\n', nodes(2,1), nodes(2,2), nodes(3,1), nodes(3,2));
closeErr = abs(nodes(1,1)) + abs(nodes(1,2)) + abs(nodes(4,1)) + abs(nodes(4,2) - 1);
fprintf('square-closure-err=%.3f\n', closeErr);
