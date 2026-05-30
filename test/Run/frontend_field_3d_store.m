% #78: 3-D indexed store/read through a struct field (the createMDP idiom).
mdp.T = zeros(3,3,2);
mdp.T(1,2,1) = 0.8;
mdp.T(1,3,1) = 0.2;
mdp.T(:,:,2) = ones(3,3);
fprintf('%.1f %.1f\n', mdp.T(1,2,1), mdp.T(1,3,1));
P = mdp.T(:,:,2);
fprintf('plane=%.0f\n', sum(P(:)));
