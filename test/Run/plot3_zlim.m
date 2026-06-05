% Regression for #238: zlim must resolve on the AOT (-emit-llvm) path, the same
% as xlim/ylim. Also exercises the 6-arg axis([...zmin zmax]) form.
figure(1);
plot3([0 1 2], [0 1 0], [0 0.5 1]);
xlim([-2 2]);
ylim([-2 2]);
zlim([-2 6]);
axis([-2 2 -2 2 0 5]);
saveas(gcf, '/tmp/plot3_zlim.png');
disp('zlim ok');
