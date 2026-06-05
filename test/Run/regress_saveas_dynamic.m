% Regression for #239: saveas must honour a runtime string filename (a
% variable / sprintf result), not only a string literal. Before the fix the
% plot lowering only handled a constant-folded matlab.const_char path and
% silently dropped a dynamic filename, writing no file — so per-frame export
% (saveas(gcf, sprintf('frame_%03d.png', k))) was a silent no-op. The fopen
% read-back below reports MISSING (fid < 0) if no file landed.
figure(1);
plot([0 1], [0 1]);
fn = sprintf('/tmp/regress_saveas_dyn_%03d.png', 7);
saveas(gcf, fn);
fid = fopen(fn, 'r');
if fid < 0
  disp('SAVEAS_DYNAMIC_MISSING')
else
  disp('SAVEAS_DYNAMIC_WRITTEN')
  fclose(fid);
end
