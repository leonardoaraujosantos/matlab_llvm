% exist(name, kind) status codes via the AOT pipeline (#404).
% Uses read-only paths present on Linux and macOS so the fixture is
% hermetic. The 'var' kind needs the live REPL workspace, so it is
% covered by test/Repl/run_tests.sh rather than here.
fprintf('file=%d\n', exist('/etc/passwd', 'file'));      % 2
fprintf('nofile=%d\n', exist('/no_such_file_404', 'file'));  % 0
fprintf('dir=%d\n', exist('/etc', 'dir'));               % 7
fprintf('nodir=%d\n', exist('/no_such_dir_404', 'dir'));     % 0
fprintf('nokind_dir=%d\n', exist('/etc'));               % 7 (no kind -> path)
