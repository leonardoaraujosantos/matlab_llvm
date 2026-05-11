% MathWorks "RF Propagation and Visualization" tutorial-style smoke,
% end-to-end: txsite + rxsite + propagationModel + pathloss +
% sigstrength + link + los.  Mirrors the page minus the ray-tracing
% (ray-traced model + siteviewer / show / building data are
% deferred).
%
% URL: https://www.mathworks.com/help/comm/ug/rf-propagation-and-
%      visualization.html

% --- Sites ---
tx = TxSite('Name', 'Apple Hill', ...
            'Latitude', 42.3001, ...
            'Longitude', -71.3504, ...
            'AntennaHeight', 10, ...
            'TransmitterFrequency', 2.5e9, ...
            'TransmitterPower', 5);

rx = RxSite('Name', 'Fenway Park', ...
            'Latitude', 42.3467, ...
            'Longitude', -71.0972);

% --- Geometry ---
d_m = link(tx, rx);
disp(d_m);                          % 21452 m (~21.5 km)
is_los = los(tx, rx);
disp(is_los);                        % 0 (Earth bulge dominates 10m + 1m antennas)

% --- Path-loss models ---
pm_fs = PropagationModel('freespace');
pl_fs = pathloss(pm_fs, rx, tx);
disp(pl_fs);                         % 127 dB

pm_lr = PropagationModel('longley-rice');
pl_lr = pathloss(pm_lr, rx, tx);
disp(pl_lr);                         % 131 dB

% --- Received signal strength ---
rxpwr_fs = sigstrength(rx, tx, pm_fs);
disp(rxpwr_fs);                      % -90 dBm  (FSPL @ 21.5 km)

rxpwr_lr = sigstrength(rx, tx, pm_lr);
disp(rxpwr_lr);                      % -94 dBm  (Longley-Rice)
