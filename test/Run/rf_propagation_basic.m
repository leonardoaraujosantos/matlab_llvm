% MathWorks "RF Propagation and Visualization" tutorial-style smoke
% test — the function-form algorithms (Longley-Rice, FSPL) are
% accessed through `TxSite` / `RxSite` / `PropagationModel` classdef
% wrappers via the kwarg sugar.  No viewer / ray-tracing.

tx = TxSite('Name', 'Apple Hill', ...
            'Latitude', 42.3001, ...
            'Longitude', -71.3504, ...
            'AntennaHeight', 10, ...
            'TransmitterFrequency', 2.5e9, ...
            'TransmitterPower', 5);

rx = RxSite('Name', 'Fenway Park', ...
            'Latitude', 42.3467, ...
            'Longitude', -71.0972);

% Free-space path loss between the two sites.
pm_fs = PropagationModel('freespace');
pl_fs = pathloss(pm_fs, rx, tx);
disp(pl_fs);                    % ~127 dB at 2.5 GHz / ~28 km

% Longley-Rice (ITM) with flat terrain (no profile).
pm_lr = PropagationModel('longley-rice');
pl_lr = pathloss(pm_lr, rx, tx);
disp(pl_lr);                    % ~132 dB (ITM diffraction/scatter losses)
