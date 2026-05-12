% Catalog antenna design + peak-gain lookup, plumbed into a link
% budget.  Mirrors the MathWorks `design(antenna, freq)` +
% `antennaGain(antenna, freq)` workflow.

% Half-wave dipole at 2.4 GHz: λ/2 ≈ 6.25 cm; peak gain ≈ 2.15 dBi.
d = design(AntDipole(), 2.4e9);
disp(d.Length);
disp(antennaGain(d, 2.4e9));

% Quarter-wave monopole at 900 MHz: λ/4 ≈ 8.33 cm; peak gain ≈ 5.15 dBi.
m = design(AntMonopole(), 900e6);
disp(m.Height);
disp(antennaGain(m, 900e6));

% Plug the antenna gain into a site's link budget.
tx = TxSite('Latitude', 42.3, 'Longitude', -71.35, ...
            'AntennaHeight', 10, ...
            'TransmitterFrequency', 2.4e9, ...
            'TransmitterPower', 1, ...
            'AntennaGain', antennaGain(d, 2.4e9));
rx = RxSite('Latitude', 42.35, 'Longitude', -71.10);
pm = PropagationModel('freespace');
disp(sigstrength(rx, tx, pm));
