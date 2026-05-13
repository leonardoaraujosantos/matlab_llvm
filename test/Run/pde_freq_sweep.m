% pde_freq_sweep.m — structuralFrequency sweep on a clamped beam.
%
% Sweeps three ω values; reports peak displacement at each one.
% The system is real symmetric indefinite (negative eigenvalues near
% resonance) and goes through MINRES.

gm = pde_mesh_cuboid_tet(0.3, 0.03, 0.03, 6, 2, 2);

model = femodel('AnalysisType', 'structuralFrequency', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);            % clamp x=0
model = pde_set_face_pressure(model, 2, 1.0e5);     % 100 kPa pressure

% 3 well-separated frequencies, each well below the first resonance
% of the beam (≈ kHz range).
freqs = zeros(3, 1);
freqs(1) = 1.0;
freqs(2) = 10.0;
freqs(3) = 100.0;
model = pde_set_freq_list(model, freqs);
model = pde_generate_mesh(model);

raw = pde_solve(model);

% The kernel result struct holds .Uhist (3N × Nf) — we use a helper
% read here so we don't have to fight the type inference.
Uh = pde_kernel_uhist(raw);

% Peak displacement at each frequency.
nf = size(Uh, 2);
ndof = size(Uh, 1);
peaks = zeros(nf, 1);
for k = 1:nf
    p = 0.0;
    for i = 1:ndof
        v = Uh(i, k);
        if v < 0
            v = -v;
        end
        if v > p
            p = v;
        end
    end
    peaks(k) = p;
end

fprintf('PDE structuralFrequency: 0.3m beam, sweep\n');
fprintf('  num frequencies:      %.0f\n', nf);
fprintf('  log10(peak @ omega1): %.0f\n', floor(log10(peaks(1))));
fprintf('  log10(peak @ omega2): %.0f\n', floor(log10(peaks(2))));
fprintf('  log10(peak @ omega3): %.0f\n', floor(log10(peaks(3))));
