% pde_freq_sweep_damped.m — damped harmonic response (K - ω²M + iωC) U = F.
%
% Same clamped beam as pde_freq_sweep but with Rayleigh damping
% (α = 0, β = 1e-4).  System becomes complex; solved via the
% 2N × 2N real bordered formulation + ILU(0)-preconditioned
% GMRES(30).  At ω well below the first resonance the response is
% ~quasi-static (small phase lag); peak displacement magnitude
% should stay bounded throughout the sweep.

gm = pde_mesh_cuboid_tet(0.3, 0.03, 0.03, 6, 2, 2);

model = femodel('AnalysisType', 'structuralFrequency', 'Geometry', gm);
model = pde_set_material(model, ...
            materialProperties('YoungsModulus', 2.0e11, ...
                               'PoissonsRatio', 0.30, ...
                               'MassDensity',   7850));
model = pde_set_face_fixed   (model, 5);
model = pde_set_face_pressure(model, 2, 1.0e5);

freqs = zeros(3, 1);
freqs(1) = 1.0;
freqs(2) = 10.0;
freqs(3) = 100.0;
model = pde_set_freq_list(model, freqs);
model = pde_set_rayleigh(model, 0.0, 1.0e-4);  % light material damping
model = pde_generate_mesh(model);

raw = pde_solve(model);
Uh  = pde_kernel_uhist(raw);

% |U| (peak magnitude) per frequency.
nf  = size(Uh, 2);
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

fprintf('PDE structuralFrequency (damped, Rayleigh):\n');
fprintf('  num frequencies:      %.0f\n', nf);
fprintf('  log10(|U| @ omega1):  %.0f\n', floor(log10(peaks(1))));
fprintf('  log10(|U| @ omega2):  %.0f\n', floor(log10(peaks(2))));
fprintf('  log10(|U| @ omega3):  %.0f\n', floor(log10(peaks(3))));
