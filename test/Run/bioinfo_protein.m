% Bioinformatics Toolbox Tier-5 — protein property + digestion.
%   molweight, atomiccomp, isoelectric, aminolookup, cleave, restrict.
fprintf('MW(AG)=%.2f\n', molweight('AG'));     % 71.0788+57.0519+18.0152 = 146.15
fprintf('MW(W)=%.2f\n', molweight('W'));        % 186.2132 + 18.0152 = 204.23

a = atomiccomp('G');                            % Gly residue (2,3,1,1,0) + water
fprintf('Gly atoms C=%.0f H=%.0f N=%.0f O=%.0f S=%.0f\n', a.C, a.H, a.N, a.O, a.S);

disp(aminolookup('ACDEF'));                     % AlaCysAspGluPhe
disp(aminolookup('Cys'));                       % C

disp('--- trypsin digest ---');
disp(cleave('AAKAARPK', 'trypsin'));            % cut after K/R, not before P

disp('--- EcoRI digest ---');
disp(restrict('GGGAATTCGGG', 'EcoRI'));

fprintf('pI(acidic DDDDD) < pI(basic KKKKK): %.0f\n', ...
        isoelectric('DDDDD') < isoelectric('KKKKK'));
