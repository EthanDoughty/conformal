% Demo file for testing conformal-action on a PR.
% Contains deliberate dimension mismatches.

A = zeros(3, 4);
B = zeros(5, 2);

% Inner dimension mismatch: 4 vs 5
C = A * B;

% Elementwise mismatch: 3x4 vs 5x2
D = A .* B;

% This one is fine
E = A';
F = E * A;
