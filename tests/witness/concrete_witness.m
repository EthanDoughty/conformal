% Concrete dimension mismatch — trivial witness (no symbolic vars needed).
A = zeros(3, 4);
B = zeros(5, 2);
C = A * B;
% EXPECT: warnings = 1
