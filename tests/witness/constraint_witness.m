% Witness leverages scalar_bindings: n=3, m=5 are known, so dims 5 != 6 directly.
n = 3;
m = 5;
A = zeros(n, m);
B = zeros(m+1, n);
C = A * B;
% EXPECT: warnings = 1
