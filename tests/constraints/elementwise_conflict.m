% EXPECT: warnings = 1
A = rand(n, m);
B = rand(p, q);
C = A + B;
n = 4;
p = 6;  % EXPECT_WARNING: W_CONSTRAINT_CONFLICT
