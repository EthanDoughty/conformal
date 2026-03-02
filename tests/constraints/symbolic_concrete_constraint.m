% EXPECT: warnings = 1
A = rand(n, 5);
B = rand(m, 3);
C = A * B;
m = 10;  % EXPECT_WARNING: W_CONSTRAINT_CONFLICT
