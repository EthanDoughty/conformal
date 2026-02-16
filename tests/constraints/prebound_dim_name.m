% EXPECT: warnings = 0
m = 5;
A = rand(n, m);
B = rand(p, 5);
C = [A; B];
m = 10;
