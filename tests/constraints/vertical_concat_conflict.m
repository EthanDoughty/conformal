% EXPECT: warnings = 1
% EXPECT: E = matrix[(n+p) x None]
A = rand(n, m);
D = rand(p, q);
E = [A; D];
q = 5;
m = 10;
