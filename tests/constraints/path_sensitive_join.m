% EXPECT: warnings = 1
A = rand(n, m);
D = rand(p, q);
if n > 5
    E = [A; D];
else
    F = [A; D];
end
q = 3;
m = 7;
