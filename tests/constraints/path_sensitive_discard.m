% EXPECT: warnings = 0
A = rand(n, m);
D = rand(p, q);
if n > 5
    E = [A; D];
else
    x = 1;
end
q = 3;
m = 7;
