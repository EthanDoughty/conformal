% EXPECT: warnings = 0
function R = f(x)
    A = rand(x, x);
    B = rand(p, q);
    C = [A; B];
    R = C;
end
D = f(5);
p = 3;
q = 7;
