% Test: Commutative dimensions join correctly
% EXPECT: warnings = 0
% EXPECT: X = matrix[(m+n) x 1]

n = 3;
m = 4;
if 1
    X = zeros(n + m, 1);
else
    X = zeros(m + n, 1);
end
