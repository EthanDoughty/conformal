% EXPECT: warnings = 0
n = 10;
if n > 0
    A = zeros(n, n);
end
% EXPECT: A = matrix[10 x 10]
