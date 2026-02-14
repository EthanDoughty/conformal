% Test: Polymorphic cache with symbolic dimension arguments
% Cache key is (func_name, (arg_shapes...)), symbolic dims must match exactly
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x n]
% EXPECT: B = matrix[n x n]
% EXPECT: C = matrix[m x m]

function y = make_square(x)
    y = x * x;
end

A = make_square(zeros(n, n));
B = make_square(zeros(n, n));  % Cache hit (same symbolic n)
C = make_square(zeros(m, m));  % Cache miss (different symbolic m)
