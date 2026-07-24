% Numeric-literal subscripts are provably double arrays (never a logical
% mask), so their extent folds to numel -- in either subscript position,
% and in combination.
% EXPECT: B = matrix[3 x 7]
% EXPECT: C = matrix[5 x 2]
% EXPECT: D = matrix[3 x 2]
% EXPECT: warnings = 0

A = randn(5, 7);
B = A([1 3 2], :);
C = A(:, [1 2]);
D = A([1 3 2], [1 2]);
