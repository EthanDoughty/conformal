% Test: spdiags builtin (sparse diagonal matrix construction)
% Verifies the 4-arg form produces an explicitly shaped matrix.
% EXPECT: warnings = 0

% 4-arg form spdiags(B, d, m, n) -> m-by-n.
% EXPECT: B = matrix[5 x 3]
B = ones(5, 3);
% EXPECT: d = matrix[3 x 1]
d = [-1; 0; 1];
% EXPECT: A = matrix[5 x 5]
A = spdiags(B, d, 5, 5);

% Different m and n.
% EXPECT: A2 = matrix[10 x 8]
A2 = spdiags(B, d, 10, 8);

% Larger square form with a single-diagonal source.
% EXPECT: A3 = matrix[100 x 100]
A3 = spdiags(ones(100, 1), 0, 100, 100);
