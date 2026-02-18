% Test tilde (~) as unused output placeholder in destructuring assignment
% EXPECT: warnings = 0

x = [3 1 4 1 5];

% [~, idx] = sort(x): discard sorted values, keep indices
[~, idx] = sort(x);
% EXPECT: idx = matrix[1 x 5]

A = zeros(3, 3);

% [~, ~, V] = svd(A): discard U and S, keep V
[~, ~, V] = svd(A);
% EXPECT: V = matrix[3 x 3]

% [vals, ~] = sort(x): keep values, discard indices
[vals, ~] = sort(x);
% EXPECT: vals = matrix[1 x 5]
