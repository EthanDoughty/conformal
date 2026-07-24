% Test: numel/length of a Cell fold to a concrete extent, inline and hoisted,
% including the for-loop-over-numel(K) idiom (Frobenius norms of a cell batch).
% EXPECT: warnings = 0
% EXPECT: K = cell[1 x 3]
% EXPECT: normK = matrix[1 x 3]
% EXPECT: m = matrix[1 x 3]
% EXPECT: n = scalar
% EXPECT: p = matrix[1 x 3]

K = {rand(3), rand(5), rand(4)};
normK = zeros(1, numel(K));
m = zeros(1, length(K));

% Hoisted form: the statement-level alias hook must agree with the inline fold.
n = numel(K);
p = zeros(1, n);

for i = 1:numel(K)
    normK(i) = norm(K{i}, 'fro');
    fprintf('Block %d is %dx%d, norm %.3f\n', i, size(K{i},1), size(K{i},2), normK(i));
end
disp(['Largest block norm: ' num2str(max(normK))]);
