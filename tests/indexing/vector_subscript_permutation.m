% Sort/permutation flagship: order is tracked as matrix[100 x 1]. Commit 2's
% value-class domain classifies order as numeric (sort's second output), so
% its numel is known and Xsorted's row count folds to exactly 100.
% EXPECT: Xsorted = matrix[100 x 3]
% EXPECT: top10 = matrix[10 x 3]
% EXPECT: warnings = 0

X = randn(100, 3);
scores = rand(100, 1);
[ss, order] = sort(scores, 'descend');
Xsorted = X(order, :);
top10 = Xsorted(1:10, :);
