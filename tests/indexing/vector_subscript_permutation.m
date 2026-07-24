% Sort/permutation flagship: order is tracked as matrix[100 x 1]. Commit 1
% keeps the column count exact even though the row count stays unknown
% (the class domain needed to also know the row count is commit 2).
% EXPECT: Xsorted = matrix[None x 3]
% EXPECT: top10 = matrix[10 x 3]
% EXPECT: warnings = 0

X = randn(100, 3);
scores = rand(100, 1);
[ss, order] = sort(scores, 'descend');
Xsorted = X(order, :);
top10 = Xsorted(1:10, :);
