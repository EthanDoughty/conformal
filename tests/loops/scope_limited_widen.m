% Scope-limited widening preserves pre-loop intervals for stable variables.
% n is assigned before the loop and never modified inside the loop body.
% With scope-limited widening, n keeps its exact interval [5,5] so that
% B = zeros(n, n) resolves to matrix[5 x 5] rather than matrix[None x None].
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[5 x 5]
n = 5;
count = 0;
for i = 1:10
    count = count + 1;
end
B = zeros(n, n);
