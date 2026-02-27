% Pentagon domain: for-loop establishes i <= n upper bound.
% When n is known to be 5, the bridge tightens i's interval to [1,5].
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[5 x 5]
n = 5;
for i = 1:n
    % Inside loop: i <= n (Pentagon bound); bridge fires since n = [5,5]
end
B = zeros(n, n);
