% Phase 2 gate: scalar counter interval converges with threshold widening.
% Without the fix, count stays at [0, 1] (2 hops). With the fix, converges
% to a wider threshold like [0, 1000] after Phase 2 re-analysis.
% The key observable: zeros(1, count) should have count as a symbolic dim
% (not Unknown/None), proving the interval stayed finite.
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: B = matrix[1 x count]
count = 0;
for i = 1:100
    count = count + 1;
end
B = zeros(1, count);
