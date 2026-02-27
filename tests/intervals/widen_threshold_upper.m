% EXPECT: warnings = 0
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: x = scalar
% Loop counter increments by 1 starting from 0.
% With threshold widening, the upper bound snaps to a finite threshold
% (not Unbounded) so x remains a finite scalar after the loop.
x = 0;
for i = 1:5
    x = x + 1;
end
% EXPECT: x = scalar
