% EXPECT: warnings = 0
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: c = scalar
% Secondary variable c decrements inside a for loop body.
% The for-loop iteration variable i is excluded from widening, but c is not.
% With threshold widening, c's lower bound snaps to a finite negative threshold
% (not Unbounded) so c remains a finite scalar after the loop.
c = 0;
for i = 1:5
    c = c - 1;
end
% EXPECT: c = scalar
