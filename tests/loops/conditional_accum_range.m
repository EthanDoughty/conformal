% Test: Conditional accumulation produces range dimension in fixpoint mode
% Only one branch of if modifies A; fixpoint widens row dim to Range
% Default mode: single pass, if-join produces Unknown rows
% Fixpoint mode: widening produces open range: at least 3 rows
% EXPECT: warnings = 1
% EXPECT: A = matrix[None x 3]
% EXPECT_FIXPOINT: warnings = 0
% EXPECT_FIXPOINT: A = matrix[3.. x 3]

A = zeros(3, 3);
for i = 1:10
    if cond
        A = [A; zeros(1, 3)];
    end
end
