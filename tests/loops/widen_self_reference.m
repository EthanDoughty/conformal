% Test: Self-referencing doubling (columns double each iteration)
% Default: single-pass sees A double from 2x3 to 2x6
% Fixpoint: rows stable (2), cols widened (3 -> 6 conflict -> None)
% EXPECT: warnings = 1
% EXPECT: A = matrix[2 x 6]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[2 x None]

A = zeros(2, 3);
for i = 1:n
    A = [A, A];
end
