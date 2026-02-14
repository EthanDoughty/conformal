% Test: Column dimension grows while row dimension stays constant
% Default: single-pass sees horzcat grow cols by 1
% Fixpoint: row preserved (both iterations have 2), col widened to None
% EXPECT: warnings = 1
% EXPECT: A = matrix[2 x 4]
% EXPECT_FIXPOINT: warnings = 1
% EXPECT_FIXPOINT: A = matrix[2 x None]

A = zeros(2, 3);
for i = 1:n
    A = [A, zeros(2, 1)];
end
