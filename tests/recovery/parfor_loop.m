% Test: parfor loops parse and analyze identically to for loops
% EXPECT: warnings = 0
% EXPECT: A = matrix[10 x 3]

A = zeros(10, 3);
parfor i = 1:10
    A(i, :) = [1 2 3];
end
