% Test 11: Suspicious comparison between matrix and scalar (warning expected)
% In MATLAB, A == 0 is elementwise and produces a logical matrix.
% Using it as a condition is typically a bug unless combined with all()/any().
%
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: x = scalar

A = zeros(3, 4);

if A == 0
    x = 1;
else
    x = 2;
end
