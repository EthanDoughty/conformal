% Test 13: Basic matrix literals, no warnings are expected
% This test validates parsing + shape inference for bracket literals:
%   A = [1 2 3]   -> 1 x 3
%   B = [1;2;3]   -> 3 x 1
%   C = [1 2;3 4] -> 2 x 2
%
% EXPECT: warnings = 0
% EXPECT: A = matrix[1 x 3]
% EXPECT: B = matrix[3 x 1]
% EXPECT: C = matrix[2 x 2]

A = [1 2 3];
B = [1;2;3];
C = [1 2; 3 4];
