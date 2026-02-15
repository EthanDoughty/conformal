% Test 23: Parse recovery - cell array access
% Curly indexing on a non-cell value emits W_CURLY_INDEXING_NON_CELL

% EXPECT: warnings = 1
% EXPECT: A = matrix[2 x 2]
% EXPECT: B = unknown
% EXPECT: C = matrix[2 x 2]
% EXPECT: D = scalar

A = zeros(2, 2);
if 1
    B = A{1};
end
C = A + A;
D = 5;
