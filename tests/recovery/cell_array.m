% Test 23: Parse recovery - cell array access (unsupported)
% Recovery should not consume 'end' inside block
% Tests that recovery stops before 'end' at depth 0

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
