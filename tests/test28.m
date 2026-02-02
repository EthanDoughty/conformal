% Test 28: Multiline matrix literal should parse normally (no recovery)
% EXPECT: warnings = 0
% EXPECT: A = matrix[2 x 2]
% EXPECT: B = matrix[2 x 2]

A = [1 2
     3 4];
B = A + A;
