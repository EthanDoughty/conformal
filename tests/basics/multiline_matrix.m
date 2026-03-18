% Test: matrix literal with leading and trailing newlines
% EXPECT: warnings = 0
% EXPECT: A = matrix[2 x 2]
% EXPECT: B = matrix[1 x 3]

A = [
1 2;
3 4
];

B = [
1 2 3
];
