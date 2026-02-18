% Test: calling no-arg function (no parentheses in definition)
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
A = noarg_helper();
