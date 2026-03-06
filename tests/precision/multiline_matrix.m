% Test: multiline matrix literals without ... continuation
% The parser should handle semicolons followed by newlines in matrix literals

A = [1 2 3;
     4 5 6;
     7 8 9];

B = [1, 0, 0;
     0, 1, 0;
     0, 0, 1];

C = [1;
     2;
     3];

result = A * B * C;
% EXPECT: A: matrix[3 x 3]
% EXPECT: B: matrix[3 x 3]
% EXPECT: C: matrix[3 x 1]
% EXPECT: result: matrix[3 x 1]
