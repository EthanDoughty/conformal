% Test: string literals inside [] after variables (transpose-vs-string disambiguation)
% The key lexer test: space before ' inside [] means string, not transpose.
% EXPECT: B = matrix[3 x 6]

% Transpose without space still works inside []
A = zeros(3, 3);
B = [A' A'];
