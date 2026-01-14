% Test 14: Matrix literal concatenation rules, one warning is expected
% A is 2 x 3 and B is 3 x 3.
% Y = [A B] is horizontal concatenation, which requires equal row counts.
% Since 2 ~= 3, the analysis should emit a warning for Y.
% X = [A; B] is vertical concatenation, which requires equal column counts.
% Since both have 3 columns, X should be valid with no warning.
%
% EXPECT: warnings = 1
% EXPECT: X = matrix[5 x 3]
% EXPECT: Y = matrix[None x 6]

A = zeros(2, 3);
B = zeros(3, 3);

X = [A; B];   % valid (5 x 3)
Y = [A B];    % invalid, warning expected (row mismatch)
