% Test 27: END inside parentheses does not trigger stop-before-END
% Tests that recovery delimiter tracking prevents stopping on END at depth>0

% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = unknown
% EXPECT: C = scalar

A = zeros(3, 4);
B = A.field(end);
C = 5;
