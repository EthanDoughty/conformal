% Test 21: Invalid non-scalar index argument, warning is expected
% idx is a 2 x 2 matrix; using it as a row subscript should warn, but the
% column subscript (1) is still exact, so only the row extent is unknown.
%
% EXPECT: warnings = 1
% EXPECT: y = matrix[None x 1]

A = zeros(3, 4);
idx = zeros(2, 2);
y = A(idx, 1);  % EXPECT_WARNING: W_NON_SCALAR_INDEX