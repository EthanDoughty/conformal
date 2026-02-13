% Test 21: Invalid non-scalar index argument, warning is expected
% idx is a 2 x 2 matrix; using it as an index should warn and result is unknown.
%
% EXPECT: warnings = 1
% EXPECT: y = unknown

A = zeros(3, 4);
idx = zeros(2, 2);
y = A(idx, 1);