% Test 16: MATLAB-like indexing slices, no warnings are expected
% A is 3 x 4.
% r = A(2,:)  -> 1 x 4
% c = A(:,3)  -> 3 x 1
% s = A(2,3)  -> scalar
%
% EXPECT: warnings = 0
% EXPECT: r = matrix[1 x 4]
% EXPECT: c = matrix[3 x 1]
% EXPECT: s = scalar

A = zeros(3, 4);
r = A(2, :);
c = A(:, 3);
s = A(2, 3);