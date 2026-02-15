% Test: Lambda assigned in if/else branches, called after join
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: y = matrix[3 x 1]

A = zeros(3, 1);
if 1 > 0
    f = @(x) x + 1;
else
    f = @(x) x * 2;
end
y = f(A);
