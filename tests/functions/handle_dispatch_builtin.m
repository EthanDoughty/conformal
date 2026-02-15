% Test: Function handle dispatches to builtin
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: y = matrix[3 x 3]

f = @zeros;
y = f(3, 3);
