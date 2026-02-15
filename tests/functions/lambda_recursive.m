% Test: Recursive lambda detected
% EXPECT: warnings = 1
% EXPECT: f = function_handle
% EXPECT: y = unknown

f = @(x) f(x - 1);
y = f(5);
