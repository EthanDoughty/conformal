% Test: Lambda storage - two lambdas get distinct IDs
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: g = function_handle

f = @(x) x + 1;
g = @(y) y * 2;
