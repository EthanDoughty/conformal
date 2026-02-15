% Test: Anonymous function assignment
% EXPECT: warnings = 0
% EXPECT: f = function_handle

f = @(x) x + 1;
