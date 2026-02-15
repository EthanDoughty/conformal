% Test: Zero-argument lambda call
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: x = scalar

f = @() 42;
x = f();
