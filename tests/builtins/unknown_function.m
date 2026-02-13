% Test 28: expanded builtins and unknown function warning
X = randn(3, 4);
Y = my_custom_func(5);
Z = Y + 1;
% EXPECT: warnings = 1
% EXPECT: X = matrix[3 x 4]
% EXPECT: Y = unknown
% EXPECT: Z = unknown
