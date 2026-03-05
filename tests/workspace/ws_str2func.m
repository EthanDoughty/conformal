% Test: str2func('funcName') returns a usable function handle
% EXPECT: warnings = 0

A = zeros(3, 4);
f = str2func('ws_scale');
y = f(A, 2);
% EXPECT: y = matrix[3 x 4]
