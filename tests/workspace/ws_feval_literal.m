% Test: feval('funcName', args) dispatches to named function
% EXPECT: warnings = 0

A = zeros(3, 4);
% feval with string literal first arg dispatches to ws_scale
y = feval('ws_scale', A, 2);
% EXPECT: y = matrix[3 x 4]
