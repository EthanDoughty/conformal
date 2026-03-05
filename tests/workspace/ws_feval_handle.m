% Test: feval(@funcHandle, args) dispatches via handle resolution
% EXPECT: warnings = 0

A = zeros(3, 4);
f = @ws_scale;
y = feval(f, A, 2);
% EXPECT: y = matrix[3 x 4]
