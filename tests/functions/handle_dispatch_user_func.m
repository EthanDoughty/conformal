% Test: Function handle dispatches to user-defined function
% EXPECT: warnings = 0
% EXPECT: f = function_handle
% EXPECT: y = matrix[3 x 3]

function out = makeMatrix(n)
    out = zeros(n, n);
end

f = @makeMatrix;
y = f(3);
