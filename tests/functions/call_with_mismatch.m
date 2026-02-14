% Test: Function called with incompatible argument shape
% Function body expects square matrix (x*x), caller passes non-square
% EXPECT: warnings = 1
% EXPECT: A = matrix[3 x 4]
% EXPECT: B = unknown

function y = square_it(x)
    y = x * x;
end

A = zeros(3, 4);
B = square_it(A);
