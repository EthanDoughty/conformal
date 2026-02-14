% Test: Function called inside loop body (cache interaction)
% Each iteration calls func with same arg shape → cache hits after first
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]

function y = identity(x)
    y = x;
end

A = zeros(3, 3);
for i = 1:5
    A = identity(A);
end
