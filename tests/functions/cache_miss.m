% Test: Polymorphic cache miss (different argument shapes)
% EXPECT: warnings = 1
% EXPECT: B = unknown
% EXPECT: D = matrix[5 x 5]

function y = square_it(x)
    y = x * x;
end

A = zeros(3, 4);
B = square_it(A);
C = zeros(5, 5);
D = square_it(C);
