% Test: Polymorphic cache hit (same argument shapes)
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]
% EXPECT: C = matrix[3 x 3]

function y = square_matrix(x)
    y = x * x;
end

A = zeros(3, 3);
B = square_matrix(A);
C = square_matrix(zeros(3, 3));
