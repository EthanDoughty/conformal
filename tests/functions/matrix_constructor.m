% Test: Function using dimension arguments to construct matrix
% Dimension aliasing: make_matrix(n, m) should return matrix[n x m], not matrix[rows x cols]
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x m]

function result = make_matrix(rows, cols)
    result = zeros(rows, cols);
end

A = make_matrix(n, m);
