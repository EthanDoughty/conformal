% Indexed assignment inside function body
% EXPECT: warnings = 0

function M = fill_matrix(n)
    M = zeros(n, n);
    for i = 1:n
        M(i, i) = 1;
    end
end

R = fill_matrix(5);
% EXPECT: R = matrix[5 x 5]

S = fill_matrix(n);
% EXPECT: S = matrix[n x n]
