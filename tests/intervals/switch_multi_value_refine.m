% Switch/case multi-value cell refinement: {1, 5} narrows n to hull [1, 5].
% zeros(n, n) inside the case body sees n in [1, 5].
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x n]
n = input('n');
switch n
    case {1, 5}
        A = zeros(n, n);
end
