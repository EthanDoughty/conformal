% Switch/case interval refinement: n is narrowed to [3,3] inside case 3.
% zeros(n, n) inside the case body sees n == 3, so A = matrix[3 x 3].
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
n = input('n');
switch n
    case 3
        A = zeros(n, n);
end
