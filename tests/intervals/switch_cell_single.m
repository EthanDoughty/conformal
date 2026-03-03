% Switch/case single-element cell literal: {3} narrows n to [3, 3].
% zeros(n, n) inside the case body resolves to matrix[3 x 3], same as case 3.
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
n = input('n');
switch n
    case {3}
        A = zeros(n, n);
end
