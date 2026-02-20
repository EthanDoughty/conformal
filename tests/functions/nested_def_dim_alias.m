% Test: Dimension aliasing through nested function
% EXPECT: warnings = 0
% EXPECT: result = matrix[n x n]

function result = outer(n)
    result = make_square(n);
    function y = make_square(k)
        y = zeros(k, k);
    end
end

result = outer(n);
