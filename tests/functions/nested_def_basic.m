% Test: Nested function definition and call
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function result = outer(n)
    result = inner(n);
    function y = inner(x)
        y = zeros(x, x);
    end
end

result = outer(3);
