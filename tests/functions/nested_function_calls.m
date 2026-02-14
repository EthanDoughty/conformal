% Test: Nested user-defined function calls (no recursion)
% outer calls inner; both should analyze correctly
% EXPECT: warnings = 0
% EXPECT: A = matrix[4 x 4]

function y = inner(x)
    y = x + x;
end

function z = outer(w)
    z = inner(w);
end

A = outer(zeros(4, 4));
