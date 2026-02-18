% Test: space-separated destructuring [a b] = expr
% EXPECT: warnings = 0
function [x y] = myfunc(a)
    x = a;
    y = zeros(2,2);
end
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[2 x 2]
[A B] = myfunc(zeros(3,3));
