% Test: Sibling nested functions call each other
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function result = outer(n)
    result = double_eye(n);
    function y = make_eye(k)
        y = eye(k);
    end
    function z = double_eye(k)
        z = make_eye(k) + make_eye(k);
    end
end

result = outer(3);
