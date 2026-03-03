% Test: persistent variable is initialized and shape is tracked
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function y = make_persistent()
    persistent x;
    x = zeros(3, 3);
    y = x;
end

result = make_persistent();
