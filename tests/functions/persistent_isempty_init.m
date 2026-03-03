% Test: persistent variable with isempty guard initialization pattern
% EXPECT: warnings = 0
% EXPECT: result = matrix[4 x 1]

function y = get_persistent()
    persistent x;
    if isempty(x)
        x = zeros(4, 1);
    end
    y = x;
end

result = get_persistent();
