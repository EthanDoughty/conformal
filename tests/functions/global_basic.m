% Test: global variable is declared and assigned inside a function, then read back
% EXPECT: warnings = 0
% EXPECT: result = matrix[2 x 2]

function y = get_with_global()
    global g;
    g = ones(2, 2);
    y = g;
end

result = get_with_global();
