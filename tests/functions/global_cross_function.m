% Test: global variable set by one function is visible in another.
% set_global_x writes global x; read_global_x reads it.
% Calling a procedure as a statement is valid MATLAB (no warning).
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function set_global_x()
    global x;
    x = zeros(3, 3);
end

function y = read_global_x()
    global x;
    y = x;
end

set_global_x();
result = read_global_x();
