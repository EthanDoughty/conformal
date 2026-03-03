% Test: global variable set by one function is visible in another.
% set_global_x writes global x; read_global_x reads it.
% The void call to set_global_x at script level emits W_PROCEDURE_IN_EXPR (expected).
% EXPECT: warnings = 1
% EXPECT: result = matrix[3 x 3]

function set_global_x()
    global x;
    x = zeros(3, 3);
end

function y = read_global_x()
    global x;
    y = x;
end

set_global_x();  % EXPECT_WARNING: W_PROCEDURE_IN_EXPR
result = read_global_x();
