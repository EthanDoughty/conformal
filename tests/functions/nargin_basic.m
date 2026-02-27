% Test: Function with 3 params called with 2 args (nargin support)
% No arg-count mismatch warning. Third param (c) is Bottom.
% EXPECT: warnings = 0
% EXPECT: result = scalar

function y = add3(a, b, c)
    if nargin < 3
        c = 0;
    end
    y = a + b + c;
end

result = add3(1, 2);
