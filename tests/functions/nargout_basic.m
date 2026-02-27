% Test: nargout tracking in single-return context
% nargout=1, second output not assigned
% EXPECT: warnings = 0
% EXPECT: y = matrix[3 x 3]

function [A, B] = twouts(n)
    A = zeros(n, n);
    if nargout > 1
        B = ones(n, n);
    end
end

y = twouts(3);
