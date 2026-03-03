% While-loop Pentagon compound condition: while i >= 1 && i <= n
% Pentagon extracts both upper and lower bounds:
%   upperBounds[i] = (n, 0)  -- suppresses upper OOB
%   lowerBounds[i] = (one, 0) -- suppresses lower OOB (with concrete bridge)
% Both OOB warnings should be suppressed.
%
% EXPECT: warnings = 0

function test_while_pentagon_compound()
    n = input('n');
    A = zeros(n, 1);
    one = 1;
    i = n;
    while i >= one && i <= n
        A(i, 1) = i;
        i = i - 1;
    end
end
