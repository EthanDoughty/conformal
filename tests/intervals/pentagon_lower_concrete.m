% Pentagon lower-bound: concrete start value (1) in for-loop.
% When lo endpoint is a variable with exact interval [1,1], the Pentagon lower
% bridge tightens i's lower bound to 1, suppressing the OOB warning.
%
% EXPECT: warnings = 0

function test_pentagon_lower_concrete()
    n = input('n');
    A = zeros(n, 5);
    s = 1;
    for i = s:5
        v = A(i, 1);
    end
end
