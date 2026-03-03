% Pentagon lower-bound tracking: for i = start:n where start is a variable >= 1.
% Pentagon records lowerBounds[i] = (start, 0), and applyPentagonLowerBridge
% tightens i's lo to start's exact value. pentagonProvesLowerBound then suppresses
% the "index may be < 1" OOB warning.
%
% EXPECT: warnings = 0

function test_pentagon_lower_bound()
    n = input('n');
    start = 1;
    A = zeros(n, n);
    for i = start:n
        A(i, 1) = i;
    end
end
