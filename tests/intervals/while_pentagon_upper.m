% While-loop Pentagon upper-bound: while i <= n suppresses OOB on A(i, 1).
% Pentagon records upperBounds[i] = (n, 0), applyPentagonBridge tightens
% i's interval, and pentagonProvesInBounds suppresses the OOB warning.
%
% EXPECT: warnings = 0

function test_while_pentagon_upper()
    n = input('n');
    A = zeros(n, 1);
    i = 1;
    while i <= n
        A(i, 1) = i;
        i = i + 1;
    end
end
