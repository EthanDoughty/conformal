% Pentagon lower-bound join: lower bounds survive/intersect across if/else branches.
% After both branches, lowerBounds[i] is only kept if it matches in both branches.
% The for-loop sets lowerBounds[i] = (start, 0) and the if/else branches do not
% reassign i, so the bound survives through the join and OOB is suppressed.
%
% EXPECT: warnings = 0

function test_pentagon_lower_branch()
    n = input('n');
    start = 1;
    A = zeros(n, n);
    flag = input('flag');
    for i = start:n
        if flag > 0
            A(i, 1) = i;
        else
            A(i, 2) = i * 2;
        end
    end
end
