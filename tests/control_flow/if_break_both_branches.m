% Test: break in then-branch, assignment in else-branch -- both branches analyzed
% Before this fix, the else branch was skipped entirely when break was in then.
% EXPECT: warnings = 0
% EXPECT: B = matrix[2 x 3]
for i = 1:10
    if i > 5
        break;
    else
        B = ones(2, 3);
    end
end
