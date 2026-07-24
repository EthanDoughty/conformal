% Commit 2 soundness guard: the value-class map must be INTERSECTED at
% branch merges, not last-write-wins. Both branches give s1/s2 the SAME
% concrete shape (100 x 1), so only the class map -- not the shape join --
% decides whether A(s1,:)/A(s2,:) folds to a concrete extent. If a class
% survived from only one branch (a plain union, or a merge that always
% favors one side, instead of intersection), y1/y2 would wrongly become
% matrix[100 x 4] instead of the sound matrix[None x 4].
% EXPECT: y1 = matrix[None x 4]
% EXPECT: y2 = matrix[None x 4]
% EXPECT: z = matrix[None x 4]
% EXPECT: warnings = 3

A = randn(100, 4);
x = randn(100, 1);

% s1: numeric on the if-branch, logical on the else-branch.
if rand > 0.5
    s1 = randperm(100)';
else
    s1 = x > 0;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
end
y1 = A(s1, :);

% s2: logical on the if-branch, numeric on the else-branch (reversed order,
% so the guard also catches a merge that always favors one particular side).
if rand > 0.5
    s2 = x > 0;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
else
    s2 = randperm(100)';
end
y2 = A(s2, :);

% Reassignment must clear the stale class: s starts numeric (randperm),
% then is reassigned to a logical mask. Env.set's failsafe clearing must
% drop the numeric class so z does not fold to numel.
s = randperm(100)';
s = x > 0;  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
z = A(s, :);
