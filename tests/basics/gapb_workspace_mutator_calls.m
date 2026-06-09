% Regression: workspace-mutating calls in various positions
% must evict trustedConsts so range folds do not use stale values.
% EXPECT: warnings = 0

% Bug 1: assignin('base',...) as ExprStmt invalidates trusted constants.
L1 = 0.2;
assignin('base', 'L1', 0.5);
z1 = 0:L1:1;
w1 = [1 2 3];
y1 = [z1; w1];
% EXPECT: z1 = matrix[1 x None]
% EXPECT: y1 = matrix[2 x None]

% Bug 2 / Bug 3: eval/evalin on Assign RHS evicts workspace.
L2 = 0.2;
dummy2 = eval('L2 = 0.5;');
z2 = 0:L2:1;
w2 = [1 2 3];
y2 = [z2; w2];
% EXPECT: z2 = matrix[1 x None]
% EXPECT: y2 = matrix[2 x None]

% Bug 4: assignin on Assign RHS.
n4 = 10;
dummy4 = assignin('base', 'n4', 12);
z4 = 0:n4/4:n4;
w4 = [1 2 3];
y4 = [z4; w4];
% EXPECT: z4 = matrix[1 x None]
% EXPECT: y4 = matrix[2 x None]

% Bug 5: evalc as ExprStmt.
n5 = 10;
evalc('n5 = 12;');
z5 = 0:n5/4:n5;
w5 = [1 2 3];
y5 = [z5; w5];
% EXPECT: z5 = matrix[1 x None]
% EXPECT: y5 = matrix[2 x None]

% Bug 6: run('script.m') as ExprStmt.
n6 = 10;
run('setup.m');
z6 = 0:n6/4:n6;
w6 = [1 2 3];
y6 = [z6; w6];
% EXPECT: z6 = matrix[1 x None]
% EXPECT: y6 = matrix[2 x None]

% Bug 7: eval nested inside disp(...) -- recursive predicate catches it.
n7 = 10;
disp(eval('n7 = 12;'));
z7 = 0:n7/4:n7;
w7 = [1 2 3];
y7 = [z7; w7];
% EXPECT: z7 = matrix[1 x None]
% EXPECT: y7 = matrix[2 x None]
