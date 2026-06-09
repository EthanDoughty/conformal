% Regression: a workspace mutator in a condition, index arg, or assignment RHS must
% evict the stale trusted constant before any range fold (no false dimension warning).
% Expected behaviour: NO dimension warning; z/bad are 1xNone (symbolic or unknown).
% EXPECT: warnings = 0

% ---- Case 5: try/catch named exception variable (catch n) ----
% The catch clause binds 'n' to the MException object; 'n' is no longer the
% integer 4.  Conservative fix: evict all trusted constants before catch body.
n5 = 4;
try
    y5 = sqrt(2);
catch n5
end
z5 = 0:1:n5;
w5 = [z5; 1 2 3];
% EXPECT: z5 = matrix[1 x None]
% EXPECT: w5 = matrix[2 x None]

% ---- Case 7: workspace mutator in if condition ----
% assignin('base','n7',20) runs before the if body; n7 is no longer 10.
n7 = 10;
if assignin('base','n7',20)
  d7 = 1;
end
z7 = 0:2.5:n7;
w7 = [1 2 3];
y7 = [z7; w7];
% EXPECT: z7 = matrix[1 x None]
% EXPECT: y7 = matrix[2 x None]

% ---- Case 8: eval in index range arg (SteppedRange) ----
% eval('n8 = 20; 1') mutates n8 before the colon range; n8 is no longer 10.
n8 = 10;
A8 = ones(1,50);
b8 = A8(eval('n8 = 20; 1'):3:end);
z8 = 0:2.5:n8;
w8 = [1 2 3];
y8 = [z8; w8];
% EXPECT: z8 = matrix[1 x None]
% EXPECT: y8 = matrix[2 x None]

% ---- Case 9: eval in IndexAssign RHS ----
% arr9(1) = eval('n9 = 20; 7') mutates n9; z9 fold must use post-mutation n9.
n9 = 10;
arr9 = zeros(1,3);
arr9(1) = eval('n9 = 20; 7');
z9 = 0:2.5:n9;
w9 = [1 2 3];
y9 = [z9; w9];
% EXPECT: z9 = matrix[1 x None]
% EXPECT: y9 = matrix[2 x None]

% ---- Case 10: cellfun(@eval,...) hidden workspace mutator ----
% @eval is a function handle to a workspace mutator; cellfun executes it.
n10 = 10;
cellfun(@eval, {'n10 = 20;'});
z10 = 0:2.5:n10;
w10 = [1 2 3];
y10 = [z10; w10];
% EXPECT: z10 = matrix[1 x None]
% EXPECT: y10 = matrix[2 x None]

% ---- Case 12: eval in IndexAssign RHS (integer range) ----
n12 = 10;
A12 = zeros(1,3);
A12(1) = eval('n12 = 4;');
z12 = 1:n12;
w12 = zeros(1,4);
bad12 = z12 + w12;
% z12 may be symbolic (matrix[1 x n12]) since n12 is an unknown scalar after eviction.
% bad12 must be 1xNone (no stale fold to concrete 10, dims incompatible with 1x4).
% EXPECT: bad12 = matrix[1 x None]

% ---- Case 13: eval in if condition ----
n13 = 10;
if eval('n13 = 4;')
  q13 = 1;
end
z13 = 1:n13;
w13 = zeros(1,4);
bad13 = z13 + w13;
% z13 may be symbolic (matrix[1 x n13]).
% EXPECT: bad13 = matrix[1 x None]

% ---- Case 14: eval in IndexAssign inside a for-loop ----
% A(k) = eval('dx14 = 0.1;') mutates dx14 inside the loop body.
% After the loop, dx14 must not be trusted as 0.5.
dx14 = 0.5;
A14 = zeros(1,3);
for k14 = 1:3
  A14(k14) = eval('dx14 = 0.1;');
end
z14 = 0:dx14:1;
w14 = zeros(1,11);
bad14 = z14 + w14;
% EXPECT: z14 = matrix[1 x None]
% EXPECT: bad14 = matrix[1 x None]

% ---- Case 15: while-loop body modifies n; post-loop range must be symbolic ----
% n15=10 before the loop; the body does n15=n15-1; after the loop n15 is unknown.
n15 = 10;
while cond15
  n15 = n15 - 1;
end
z15 = 1:n15;
w15 = zeros(1,4);
bad15 = z15 + w15;
% z15 may be symbolic (matrix[1 x n15]).
% EXPECT: bad15 = matrix[1 x None]
