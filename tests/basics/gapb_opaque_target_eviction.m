% Regression: OpaqueStmt assignment targets must
% be evicted from trustedConsts at the statement site (bug 11) and must be included
% in collectModifiedVars so the loop pre-scan drops them before the body runs
% (bugs 12, 13 for-loop and while-loop variants).
%
% Each OpaqueStmt (recovered syntax error) emits W_UNSUPPORTED_STMT (3 total).
% No dimension warnings must fire despite the opaque assignments.
% EXPECT: warnings = 3

% Bug 11: top-level OpaqueStmt target eviction.
dx11 = 0.1;
dx11 = 0.5 +
v11 = 0:dx11:1;
w11 = ones(1, 7);
y11 = [v11; w11];
% EXPECT: v11 = matrix[1 x None]
% EXPECT: y11 = matrix[2 x None]

% Bug 12: for-loop OpaqueStmt target -- collectModifiedVars must include it.
dx12 = 0.1;
for k12 = 1:2
  dx12 = foo bar
end
v12 = 0:dx12:1;
w12 = ones(1, 7);
y12 = [v12; w12];
% EXPECT: v12 = matrix[1 x None]
% EXPECT: y12 = matrix[2 x None]

% Bug 13: while-loop OpaqueStmt target.
dx13 = 0.1;
j13 = 0;
while j13 < 3
  dx13 = foo bar
  j13 = j13 + 1;
end
v13 = 0:dx13:1;
w13 = ones(1, 7);
y13 = [v13; w13];
% EXPECT: v13 = matrix[1 x None]
% EXPECT: y13 = matrix[2 x None]
