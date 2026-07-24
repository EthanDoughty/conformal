% Test: Two-output min/max on vectors and the max(A,[],2) form (Cluster B2).
% MATLAB returns two SCALARS for [m,i] = max(v) / min(v) on a vector, not
% a 1-by-n row. The matrix form is unchanged. [m,i] = max(A,[],2) reduces
% along dim 2, giving one row per input row (r-by-1), not the input's
% column count.

% EXPECT: warnings = 0

v = zeros(1, 10);
[mx, ix] = max(v);
[mn, in] = min(v);
% EXPECT: mx = scalar
% EXPECT: ix = scalar
% EXPECT: mn = scalar
% EXPECT: in = scalar

% Downstream consequence: indexing v by the (now scalar) index yields a
% scalar, not a 1-by-unknown row.
p = v(ix);
% EXPECT: p = scalar

% Column vector: also scalar, not matrix[1 x 1].
vc = zeros(10, 1);
[mxc, ixc] = max(vc);
% EXPECT: mxc = scalar
% EXPECT: ixc = scalar

% A genuine matrix (not a vector): unchanged, guards against over-correction.
A = zeros(4, 5);
[mA, iA] = max(A);
% EXPECT: mA = matrix[1 x 5]
% EXPECT: iA = matrix[1 x 5]

% max(A,[],2): reduces along columns, one result per row.
[mA2, iA2] = max(A, [], 2);
% EXPECT: mA2 = matrix[4 x 1]
% EXPECT: iA2 = matrix[4 x 1]
