% Test: meshgrid's row/column order (Cluster B2).
% [X,Y] = meshgrid(x,y): BOTH outputs are numel(y)-by-numel(x). Rows come
% from the SECOND argument. This file deliberately uses a and b with
% DIFFERENT lengths (41 vs 5) so a transposed rule cannot pass by accident.

% EXPECT: warnings = 0

a = zeros(1, 41);
b = zeros(1, 5);
[X, Y] = meshgrid(a, b);
% EXPECT: X = matrix[5 x 41]
% EXPECT: Y = matrix[5 x 41]

% Inline ranges: the parser emits these as SteppedRange/Range IndexArgs,
% exercising argToShapeWithRanges through both paths in the same call.
[X2, Y2] = meshgrid(-2:0.1:2, 1:5);
% EXPECT: X2 = matrix[5 x 41]
% EXPECT: Y2 = matrix[5 x 41]

% Symbolic form: free dimension names carried through unchanged.
[X3, Y3] = meshgrid(1:nc, 1:nr);
% EXPECT: X3 = matrix[nr x nc]
% EXPECT: Y3 = matrix[nr x nc]

% One-argument form: meshgrid(a) == meshgrid(a,a), so it is n-by-n.
X4 = meshgrid(a);
% EXPECT: X4 = matrix[41 x 41]

% Single-output form of the two-argument call: same shape as the first
% multi-output target.
X5 = meshgrid(a, b);
% EXPECT: X5 = matrix[5 x 41]

% Three-output form builds a 3-D grid the 2-D shape domain can't represent,
% so it must stay unknown rather than guess a 2-D shape.
[Xu, Yu, Zu] = meshgrid(a, b, a);
% EXPECT: Xu = unknown
% EXPECT: Yu = unknown
% EXPECT: Zu = unknown
