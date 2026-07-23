% Test: constructor adopts size(X)'s extents as [rows cols], not X x X
% EXPECT: warnings = 0
% EXPECT: t = matrix[1 x 1001]
% EXPECT: n = matrix[1 x 1001]
% EXPECT: z = matrix[1 x 1001]
% EXPECT: o = matrix[1 x 1001]
% EXPECT: r = matrix[1 x 1001]
% EXPECT: nn = matrix[1 x 1001]
% EXPECT: tt = matrix[1 x 1001]
% EXPECT: ee = matrix[1 x 1001]
% EXPECT: y = matrix[1 x 1001]
% EXPECT: K = cell[1 x 3]
% EXPECT: zk = matrix[1 x 3]
% EXPECT: ck = cell[1 x 3]
% EXPECT: A = matrix[m x 4]
% EXPECT: oA = matrix[m x 4]
% EXPECT: s = scalar
% EXPECT: zs = matrix[1 x 1]

t = 0:0.01:10;
n = randn(size(t));
z = zeros(size(t));
o = ones(size(t));
r = rand(size(t));
nn = nan(size(t));
tt = true(size(t));
ee = eye(size(t));
y = sin(t) + 0.1*randn(size(t));

K = {1, 2, 3};
zk = zeros(size(K));
ck = cell(size(K));

A = zeros(m, 4);
oA = ones(size(A));

s = 42;
zs = zeros(size(s));
