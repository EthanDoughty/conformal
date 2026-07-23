% Test: a size vector passed to a 1-arg constructor must not be read as a
% scalar extent (the confidently-wrong matrix[sz x sz] fabrication)
% EXPECT: warnings = 0
% EXPECT: sz = matrix[1 x 2]
% EXPECT: A = matrix[None x None]
% EXPECT: v = matrix[1 x 5]
% EXPECT: B = matrix[None x None]
% EXPECT: t = matrix[1 x 3]
% EXPECT: sz2 = matrix[1 x 2]
% EXPECT: E = matrix[None x None]
% EXPECT: n = scalar
% EXPECT: C = matrix[6 x 6]
% EXPECT: D = matrix[p x p]

sz = [3 4];
A = zeros(sz);

v = zeros(1, 5);
B = ones(v);

t = [1 2 3];
sz2 = size(t);
E = zeros(sz2);

n = 6;
C = zeros(n);

D = zeros(p);
