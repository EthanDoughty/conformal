% Test: MATLAB predefined constants are recognized as scalars
% EXPECT: warnings = 0
% EXPECT: a = scalar
% EXPECT: b = scalar
% EXPECT: c = scalar
% EXPECT: d = scalar
% EXPECT: e = scalar
% EXPECT: f = scalar
% EXPECT: g = matrix[3 x 3]

a = pi;
b = inf;
c = eps;
d = true;
e = nan;
f = i;

% Constants work in expressions
g = pi * ones(3, 3);

% User assignment shadows constant
pi = 42;
% EXPECT: pi = scalar
