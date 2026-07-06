% Multi-assign with indexed/field targets: before MultiTarget support these
% statements fell to parse-recovery opaque, so b stayed unbound and A kept a
% stale shape. Now the plain target binds and the indexed base goes conservative.

A = zeros(2, 2);
x = zeros(3, 4);
[A(1), b] = size(x);

% EXPECT: warnings = 0
% EXPECT: A = unknown
% EXPECT: b = scalar
