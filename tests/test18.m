% Test 18: Indexing a scalar is suspicious, warning is expected
% x is scalar; y = x(1) should warn and be treated as unknown.
%
% EXPECT: warnings = 1
% EXPECT: x = scalar
% EXPECT: y = unknown

x = 7;
y = x(1);