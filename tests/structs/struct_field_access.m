% Test: Read struct fields
% EXPECT: warnings = 0
% EXPECT: s = struct{a: matrix[2 x 2], b: scalar}
% EXPECT: A = matrix[2 x 2]
% EXPECT: val = scalar

s.a = zeros(2, 2);
s.b = 10;
A = s.a;
val = s.b;
