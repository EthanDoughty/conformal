% EXPECT: warnings = 0
% EXPECT: e1 = scalar
% EXPECT: n1 = scalar
% EXPECT: l1 = scalar
% EXPECT: c1 = scalar
% EXPECT: nan1 = scalar
% EXPECT: inf1 = scalar
% EXPECT: fin1 = scalar
% EXPECT: sym1 = scalar

e1 = isempty(zeros(0, 3));
n1 = isnumeric(5);
l1 = islogical(true(2));
c1 = ischar('hello');
nan1 = isnan(5);
inf1 = isinf(5);
fin1 = isfinite(5);
sym1 = issymmetric(eye(3));
