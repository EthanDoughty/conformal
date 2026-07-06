% Leading-dot and trailing-dot float literals: before NUMBER pattern support,
% .5 lexed as DOT + NUMBER and broke the enclosing statement to opaque, so A
% stayed unknown. Dot-operators must keep binding their dot (2.^x is
% elementwise), which the EXPECT on d pins down.

A = zeros(2, 3) * .5;
b = 90. + 1.E-8;
c = [-.75 .75];
x = ones(2, 2);
d = 2.^x;

% EXPECT: warnings = 0
% EXPECT: A = matrix[2 x 3]
% EXPECT: b = scalar
% EXPECT: c = matrix[1 x 2]
% EXPECT: d = matrix[2 x 2]
