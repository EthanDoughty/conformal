% Regression: overflow and degenerate colon ranges.
% EXPECT: warnings = 0

% Bug 2: range endpoint exactly at Int32.MaxValue + 1 must degrade to Unknown.
z_ob = 1:2147483648;
w_ob = zeros(1, 50);
y_ob = z_ob + w_ob;
% EXPECT: z_ob = matrix[1 x None]
% EXPECT: y_ob = matrix[1 x None]

% Bug 11: plain non-stepped colon at Int32.MaxValue boundary.
% 0:2147483645 and 0:2147483646 are within range; 0:2147483647 overflows Int32.
a_ovf = 0:2147483645;
b_ovf = 0:2147483646;
c_ovf = 0:2147483647;
d_ovf = 0:1:2147483647;
% EXPECT: a_ovf = matrix[1 x 2147483646]
% EXPECT: b_ovf = matrix[1 x 2147483647]
% EXPECT: c_ovf = matrix[1 x None]
% EXPECT: d_ovf = matrix[1 x None]

% Bug 12: reverse plain colon b < a should produce empty 1x0, not negative.
t12 = 5:1;
u12 = zeros(1, 0);
v12 = t12 + u12;
% EXPECT: t12 = matrix[1 x 0]
% EXPECT: u12 = matrix[1 x 0]
% EXPECT: v12 = matrix[1 x 0]

% Bug 13: Inf/NaN endpoints must produce Unknown, not a symbolic (Inf+1) dim.
a13 = 0:Inf;
b13 = 0:NaN;
% EXPECT: a13 = matrix[1 x None]
% EXPECT: b13 = matrix[1 x None]
