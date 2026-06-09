% Regression: LhsAssign handler must evict trustedConsts.
% Any mutation via a chained LHS (M(2)(3)=, L{1}(2)=, L(1).x(2)=) must drop the
% trusted constant for the base variable so the downstream range cannot fold stale.
% W_CURLY_INDEXING_NON_CELL and W_FIELD_ACCESS_NON_STRUCT are expected for bugs 4 and 5.
% EXPECT: warnings = 2

% Bug 1: M(2)(3)=1 with prior M=5
M = 5;
M(2)(3) = 1;
q = 0:M/5:M;
p = zeros(1, 99);
r = q + p;
% EXPECT: q = matrix[1 x None]
% EXPECT: r = matrix[1 x None]

% Bug 4: L{1}(2)=9 with prior L=0.2 (also triggers W_CURLY_INDEXING_NON_CELL)
L4 = 0.2;
L4{1}(2) = 9;
w4 = 0:L4/400:L4/4;
q4 = zeros(1, 50);
bad4 = w4 .* q4;
% EXPECT: w4 = matrix[1 x None]
% EXPECT: bad4 = matrix[1 x None]

% Bug 5: L(1).x(2)=7 with prior L=0.2 (also triggers W_FIELD_ACCESS_NON_STRUCT)
L5 = 0.2;
L5(1).x(2) = 7;
a5 = 0:L5/400:L5/4;
b5 = zeros(1, 5);
c5 = a5 .* b5;
% EXPECT: a5 = matrix[1 x None]
% EXPECT: c5 = matrix[1 x None]
