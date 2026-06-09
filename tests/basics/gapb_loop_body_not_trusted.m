% Regression: loop-body reassigned variables must
% not be used as trusted constants when folding ranges inside the loop body.
% EXPECT: warnings = 0

% Bug 8: integer stepped range 1:1:n where n is modified inside the loop.
n8 = 5;
for k8 = 1:3
    n8 = n8 + 1;
end
a8 = 1:1:n8;
b8 = zeros(1, 5);
c8 = a8 .* b8;
% EXPECT: a8 = matrix[1 x None]
% EXPECT: c8 = matrix[1 x None]

% Bug 9: stepped range inside loop, step variable reassigned in same loop.
h9 = 0.5;
for k9 = 1:2
    g9 = 0:h9:1;
    h9 = h9 / 2;
end
v9 = g9 + zeros(1, 5);
% EXPECT: g9 = matrix[1 x None]
% EXPECT: v9 = matrix[1 x None]

% Bug 10: stepped range inside loop, step reassigned (different initial value).
h10 = 0.1;
for k10 = 1:3
    g10 = 0:h10:1;
    h10 = h10/2;
end
post10 = g10 + zeros(1, 41);
% EXPECT: g10 = matrix[1 x None]
% EXPECT: post10 = matrix[1 x None]
