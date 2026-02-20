% Test: concrete scalar values propagate into dimension constructors
m = 3;
A = zeros(m, m);
% EXPECT: A = matrix[3 x 3]

n = 4;
B = eye(m, n);
% EXPECT: B = matrix[3 x 4]

k = m + 1;
C = rand(k, k);
% EXPECT: C = matrix[4 x 4]

D = ones(m, k);
% EXPECT: D = matrix[3 x 4]

% Non-constant: should remain symbolic
p = size(A, 1);
E = zeros(p, p);
% EXPECT: E = matrix[p x p]

% EXPECT: warnings = 0
