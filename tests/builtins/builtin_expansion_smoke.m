% Smoke test for builtin expansion batches 6a-6e.
% EXPECT: warnings = 0
% EXPECT: A = matrix[3 x 3]
% EXPECT: B = matrix[3 x 3]
% EXPECT: C = matrix[3 x 3]
% EXPECT: D = matrix[3 x 3]

% Batch 6a: passthrough
A = zeros(3, 3);
B = sortrows(A);

% Batch 6b: fixed-dimension transform
C = axang2rotm([1 0 0 0.5]);

% Batch 6c: reduction (mode is a reduction like mean)
x = [1; 2; 3; 4; 5];
m = mode(x);

% Batch 6d: cov returns [p x p] from [n x p]
D = cov(A);
