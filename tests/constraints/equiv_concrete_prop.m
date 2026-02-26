% Test: concrete value propagation through equivalence class.
% A*B records A.cols == B.rows => 5 == m. Post-analysis resolves m -> 5.
% EXPECT: warnings = 0
% EXPECT: D = matrix[5 x 3]
A = rand(n, 5);
B = rand(m, 3);
C = A * B;
D = zeros(m, 3);
