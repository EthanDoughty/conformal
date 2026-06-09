% Regression: TRUE POSITIVES that must still warn after hardening.
% These demonstrate that the guard/index-extent and reshape paths correctly fold
% under equality guards, detecting real dimension mismatches.
% EXPECT: warnings = 5

% ---- TP Case 1: size()-aliased n under if n==4 equality guard ----
% size(A,2) is aliased to the column dim; under n==4 the index extent folds to 4.
% z(1x4) + zeros(1x3) -> mismatch.
A_tp1 = rand(3, randi(50));
n_tp1 = size(A_tp1, 2);
if n_tp1 == 4
  z_tp1 = A_tp1(1, 1:n_tp1);
  w_tp1 = z_tp1 + zeros(1, 3);  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH
end

% ---- TP Case 2: stepped index range under if n==9 equality guard ----
% Under n==9 the stepped range 1:2:9 = [1 3 5 7 9] -> length 5.
% z(1x5) + zeros(1x3) -> mismatch.
A_tp2 = ones(3, 100);
n_tp2 = 0;
if n_tp2 == 9
  z_tp2 = A_tp2(1, 1:2:n_tp2);
  w_tp2 = z_tp2 + zeros(1, 3);  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH
end

% ---- TP Case 3: index range folds loop-body var from valueRanges ----
% For-loop body always sets s=5 (single iteration 1:1); z = A(1, 1:5) -> 1x5.
% z(1x5) + zeros(1x3) -> mismatch.
A_tp3 = ones(3, 100);
s_tp3 = 0;
for k_tp3 = 1:1
  s_tp3 = 5;
end
z_tp3 = A_tp3(1, 1:s_tp3);
w_tp3 = z_tp3 + zeros(1, 3);  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH

% ---- TP Case 4: reshape folds n under if n==5 equality guard ----
% Under n==5: reshape(A[3x100], 3, 5) -> 3x5. B(3x5) + ones(3x3) -> mismatch.
A_tp4 = ones(3, 100);
n_tp4 = 0;
if n_tp4 == 5
  B_tp4 = reshape(A_tp4, 3, n_tp4);  % EXPECT_WARNING: W_RESHAPE_MISMATCH
  w_tp4 = B_tp4 + ones(3, 3);  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH
end
