% Test: Accumulation with symbolic range a:b
% EXPECT: warnings = 1
% EXPECT_FIXPOINT: E = matrix[(b + 2 - a) x 3]

E = zeros(1, 3);
for i = a:b
    E = [E; zeros(1, 3)];  % EXPECT_WARNING: W_REASSIGN_INCOMPATIBLE
end
