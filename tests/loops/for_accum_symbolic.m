% Test: Accumulation with symbolic range a:b
% EXPECT: warnings = 1
% EXPECT_FIXPOINT: E = matrix[(-a+b+2) x 3]

E = zeros(1, 3);
for i = a:b
    E = [E; zeros(1, 3)];
end
