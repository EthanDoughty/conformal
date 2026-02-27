% Pentagon must NOT suppress OOB when bound variable exceeds matrix dim.
% i <= 10 via Pentagon, but A has row dim 5. Pentagon does not prove in-bounds.
% EXPECT: warnings >= 1
A = zeros(5, 5);
for i = 1:10
    x = A(i, 1);
end
