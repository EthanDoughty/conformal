% Pentagon upper-bound suppression: for i = 1:5 with A = zeros(5,5).
% i <= 5 via Pentagon (direct bound) and A has row dim 5.
% EXPECT: warnings = 0
A = zeros(5, 5);
for i = 1:5
    x = A(i, 1);
end
