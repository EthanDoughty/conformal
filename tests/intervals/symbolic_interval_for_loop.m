% EXPECT: warnings = 0
A = zeros(n, 3);
for i = 1:n
    x = A(i, 1);
end
% This test passes today (no interval = no OOB check). After v1.8.0, i gets
% Interval(1, SymDim('n')), which must NOT introduce false positives.
% EXPECT: A = matrix[n x 3]
