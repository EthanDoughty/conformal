% Test: function scope isolation prevents constraint leakage to caller.
% Inside multiply(), A*B records 5 == m (X.cols == Y.rows). But SnapshotScope
% restores the caller's DimEquiv on function exit, so 5 == m does not persist.
% After the call, D = zeros(m, 3) remains matrix[m x 3] (m not resolved to 5).
% EXPECT: warnings = 0
% EXPECT: D = matrix[m x 3]
function C = multiply(A, B)
    C = A * B;
end
X = rand(3, 5);
Y = rand(m, 7);
Z = multiply(X, Y);
D = zeros(m, 3);
