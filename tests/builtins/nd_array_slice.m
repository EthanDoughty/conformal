% Test: 3D array slice extraction via ndArraySlices metadata.
% zeros(m,n,p) stores 2D slice metadata; A(:,:,k) returns the slice shape.
% EXPECT: warnings = 0
% EXPECT: Q1 = matrix[3 x 3]
% EXPECT: Q2 = matrix[n x n]
% EXPECT: Q3 = matrix[3 x 3]
% EXPECT: S = matrix[2 x 2]
% EXPECT: T = scalar

function [Q1, Q2, Q3, S, T] = test_3d()
    % Concrete 3D: zeros(3, 3, 10)
    P1 = zeros(3, 3, 10);
    Q1 = P1(:,:,1);

    % Symbolic 3D: zeros(n, n, T)
    P2 = zeros(n, n, T);
    Q2 = P2(:,:,t);

    % Assignment propagation: R = P1 copies metadata
    R = P1;
    Q3 = R(:,:,5);

    % Kalman-like pattern: 3D in loop
    P3 = zeros(2, 2, 10);
    for t2 = 1:10
        P3(:,:,t2) = eye(2);
    end
    S = P3(:,:,1);

    % size(A, 3) returns third dimension
    P4 = zeros(4, 4, 20);
    T = size(P4, 3);
end

[Q1, Q2, Q3, S, T] = test_3d();
