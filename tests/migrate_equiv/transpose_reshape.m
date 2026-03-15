% SKIP_TEST
% Transpose and reshape
A = [1 2 3; 4 5 6];
B = A';
C = reshape(A, 3, 2);
disp(B);
disp(C);
disp(size(B));
disp(size(C));
