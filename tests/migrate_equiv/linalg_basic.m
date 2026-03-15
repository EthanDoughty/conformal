% SKIP_TEST
% Basic linear algebra
A = [1 2; 3 4];
b = [5; 6];
x = A \ b;
disp(x);
disp(det(A));
disp(norm(b));
disp(diag(A));
