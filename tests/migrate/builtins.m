A = eye(4);
B = inv(A);
n = length(A);
s = size(A, 1);
v = diag(A);
d = det(A);
disp(d);
