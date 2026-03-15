% SKIP_TEST
% Matrix arithmetic and element-wise ops
A = [1 2; 3 4];
B = [5 6; 7 8];
C = A + B;
D = A * B;
E = A .* B;
F = A * 2;
G = 3 * B;
disp(C);
disp(D);
disp(E);
disp(F);
disp(G);
