% SKIP_TEST
% Logical operations
a = true;
b = false;
disp(a & b);
disp(a | b);
disp(~a);
disp(~b);
A = [1 0 1; 0 1 0];
disp(~A);
disp(any(A(:)));
disp(all(A(:)));
