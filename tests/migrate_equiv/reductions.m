% SKIP_TEST
% Sum, mean, min, max
A = [1 2 3; 4 5 6];
disp(sum(A(:)));
disp(sum(A, 1));
disp(sum(A, 2));
disp(mean(A(:)));
disp(min(A(:)));
disp(max(A(:)));
