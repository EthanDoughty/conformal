% Test: Cell with symbolic dimensions
% EXPECT: warnings = 0
% EXPECT: A = matrix[n x m]
% EXPECT: c = cell[1 x 2]
% EXPECT: x = unknown

n = 3;
m = 4;
A = zeros(n, m);
c = {A, A};
x = c{1};
