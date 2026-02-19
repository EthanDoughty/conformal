% Test: domain builtins with shape handlers
% EXPECT: warnings = 0
% EXPECT: Y = matrix[4 x 4]
% EXPECT: Z = matrix[4 x 4]
% EXPECT: S = matrix[3 x 3]
% EXPECT: p = matrix[1 x 4]
% EXPECT: yy = matrix[5 x 1]
% EXPECT: n = scalar

A = rand(4, 4);

% FFT preserves shape
Y = fft(A);
Z = ifft(Y);

% sparse/full passthrough
S = full(sparse(3, 3));

% polyfit returns row vector
x = [1 2 3 4 5];
y = [2 4 5 4 5];
p = polyfit(x, y, 3);

% polyval returns same shape as input
yy = polyval(p, zeros(5, 1));

% ndims returns scalar
n = ndims(A);
