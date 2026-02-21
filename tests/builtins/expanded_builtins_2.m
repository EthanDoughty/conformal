% Test: newly recognized MATLAB builtins (dogfood sprint additions)
% EXPECT: warnings = 0

% File I/O builtins (recognized, no shape handler)
fid = fopen('data.txt', 'r');
line = fgets(fid);
fseek(fid, 0, 'bof');
pos = ftell(fid);
C = textscan(fid, '%f');
fclose(fid);

% Statistics/Signal toolbox builtins (recognized, no shape handler)
x = randn(100, 1);
autocorr(x);
y = mvnpdf(x, 0, 1);
