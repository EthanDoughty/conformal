% Test: Core builtins found missing in dogfood corpus
% EXPECT: warnings = 0

% Degree trig (passthrough)
A = [1 2; 3 4];
B = tand(A);
% EXPECT: B = matrix[2 x 2]

C = sind(A);
% EXPECT: C = matrix[2 x 2]

% bsxfun broadcasting
x = randn(3, 1);
M = randn(3, 4);
R = bsxfun(@minus, M, x);
% EXPECT: R = matrix[3 x 4]

R2 = bsxfun(@times, M, 2);
% EXPECT: R2 = matrix[3 x 4]

% interpft
sig = randn(100, 1);
sig2 = interpft(sig, 256);
% EXPECT: sig2 = matrix[256 x 1]

% Window functions
w1 = hanning(64);
% EXPECT: w1 = matrix[64 x 1]

w2 = tukeywin(128);
% EXPECT: w2 = matrix[128 x 1]

% regexprep
s = regexprep('hello world', 'world', 'earth');
% EXPECT: s = string

% rmfield
st = struct('x', 1, 'y', 2, 'z', 3);
st2 = rmfield(st, 'z');
% EXPECT: st2 = struct{x: scalar, y: scalar, z: scalar}

% nchoosek
n = nchoosek(10, 3);
% EXPECT: n = scalar

% Recognized-only (no W_UNKNOWN_FUNCTION)
axes;
caxis([0 1]);
data = ncread('file.nc', 'var');
copyfile('a.m', 'b.m');
opts = optimoptions('fmincon');
info = geotiffinfo('test.tif');
