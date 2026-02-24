% Test: Signal Processing Toolbox builtins shape inference
% EXPECT: warnings = 0

x = randn(100, 1);
b = [1 0.5 0.25];
a = [1 -0.8];

% filter
y = filter(b, a, x);
% EXPECT: y = matrix[100 x 1]

% filtfilt
y2 = filtfilt(b, a, x);
% EXPECT: y2 = matrix[100 x 1]

% conv -- symbolic length
c = conv(x, [1; -1]);
% EXPECT: c = matrix[101 x 1]

% conv with known lengths
c2 = conv([1 2 3], [4 5]);
% EXPECT: c2 = matrix[1 x 4]

% xcorr
r = xcorr(x);
% EXPECT: r = matrix[None x 1]

% window functions
w1 = hamming(64);
% EXPECT: w1 = matrix[64 x 1]

w2 = hann(128);
% EXPECT: w2 = matrix[128 x 1]

w3 = blackman(256);
% EXPECT: w3 = matrix[256 x 1]

w4 = kaiser(64, 5);
% EXPECT: w4 = matrix[64 x 1]

% butter
[b_filt, a_filt] = butter(4, 0.5);
% EXPECT: b_filt = matrix[1 x 5]
% EXPECT: a_filt = matrix[1 x 5]

% recognized-only (no W_UNKNOWN_FUNCTION)
S = spectrogram(x);
P = pwelch(x);
peaks = findpeaks(x);
