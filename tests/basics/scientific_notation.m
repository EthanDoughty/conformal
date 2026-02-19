% Test: scientific notation in numeric literals
% 125e6, 1e-10, 2.5e3, etc.

% EXPECT: warnings = 0

fc = 195e6;
fs = 250e6;
clk = 1e9;
tiny = 1e-10;
ratio = 2.5e3;
bigE = 1E6;
plusE = 1e+3;
