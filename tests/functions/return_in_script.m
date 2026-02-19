% Test: Return in script context stops analysis (valid MATLAB — exits script)
% EXPECT: warnings = 0

A = zeros(3, 3);
return;
B = A * A;
