% Test: Return in script context stops analysis
% EXPECT: warnings = 1

A = zeros(3, 3);
return;
B = A * A;
