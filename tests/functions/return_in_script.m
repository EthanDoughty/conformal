% Test: Return in script context stops analysis (valid MATLAB — exits script)

A = zeros(3, 3);
return; % EXPECT_WARNING: W_RETURN_OUTSIDE_FUNCTION
B = A * A;
