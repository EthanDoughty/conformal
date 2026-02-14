% Test: Mixed file — function definition followed by script code
% EXPECT: warnings = 0
% EXPECT: X = matrix[5 x 5]
% EXPECT: Y = matrix[5 x 5]

function out = identity_wrapper(in)
    out = in;
end

X = zeros(5, 5);
Y = identity_wrapper(X);
