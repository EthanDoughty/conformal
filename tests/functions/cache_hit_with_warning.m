% Test: Cache hit must replay warnings at each call site
% EXPECT: warnings = 2
% EXPECT: B = unknown
% EXPECT: C = unknown

function y = bad_multiply(x)
    y = x * x;
end

A = zeros(3, 4);
B = bad_multiply(A);
C = bad_multiply(A);
