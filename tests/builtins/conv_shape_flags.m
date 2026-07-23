% Test: conv honors its trailing shape flag ('full'/'same'/'valid')
% EXPECT: warnings = 0

x = ones(1, 1000);
h = ones(1, 10);

% No flag -> 'full', length m+n-1
c_full = conv(x, h);
% EXPECT: c_full = matrix[1 x 1009]

% 'same' -> central part, same size as the first argument
c_same = conv(x, h, 'same');
% EXPECT: c_same = matrix[1 x 1000]

% 'valid' -> max(m-n+1, 0)
c_valid = conv(x, h, 'valid');
% EXPECT: c_valid = matrix[1 x 991]

% Live false-positive repro: an FIR-smoothing residual must NOT draw a
% shape-mismatch warning now that 'same' is honored.
r = x - conv(x, h, 'same');
% EXPECT: r = matrix[1 x 1000]

% Column-vector orientation is preserved by 'same' returning args[0] verbatim
xcol = ones(20, 1);
hcol = ones(4, 1);
c_same_col = conv(xcol, hcol, 'same');
% EXPECT: c_same_col = matrix[20 x 1]

% Symbolic 'valid': max() is not faked over symbolic dims
c_valid_sym = conv(zeros(1, n), zeros(1, m), 'valid');
% EXPECT: c_valid_sym = unknown
