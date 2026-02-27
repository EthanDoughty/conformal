% Test: W_CODER_VARIABLE_SIZE fires for variable with Unknown dimension.
% find() returns a variable-size row vector: matrix[1 x Unknown].
% MODE: coder

% EXPECT: warnings = 1

idx = find([1 0 1]);
