% Test: W_CODER_DYNAMIC_FIELD fires for dynamic struct field access s.(expr).
% MODE: coder

% EXPECT: warnings = 1

s.x = 1;
name = 'x';
v = s.(name);
