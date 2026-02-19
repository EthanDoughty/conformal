% Test: s.(expr) dynamic field access parses correctly, evaluates to unknown
% EXPECT: warnings = 0
% EXPECT: val = unknown

s.x = 1;
fname = 'x';
val = s.(fname);
