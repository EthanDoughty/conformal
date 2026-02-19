% Test: struct() constructor with field tracking
% EXPECT: warnings = 0
% EXPECT: s1 = struct{}
% EXPECT: s2 = struct{x: scalar, y: matrix[3 x 1]}
% EXPECT: val = scalar

% Empty struct
s1 = struct();

% Struct with field names and values
s2 = struct('x', 1, 'y', zeros(3, 1));

% Field access on constructed struct
val = s2.x;
