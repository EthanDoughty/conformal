% Test: parent-directory function wins over same-name function in subdirectory.
% ws_scale is defined in tests/workspace/ws_scale.m (parent dir).
% A same-named ws_scale.m in subdir/ would be shadowed by the parent.
% TestRunner uses maxDepth=1, so it only sees ws_scale from the parent dir.
% EXPECT: warnings = 0

A = zeros(3, 4);
% ws_scale from the parent directory: B = A * factor
B = ws_scale(A, 2);
% EXPECT: B = matrix[3 x 4]
