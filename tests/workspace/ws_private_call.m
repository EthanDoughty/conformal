% Test: function in private/ resolves for callers in the same directory
% EXPECT: warnings = 0

A = zeros(3, 4);
B = ws_private_helper(A);
% EXPECT: B = matrix[3 x 4]
