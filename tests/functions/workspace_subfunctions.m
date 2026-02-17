A = zeros(3, 4);
B = workspace_subfunctions_helper(A);
% EXPECT: B = matrix[3 x 4]
% EXPECT: warnings = 0
