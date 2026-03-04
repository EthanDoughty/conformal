% Test: Dynamic cell assignment preserves existing element tracking.
% When c{i} = expr with non-singleton i, existing elements should be
% joined with rhs rather than destroyed.
% EXPECT: warnings = 0

function test_preserve()
    c = cell(1, 3);
    c{1} = zeros(2, 2);
    c{2} = ones(3, 3);
    c{3} = eye(4);
    % c{1} = matrix[2x2], c{2} = matrix[3x3], c{3} = matrix[4x4]
    x = c{1};
    % x should be matrix[2x2]
end
