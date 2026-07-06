% Comma-separated statement chains: recovery must stop at a depth-0 comma so
% each segment parses (or recovers) on its own. Before, the whole line was one
% opaque statement and x stayed unbound.

hold off, x = ones(2, 2), y = x * ones(2, 1);

% EXPECT: warnings = 0
% EXPECT: x = matrix[2 x 2]
% EXPECT: y = matrix[2 x 1]
