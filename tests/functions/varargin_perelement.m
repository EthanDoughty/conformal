% Test: varargin{1} returns the actual shape of the first extra argument
% When called with a 3x3 matrix as the extra arg, varargin{1} = matrix[3 x 3]
% EXPECT: warnings = 0
% EXPECT: result = matrix[3 x 3]

function y = first_extra(a, varargin)
    y = varargin{1};
end

M = zeros(3, 3);
result = first_extra(1, M);
