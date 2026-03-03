% Test: nargin guard before accessing varargin{1}
% Analysis joins both branches (varargin{1}=matrix and a=scalar), giving unknown at the join
% The key test here is: no arg-count warning, no crash accessing varargin{1}
% EXPECT: warnings = 0
% EXPECT: result = unknown

function y = maybe_use(a, varargin)
    if nargin > 1
        y = varargin{1};
    else
        y = a;
    end
end

M = zeros(4, 4);
result = maybe_use(1, M);
