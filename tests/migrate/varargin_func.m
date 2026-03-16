function result = varargin_func(x, varargin)
    n = nargin;
    if nargin > 1
        y = varargin{1};
    else
        y = 0;
    end
    result = x + y;
end
