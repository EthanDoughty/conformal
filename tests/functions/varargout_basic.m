% Test: Function with varargout called with multi-target assignment does not error
% The extra targets beyond named outputs get UnknownShape (conservative)
% EXPECT: warnings = 0
% EXPECT: a = unknown
% EXPECT: b = unknown

function varargout = myfunc(x)
    varargout{1} = x * 2;
end

[a, b] = myfunc(3);
