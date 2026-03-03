% Test: Function with varargin called with extra args does not emit arg-count warning
% nargin is total argument count including extras
% varargin is bound as a cell with the correct element count
% EXPECT: warnings = 0
% EXPECT: result = scalar

function y = sum2plus(a, b, varargin)
    y = a + b + nargin;
end

result = sum2plus(1, 2, 10, 20);
