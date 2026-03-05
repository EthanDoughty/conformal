% Test: feval(dynamicVar, args) returns unknown shape (conservative)
% EXPECT: warnings = 0

function y = ws_feval_dynamic(x, name)
    y = feval(name, x);
    % y is unknown because 'name' is a dynamic expression
end
