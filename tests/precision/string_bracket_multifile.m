% Test: STRING tokens with bracket-like values in multi-function files
% Verifies that STRING '(' does not confuse ParsePrefix, detectEndlessFunctions,
% or RecoverToStmtBoundary when multiple functions share a file.

function r = test()
  r = helper('(');
end

function y = helper(s)
  y = ['prefix' s 'suffix'];
  z = ['(' s ')'];
  w = {'[' s ']'};
end
% EXPECT: r: unknown
