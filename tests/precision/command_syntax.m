% Test: MATLAB command syntax (function calls without parentheses)
% The lexer must treat ' as string delimiter after known command names.

function r = test()
  x = 42;
  warning 'This is a warning message';
  error 'This has special chars: !@#$%';
  r = x + 1;
end
% EXPECT: r: 1x1
