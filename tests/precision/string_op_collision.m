% Test: string values that collide with operator names in precedence table
% Single-quoted '*' inside a matrix literal must not be confused with multiply
% This catches the bug where STRING('*') matched the '*' operator precedence entry

function r = test()
  T = 'hello';
  x = [T '*'];
  y = isempty(dir([T '*']));
  if isempty(dir([T '*']))
    r = 1;
  end
end
% EXPECT: T: string
% EXPECT: x: matrix[1 x 2]
