% Test: ! shell escape operator
% MATLAB's ! sends the rest of the line to the operating system.
% The lexer should treat it as a no-op, not crash.

function r = test()
  x = 42;
  !rm -f tempfile.txt
  !echo hello world
  r = x + 1;
end
% EXPECT: r: 1x1
