% Test: Stepped range as first-class IndexArg
% a:step:b in index position should be parsed as SteppedRange, not BinOp

function r = test()
  A = zeros(10, 5);
  x = A(1:2:9, :);
  r = x;
end
% EXPECT: r: 5x5
