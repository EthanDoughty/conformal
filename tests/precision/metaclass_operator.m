% Test: metaclass operator ? should not crash the parser
% MATLAB uses ?ClassName to get the meta.class object

mc = ?MException;
x = 42;
% No crash expected
% EXPECT: x: scalar
