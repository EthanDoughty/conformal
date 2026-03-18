% Test: 5.*y should lex as NUMBER(5) + DOTOP(.*) + ID(y), not NUMBER(5.) + OP(*) + ID(y)
% EXPECT: warnings = 0

y = ones(3, 3);
x = 5.*y;
