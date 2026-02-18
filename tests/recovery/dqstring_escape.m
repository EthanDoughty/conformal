% Test: double-quoted strings with "" escaping
% EXPECT: warnings = 0

x = "hello";
% EXPECT: x = string

y = "he said ""hello""";
% EXPECT: y = string

z = "";
% EXPECT: z = string

w = "it's fine";
% EXPECT: w = string
