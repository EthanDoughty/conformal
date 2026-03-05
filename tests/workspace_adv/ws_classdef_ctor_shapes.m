% Test: cross-file classdef constructor should propagate property shapes
% EXPECT: warnings = 1
% EXPECT: h = struct{A: matrix[3 x 4]}
h = XFileClass(zeros(3, 4));
y = h.apply(ones(2, 1));
