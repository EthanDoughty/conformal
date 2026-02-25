% Test: Struct field accumulation pattern
% s.data = [s.data; row] detected and refined
% EXPECT: warnings = 0
% EXPECT_FIXPOINT: s = struct{data: matrix[12 x 3]}

s.data = zeros(2, 3);
for i = 1:10
    s.data = [s.data; zeros(1, 3)];
end
