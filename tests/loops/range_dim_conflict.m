% Test: Range dimension conflict detection
% Loop unconditionally overwrites A to 5x4 each iteration (from initial 3x4)
% Fixpoint widens row dim to Range(3,5); B has 7 rows: disjoint => warning
% Default: single pass gives A=5x4 rows vs B=7x4 rows: concrete conflict
% EXPECT: warnings = 2
% EXPECT_FIXPOINT: warnings = 1

A = zeros(3, 4);
for i = 1:5
    A = zeros(5, 4);
end
B = zeros(7, 4);
C = A + B;
