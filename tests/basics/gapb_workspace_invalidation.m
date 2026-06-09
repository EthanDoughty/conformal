% Regression: workspace-invalidating operations
% must drop trustedConsts so range folds do not use stale values.
% EXPECT: warnings = 0

% Bug 3 / Bug 14: load invalidates trusted constants.
L_ld = 0.2;
load params.mat
w_ld = 0:L_ld/400:L_ld/4;
q_ld = zeros(1, 50);
bad_ld = w_ld .* q_ld;
% EXPECT: w_ld = matrix[1 x None]
% EXPECT: bad_ld = matrix[1 x None]

% Bug 7: clear -regexp ^L parses as a BinOp but must still drop trustedConsts.
L7 = 0.2;
clear -regexp ^L7
a7 = 0:L7/400:L7/4;
b7 = zeros(1, 5);
c7 = a7 .* b7;
% EXPECT: a7 = matrix[1 x None]
% EXPECT: c7 = matrix[1 x None]

% Bug 16: eval/evalin reassigns variables; trusted constants must be dropped.
L16 = 0.2;
eval('L16 = 0.5');
r16 = 0:L16/400:0.05;
s16 = zeros(1, 41);
chk16 = r16 + s16;
% EXPECT: r16 = matrix[1 x None]
% EXPECT: chk16 = matrix[1 x None]
