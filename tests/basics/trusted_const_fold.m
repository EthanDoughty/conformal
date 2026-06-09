% Test: Gap B conservative redesign -- trusted-constant store for range folding.
% Covers all MUST-FOLD, MUST-NOT-FOLD, overflow, and regression cases from TASK.md.
%
% MUST-FOLD cases: trustedConsts populated at depth 0, range folds to concrete length.
% MUST-NOT-FOLD cases: 1xNone, NO dimension warning (soundness).
% OVERFLOW cases: no false dimension warning (Unknown is safe).

% --- MUST-FOLD: forum case ---
% Lambda=0.2 at depth 0; 0:Lambda/400:Lambda/4 = 0:0.0005:0.05 -> 1x101.
Lambda = 0.2;
zi = 0.05;
z = 0:Lambda/400:Lambda/4;
% EXPECT: z = matrix[1 x 101]
% Forum mismatch: d.^2+(z-zi) where d is 1x5 -> shape mismatch.
d = zeros(1, 5);
forum_expr = d.^2 + (z - zi);  % EXPECT_WARNING: W_ELEMENTWISE_MISMATCH

% --- MUST-FOLD: integer constant ---
% n=10 at depth 0; 0:n/4:n = 0:2.5:10 -> 1x5.
n = 10;
r_int = 0:n/4:n;
% EXPECT: r_int = matrix[1 x 5]

% --- MUST-FOLD: chained constants ---
% a=0.5, b=a/2=0.25 both in trustedConsts at depth 0; 0:b:1 = 0:0.25:1 -> 1x5.
a_ch = 0.5;
b_ch = a_ch / 2;
r_chain = 0:b_ch:1;
% EXPECT: r_chain = matrix[1 x 5]

% --- MUST-FOLD: loop does not touch dx ---
% dx=0.1 at depth 0; loop assigns k and y but not dx; dx remains in trustedConsts.
dx = 0.1;
for k = 1:3
    y_loop = k;
end
r_dx = 0:dx:1;
% EXPECT: r_dx = matrix[1 x 11]

% --- MUST-NOT-FOLD: both branches assign L (removed from trustedConsts) ---
% After the if/else, L is not in trustedConsts -> 1xNone.
if cond_a
    L_both = 0.2;
else
    L_both = 0.5;
end
r_both = 0:L_both/400:L_both/4;
% EXPECT: r_both = matrix[1 x None]

% --- MUST-NOT-FOLD: only one branch assigns L ---
% L_one is never defined on the else path; removed from trustedConsts by if-branch assignment.
if cond_b
    L_one = 0.2;
end
r_one = 0:L_one/400:L_one/4;
% EXPECT: r_one = matrix[1 x None]

% --- MUST-NOT-FOLD: reassignment at depth 0 overwrites with non-foldable value ---
% L_rand is reassigned to rand(); trustedConsts entry cleared for that name (non-foldable RHS).
L_rand = 0.2;
L_rand = rand();
r_rand = 0:L_rand:1;
% EXPECT: r_rand = matrix[1 x None]

% --- Reassignment at depth 0 to a different constant ---
% TASK.md notes "(reassign)" for L=0.2; L=0.5 -- the last depth-0 assignment wins:
% L_re = 0.5 replaces 0.2 in trustedConsts -> 0:0.5/400:0.5/4 = 0:0.00125:0.125 -> 1x101.
% This is CORRECT: the trusted store always reflects the LAST unconditional assignment.
L_re = 0.2;
L_re = 0.5;
r_re = 0:L_re/400:L_re/4;
% EXPECT: r_re = matrix[1 x 101]

% --- MUST-NOT-FOLD: loop modifies h -> removed from trustedConsts at loop entry ---
h_loop0 = 0.5;
for k2 = 1:0
    h_loop0 = 0.25;
end
r_hloop0 = 0:h_loop0:1;
% EXPECT: r_hloop0 = matrix[1 x None]

% --- MUST-NOT-FOLD: loop body modifies h ---
h_loop = 0.5;
for k3 = 1:3
    h_loop = h_loop / 2;
end
r_hloop = 0:h_loop:1;
% EXPECT: r_hloop = matrix[1 x None]

% --- MUST-NOT-FOLD: clear n removes it from trustedConsts ---
n_clr = 10;
clear n_clr
r_nclr = 0:n_clr/4:n_clr;
% EXPECT: r_nclr = matrix[1 x None]

% --- MUST-NOT-FOLD: clear L ---
L_clr = 0.2;
clear L_clr
r_lclr = 0:L_clr/400:L_clr/4;
% EXPECT: r_lclr = matrix[1 x None]

% --- MUST-NOT-FOLD: clearvars (drops all trusted constants) ---
L_cv = 0.2;
clearvars
r_cv = 0:L_cv/400:L_cv/4;
% EXPECT: r_cv = matrix[1 x None]

% --- MUST-NOT-FOLD: clear all ---
L_ca = 0.2;
clear all
r_ca = 0:L_ca/400:L_ca/4;
% EXPECT: r_ca = matrix[1 x None]

% --- MUST-NOT-FOLD: indexed assignment L(1)=9 removes L from trustedConsts ---
L_idx = 0.2;
L_idx(1) = 9;
r_idx = 0:L_idx/400:L_idx/4;
% EXPECT: r_idx = matrix[1 x None]

% --- MUST-NOT-FOLD: struct assignment L.f=5 removes L from trustedConsts ---
L_sf = 0.2;
L_sf.f = 5;
r_sf = 0:L_sf/400:L_sf/4;
% EXPECT: r_sf = matrix[1 x None]

% --- MUST-NOT-FOLD: clear inside if-branch (depth > 0) ---
% Even though L_cif was set at depth 0, clear inside a branch drops all trusted consts.
L_cif = 0.2;
if cond_c
    clear L_cif
end
r_cif = 0:L_cif/400:L_cif/4;
% EXPECT: r_cif = matrix[1 x None]

% --- OVERFLOW: 0:1:2147483647 -> no false dimension warning ---
% This stepped range has ~2^31 elements; steppedRangeLengthFloat returns Unknown.
r_overflow1 = 0:1:2147483647;
% EXPECT: r_overflow1 = matrix[1 x None]

% --- OVERFLOW: 0:0.0001:1000000 -> no false dimension warning ---
% This range has 10^10 elements; steppedRangeLengthFloat returns Unknown.
r_overflow2 = 0:0.0001:1000000;
% EXPECT: r_overflow2 = matrix[1 x None]
