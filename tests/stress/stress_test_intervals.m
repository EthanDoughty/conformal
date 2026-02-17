% Stress Test: Interval Analysis Edge Cases
% Boundary conditions, conditional refinement, interactions with other systems.
% EXPECT: warnings = 5

% ==========================================================================
% Boundary: zeros(0, 3) — zero dimension (valid, no warning)
% ==========================================================================
empty_mat = zeros(0, 3);
% EXPECT: empty_mat = matrix[0 x 3]

% ==========================================================================
% Boundary: exact index match (no OOB)
% ==========================================================================
arr5 = zeros(5, 5);
safe_idx = arr5(5, 5);
% No OOB warning — index exactly at boundary
% EXPECT: safe_idx = scalar

% ==========================================================================
% For-loop index with exact boundary (safe)
% ==========================================================================
arr_loop = zeros(5, 3);
for i = 1:5
    val_i = arr_loop(i, 1);
end

% ==========================================================================
% For-loop index OOB — exceeds dimension
% ==========================================================================
small = zeros(3, 3);
for j = 1:5
    oob_j = small(j, 1);
end

% ==========================================================================
% Division by zero: reassignment to zero
% ==========================================================================
x = 5;
x = 0;
div_zero = 1 / x;

% ==========================================================================
% Division guarded by ~= 0 check
% ==========================================================================
function safe = guarded_div(d)
    if d ~= 0
        safe = 1 / d;
    else
        safe = 0;
    end
end

gd = guarded_div(5);
% EXPECT: gd = scalar

% ==========================================================================
% Compound condition narrowing: x > 0 && x < 10
% ==========================================================================
function r = compound_check(x)
    arr = zeros(10, 1);
    if x > 0 && x < 10
        r = arr(x, 1);
    else
        r = 0;
    end
end

cc = compound_check(5);
% EXPECT: cc = scalar

% ==========================================================================
% Nested condition narrowing
% ==========================================================================
function r2 = nested_check(x)
    arr2 = zeros(10, 1);
    if x > 0
        if x < 10
            r2 = arr2(x, 1);
        else
            r2 = 0;
        end
    else
        r2 = 0;
    end
end

nc = nested_check(5);
% EXPECT: nc = scalar

% ==========================================================================
% Interval propagation through scalar arithmetic
% ==========================================================================
function r3 = arith_interval(x)
    arr3 = zeros(10, 1);
    y = x + 2;
    if x > 0 && x < 5
        r3 = arr3(y, 1);
    else
        r3 = 0;
    end
end

ai = arith_interval(3);
% EXPECT: ai = scalar

% ==========================================================================
% W_POSSIBLY_NEGATIVE_DIM with symbolic dimension
% ==========================================================================
function M = maybe_neg(n)
    M = zeros(n - 5, 3);
end

mn_result = maybe_neg(10);

% ==========================================================================
% W_INDEX_OUT_OF_BOUNDS: definite OOB
% ==========================================================================
fix5 = zeros(5, 5);
definite_oob = fix5(6, 1);

% ==========================================================================
% W_DIVISION_BY_ZERO: literal zero divisor
% ==========================================================================
dz_literal = 10 / 0;

% ==========================================================================
% Interval in else branch: if x > 5 → else: x <= 5
% ==========================================================================
function r4 = else_refine(x)
    arr4 = zeros(5, 1);
    if x > 5
        r4 = 0;
    else
        % x <= 5 here — accessing arr4(x, 1) should be safe if x > 0
        if x > 0
            r4 = arr4(x, 1);
        else
            r4 = 0;
        end
    end
end

er = else_refine(3);
% EXPECT: er = scalar

% ==========================================================================
% For-loop symbolic bounds: interval is [1, n]
% ==========================================================================
function r5 = symbolic_loop(n)
    A5 = zeros(n, 1);
    for idx = 1:n
        r5 = A5(idx, 1);
    end
end

sl = symbolic_loop(10);
% EXPECT: sl = scalar

% ==========================================================================
% Zeros with negative literal dimension
% ==========================================================================
neg_dim = zeros(-1, 3);
% EXPECT: neg_dim = matrix[None x 3]
