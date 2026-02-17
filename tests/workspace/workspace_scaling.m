% Stress Test: Cross-File Workspace Scaling (Comprehensive)
% Exercises cross-file shape inference with 26 helper functions.
% Tests: direct calls, chains (depth 5), diamond patterns, fan-out/fan-in,
%        polymorphic caching, symbolic dims, multi-return, concat, linear
%        algebra patterns, and cycle detection.
% EXPECT: warnings = 0

% ==========================================================================
% SECTION 1: Direct cross-file calls (15 helpers, basic coverage)
% ==========================================================================

A34 = zeros(3, 4);
B34 = ones(3, 4);

% Simple addition
d1 = ws_add_matrices(A34, B34);
% EXPECT: d1 = matrix[3 x 4]

% Scalar multiplication
d2 = ws_scale(A34, 2);
% EXPECT: d2 = matrix[3 x 4]

% Multi-return destructuring
[d3_t, d3_s] = ws_transform(A34);
% EXPECT: d3_t = matrix[4 x 3]
% EXPECT: d3_s = matrix[3 x 3]

% Reduction
d4 = ws_reduce(A34);
% EXPECT: d4 = matrix[1 x 4]

% Reshape
d5 = ws_reshape_safe(A34, 4, 3);
% EXPECT: d5 = matrix[4 x 3]

% Kron product
d6 = ws_kron_pair(zeros(2, 3), ones(4, 5));
% EXPECT: d6 = matrix[8 x 15]

% Outer product: (4x1) * (3x1)' = (4x1) * (1x3) = 4x3
u = zeros(4, 1);
v = ones(3, 1);
d7 = ws_outer_product(u, v);
% EXPECT: d7 = matrix[4 x 3]

% Gram matrix: A'A for 3x4 → 4x4
d8 = ws_gram(A34);
% EXPECT: d8 = matrix[4 x 4]

% Projection: v*(v'*A) for A=4x4, v=4x1 → v'*A=1x4, v*(1x4)=4x4
d9 = ws_project(eye(4), zeros(4, 1));
% EXPECT: d9 = matrix[4 x 4]

% Linear solve: inv(I3)*b = b (3x1)
d10 = ws_solve(eye(3), ones(3, 1));
% EXPECT: d10 = matrix[3 x 1]

% Symbolic constructors
d11 = ws_make_sym(5);
% EXPECT: d11 = matrix[5 x 5]

d12 = ws_make_rect(3, 7);
% EXPECT: d12 = matrix[3 x 7]

% Normalize (calls ws_scale internally)
d13 = ws_normalize(A34);
% EXPECT: d13 = matrix[3 x 4]

% Vertical concat
d14 = ws_stack_rows(A34, B34);
% EXPECT: d14 = matrix[6 x 4]

% Horizontal concat
d15 = ws_stack_cols(A34, ones(3, 2));
% EXPECT: d15 = matrix[3 x 6]

% ==========================================================================
% SECTION 2: Chained cross-file calls (depth 2-3)
% ==========================================================================

% Chain depth 2: ws_chain_add → ws_add_matrices (x2)
ch1 = ws_chain_add(A34, B34, ones(3, 4));
% EXPECT: ch1 = matrix[3 x 4]

% Pipeline depth 3: ws_pipeline → ws_transform → ws_reduce → ws_scale
ch2 = ws_pipeline(A34);
% EXPECT: ch2 = matrix[1 x 3]

% Compose depth 2: ws_compose → ws_gram (x2)
sq4 = eye(4);
ch3 = ws_compose(sq4, sq4);
% EXPECT: ch3 = matrix[4 x 4]

% Diamond top depth 3: ws_diamond_top → ws_diamond_left → ws_gram
%                                     → ws_diamond_right → ws_reduce
%                                     → ws_add_matrices
ch4 = ws_diamond_top(A34);
% EXPECT: ch4 = matrix[4 x 4]

% ==========================================================================
% SECTION 3: Deep chains (depth 4-5)
% ==========================================================================

% ws_deep_chain (depth 3): → ws_compose → ws_gram (x2) + ws_gram
deep1 = ws_deep_chain(sq4);
% EXPECT: deep1 = matrix[4 x 4]

% ws_mega_pipeline (depth 5): → ws_gram → ws_normalize → ws_scale
%                              → ws_transform → ws_reduce → ws_scale
deep2 = ws_mega_pipeline(A34);
% EXPECT: deep2 = matrix[1 x 4]

% Manual deep chain: stack → gram → reduce → add (depth 4 via caller)
deep3_a = ws_stack_cols(A34, B34);
% EXPECT: deep3_a = matrix[3 x 8]
deep3_b = ws_gram(deep3_a);
% EXPECT: deep3_b = matrix[8 x 8]
deep3_c = ws_reduce(deep3_b);
% EXPECT: deep3_c = matrix[1 x 8]
deep3_d = ws_add_matrices(deep3_c, deep3_c);
% EXPECT: deep3_d = matrix[1 x 8]

% ==========================================================================
% SECTION 4: Fan-out (one call → multiple outputs)
% ==========================================================================

% ws_fan_out returns 4 values from 3 different helpers
[fo_r, fo_g, fo_t, fo_s] = ws_fan_out(A34);
% EXPECT: fo_r = matrix[1 x 4]
% EXPECT: fo_g = matrix[4 x 4]
% EXPECT: fo_t = matrix[4 x 3]
% EXPECT: fo_s = matrix[3 x 3]

% Manual fan-out: same input A34 to 7 different external functions
fan_a = ws_reduce(A34);
% EXPECT: fan_a = matrix[1 x 4]
fan_b = ws_gram(A34);
% EXPECT: fan_b = matrix[4 x 4]
fan_c = ws_scale(A34, 3);
% EXPECT: fan_c = matrix[3 x 4]
fan_d = ws_normalize(A34);
% EXPECT: fan_d = matrix[3 x 4]
[fan_e, fan_f] = ws_transform(A34);
% EXPECT: fan_e = matrix[4 x 3]
% EXPECT: fan_f = matrix[3 x 3]
fan_g = ws_outer_product(zeros(3, 1), ones(4, 1));
% EXPECT: fan_g = matrix[3 x 4]

% ==========================================================================
% SECTION 5: Fan-in (many sources feed into one)
% ==========================================================================

fi_a = ws_make_sym(3);
% EXPECT: fi_a = matrix[3 x 3]
fi_b = ws_gram(eye(3));
% EXPECT: fi_b = matrix[3 x 3]
fi_c = ws_compose(eye(3), eye(3));
% EXPECT: fi_c = matrix[3 x 3]
fi_result = ws_add_matrices(ws_add_matrices(fi_a, fi_b), fi_c);
% EXPECT: fi_result = matrix[3 x 3]

% ==========================================================================
% SECTION 6: Diamond dependency pattern (polymorphic)
% ==========================================================================

% Diamond: caller → ws_diamond_top → ws_diamond_left → ws_gram
%                                   → ws_diamond_right → ws_reduce
%                                   → ws_add_matrices
% ws_gram(A) = A'A, ws_diamond_top combines gram result with itself

dia1 = ws_diamond_top(eye(5));
% EXPECT: dia1 = matrix[5 x 5]
dia2 = ws_diamond_top(zeros(3, 4));
% EXPECT: dia2 = matrix[4 x 4]
dia3 = ws_diamond_top(ones(6, 6));
% EXPECT: dia3 = matrix[6 x 6]

% ==========================================================================
% SECTION 7: Polymorphic caching stress
% ==========================================================================

% ws_gram called with 8 different shapes → 8 cache entries
poly_a = ws_gram(zeros(2, 3));
% EXPECT: poly_a = matrix[3 x 3]
poly_b = ws_gram(zeros(3, 2));
% EXPECT: poly_b = matrix[2 x 2]
poly_c = ws_gram(zeros(4, 4));
% EXPECT: poly_c = matrix[4 x 4]
poly_d = ws_gram(zeros(5, 1));
% EXPECT: poly_d = matrix[1 x 1]
poly_e = ws_gram(zeros(1, 5));
% EXPECT: poly_e = matrix[5 x 5]
poly_f = ws_gram(zeros(10, 10));
% EXPECT: poly_f = matrix[10 x 10]
poly_g = ws_gram(zeros(1, 1));
% EXPECT: poly_g = matrix[1 x 1]
poly_h = ws_gram(zeros(7, 3));
% EXPECT: poly_h = matrix[3 x 3]

% ws_add_matrices with 5 different shapes
poly_add1 = ws_add_matrices(zeros(1, 1), ones(1, 1));
% EXPECT: poly_add1 = matrix[1 x 1]
poly_add2 = ws_add_matrices(zeros(2, 2), ones(2, 2));
% EXPECT: poly_add2 = matrix[2 x 2]
poly_add3 = ws_add_matrices(zeros(3, 3), ones(3, 3));
% EXPECT: poly_add3 = matrix[3 x 3]
poly_add4 = ws_add_matrices(zeros(4, 4), ones(4, 4));
% EXPECT: poly_add4 = matrix[4 x 4]
poly_add5 = ws_add_matrices(zeros(5, 5), ones(5, 5));
% EXPECT: poly_add5 = matrix[5 x 5]

% ws_reduce with 4 different shapes
poly_red1 = ws_reduce(zeros(1, 10));
% EXPECT: poly_red1 = matrix[1 x 10]
poly_red2 = ws_reduce(zeros(5, 5));
% EXPECT: poly_red2 = matrix[1 x 5]
poly_red3 = ws_reduce(zeros(10, 1));
% EXPECT: poly_red3 = matrix[1 x 1]
poly_red4 = ws_reduce(zeros(3, 7));
% EXPECT: poly_red4 = matrix[1 x 7]

% ws_transform with 3 different shapes
[pt1_a, pt1_b] = ws_transform(zeros(2, 5));
% EXPECT: pt1_a = matrix[5 x 2]
% EXPECT: pt1_b = matrix[2 x 2]
[pt2_a, pt2_b] = ws_transform(zeros(5, 2));
% EXPECT: pt2_a = matrix[2 x 5]
% EXPECT: pt2_b = matrix[5 x 5]
[pt3_a, pt3_b] = ws_transform(eye(6));
% EXPECT: pt3_a = matrix[6 x 6]
% EXPECT: pt3_b = matrix[6 x 6]

% ==========================================================================
% SECTION 8: Symbolic dimension propagation across file boundaries
% ==========================================================================

% Concrete values through symbolic constructors
sym1 = ws_make_sym(3);
% EXPECT: sym1 = matrix[3 x 3]
sym2 = ws_make_sym(10);
% EXPECT: sym2 = matrix[10 x 10]

% Rectangular symbolic constructors
sym3 = ws_make_rect(4, 6);
% EXPECT: sym3 = matrix[4 x 6]
sym4 = ws_make_rect(2, 8);
% EXPECT: sym4 = matrix[2 x 8]

% Symbolic through gram: ws_gram(3x5) → 5x5
sym5 = ws_gram(ws_make_rect(3, 5));
% EXPECT: sym5 = matrix[5 x 5]

% Symbolic through pipeline (4 cross-file hops)
sym6 = ws_pipeline(ws_make_rect(4, 6));
% EXPECT: sym6 = matrix[1 x 4]

% Symbolic through compose (calls ws_gram twice)
sym7 = ws_compose(ws_make_rect(3, 5), ws_make_rect(3, 5));
% EXPECT: sym7 = matrix[5 x 5]

% ==========================================================================
% SECTION 9: Cross-file concat patterns
% ==========================================================================

% Vertical stack: (3x5) ; (4x5) = 7x5
cat1 = ws_stack_rows(zeros(3, 5), ones(4, 5));
% EXPECT: cat1 = matrix[7 x 5]

% Horizontal stack: (3x5) , (3x4) = 3x9
cat2 = ws_stack_cols(zeros(3, 5), ones(3, 4));
% EXPECT: cat2 = matrix[3 x 9]

% Chained vertical stack: ((2x3);(2x3));(2x3) = 6x3
cat3 = ws_stack_rows(ws_stack_rows(zeros(2, 3), ones(2, 3)), ones(2, 3));
% EXPECT: cat3 = matrix[6 x 3]

% Stack then gram: stack (3x4);(2x4) = 5x4, gram = 4x4
cat4 = ws_gram(ws_stack_rows(zeros(3, 4), ones(2, 4)));
% EXPECT: cat4 = matrix[4 x 4]

% ==========================================================================
% SECTION 10: Linear algebra patterns (realistic use cases)
% ==========================================================================

% Gram of tall matrix: (5x3)'(5x3) = 3x3
A_ls = zeros(5, 3);
b_ls = ones(5, 1);
AtA = ws_gram(A_ls);
% EXPECT: AtA = matrix[3 x 3]

% Projection with intentional dim mismatch (v is 3x1 but A is 5x3)
% Inside ws_project: v'*A = (1x3)*(5x3) → inner dim 3!=5 → unknown
% Cross-file body warnings are suppressed, result becomes unknown
Atb = ws_project(A_ls, zeros(3, 1));
% EXPECT: Atb = unknown

% Solve: inv(3x3) * (3x1) = 3x1
x_ls = ws_solve(AtA, zeros(3, 1));
% EXPECT: x_ls = matrix[3 x 1]

% Correct projection: v*(v'*A) where v=4x1, A=4x4
% v'=1x4, v'*A=1x4, v*(1x4)=4x4
proj_v = ws_project(eye(4), zeros(4, 1));
% EXPECT: proj_v = matrix[4 x 4]

% Block matrix: [[I3, 0]; [0, I3]] = 6x6
blk_top = ws_stack_cols(eye(3), zeros(3, 3));
% EXPECT: blk_top = matrix[3 x 6]
blk_bot = ws_stack_cols(zeros(3, 3), eye(3));
% EXPECT: blk_bot = matrix[3 x 6]
blk_full = ws_stack_rows(blk_top, blk_bot);
% EXPECT: blk_full = matrix[6 x 6]

% ==========================================================================
% SECTION 11: Cycle detection
% ==========================================================================

% ws_recursive_a → ws_recursive_b → ws_recursive_a (cycle → unknown)
cyc1 = ws_recursive_a(zeros(3, 3));
% EXPECT: cyc1 = unknown
cyc2 = ws_recursive_a(eye(5));
% EXPECT: cyc2 = unknown
cyc3 = ws_recursive_b(ones(2, 2));
% EXPECT: cyc3 = unknown

% ==========================================================================
% SECTION 12: High-volume sequential calls (18 steps, cache stress)
% ==========================================================================

seq1 = ws_add_matrices(zeros(3, 3), ones(3, 3));
% EXPECT: seq1 = matrix[3 x 3]
seq2 = ws_scale(seq1, 2);
% EXPECT: seq2 = matrix[3 x 3]
seq3 = ws_gram(seq2);
% EXPECT: seq3 = matrix[3 x 3]
[seq4, seq5] = ws_transform(seq3);
% EXPECT: seq4 = matrix[3 x 3]
% EXPECT: seq5 = matrix[3 x 3]
seq6 = ws_reduce(seq4);
% EXPECT: seq6 = matrix[1 x 3]
seq7 = ws_normalize(seq6);
% EXPECT: seq7 = matrix[1 x 3]
seq8 = ws_add_matrices(seq7, seq7);
% EXPECT: seq8 = matrix[1 x 3]
seq9 = ws_compose(seq8, seq8);
% EXPECT: seq9 = matrix[3 x 3]
seq10 = ws_deep_chain(seq9);
% EXPECT: seq10 = matrix[3 x 3]
seq11 = ws_reduce(seq10);
% EXPECT: seq11 = matrix[1 x 3]
seq12 = ws_stack_cols(seq11, seq11);
% EXPECT: seq12 = matrix[1 x 6]
seq13 = ws_gram(seq12);
% EXPECT: seq13 = matrix[6 x 6]
seq14 = ws_reduce(seq13);
% EXPECT: seq14 = matrix[1 x 6]
seq15 = ws_add_matrices(seq14, seq14);
% EXPECT: seq15 = matrix[1 x 6]
seq16 = ws_diamond_top(zeros(4, 4));
% EXPECT: seq16 = matrix[4 x 4]
seq17 = ws_mega_pipeline(zeros(4, 4));
% EXPECT: seq17 = matrix[1 x 4]

% ==========================================================================
% SECTION 13: Mixed shapes through same deep call paths
% ==========================================================================

% ws_pipeline with 3 different input shapes
mix1 = ws_pipeline(eye(5));
% EXPECT: mix1 = matrix[1 x 5]
mix2 = ws_pipeline(zeros(2, 8));
% EXPECT: mix2 = matrix[1 x 2]
mix3 = ws_pipeline(zeros(8, 2));
% EXPECT: mix3 = matrix[1 x 8]

% ws_mega_pipeline with 3 different shapes (5-deep polymorphic)
mix4 = ws_mega_pipeline(eye(3));
% EXPECT: mix4 = matrix[1 x 3]
mix5 = ws_mega_pipeline(zeros(5, 3));
% EXPECT: mix5 = matrix[1 x 3]
mix6 = ws_mega_pipeline(zeros(2, 7));
% EXPECT: mix6 = matrix[1 x 7]

% ws_deep_chain with 3 different shapes (3-deep polymorphic)
mix7 = ws_deep_chain(eye(3));
% EXPECT: mix7 = matrix[3 x 3]
mix8 = ws_deep_chain(zeros(4, 6));
% EXPECT: mix8 = matrix[6 x 6]
mix9 = ws_deep_chain(zeros(6, 4));
% EXPECT: mix9 = matrix[4 x 4]
