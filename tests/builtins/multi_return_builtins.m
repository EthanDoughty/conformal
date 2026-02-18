% Test: Multi-return builtin shape rules (v1.11.0)
% EXPECT: warnings = 0

% === Setup ===
A = zeros(4, 4);
B = zeros(3, 5);

% === eig ===
d = eig(A);
% EXPECT: d = matrix[4 x 1]
[V, D] = eig(A);
% EXPECT: V = matrix[4 x 4]
% EXPECT: D = matrix[4 x 4]

% === svd ===
sv = svd(B);
% EXPECT: sv = matrix[None x 1]
[U, S, Vt] = svd(B);
% EXPECT: U = matrix[3 x 3]
% EXPECT: S = matrix[3 x 5]
% EXPECT: Vt = matrix[5 x 5]

% === lu ===
[L2, U2] = lu(B);
% EXPECT: L2 = matrix[3 x 3]
% EXPECT: U2 = matrix[3 x 5]
[L3, U3, P3] = lu(B);
% EXPECT: L3 = matrix[3 x 5]
% EXPECT: U3 = matrix[5 x 5]
% EXPECT: P3 = matrix[3 x 3]

% === qr ===
[Q, R] = qr(B);
% EXPECT: Q = matrix[3 x 3]
% EXPECT: R = matrix[3 x 5]

% === chol ===
[Rc, p] = chol(A);
% EXPECT: Rc = matrix[4 x 4]
% EXPECT: p = scalar

% === size ===
[m, n] = size(B);
% EXPECT: m = scalar
% EXPECT: n = scalar

% === sort ===
v = zeros(5, 1);
[sv2, si] = sort(v);
% EXPECT: sv2 = matrix[5 x 1]
% EXPECT: si = matrix[5 x 1]
[sm, smi] = sort(B);
% EXPECT: sm = matrix[3 x 5]
% EXPECT: smi = matrix[3 x 5]

% === find ===
[fr, fc] = find(A);
% EXPECT: fr = matrix[1 x None]
% EXPECT: fc = matrix[1 x None]
[fr3, fc3, fv3] = find(A);
% EXPECT: fr3 = matrix[1 x None]
% EXPECT: fc3 = matrix[1 x None]
% EXPECT: fv3 = matrix[1 x None]

% === unique ===
[uu, uia] = unique(v);
% EXPECT: uu = matrix[1 x None]
% EXPECT: uia = matrix[None x 1]
[uu3, uia3, uic3] = unique(v);
% EXPECT: uu3 = matrix[1 x None]
% EXPECT: uia3 = matrix[None x 1]
% EXPECT: uic3 = matrix[None x 1]

% === min/max ===
[Mn, Mn_i] = min(B);
% EXPECT: Mn = matrix[1 x 5]
% EXPECT: Mn_i = matrix[1 x 5]
[Mx, Mx_i] = max(B);
% EXPECT: Mx = matrix[1 x 5]
% EXPECT: Mx_i = matrix[1 x 5]
[ms, msi] = min(v);
% EXPECT: ms = matrix[1 x 1]
% EXPECT: msi = matrix[1 x 1]

% === Symbolic dimensions ===
C = zeros(n, n);
[Vs, Ds] = eig(C);
% EXPECT: Vs = matrix[n x n]
% EXPECT: Ds = matrix[n x n]
