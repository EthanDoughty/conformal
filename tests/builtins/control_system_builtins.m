% Test: Control System Toolbox builtins shape inference
% EXPECT: warnings = 0

% State-space system matrices
A = [0 1; -2 -3];       % 2x2
B = [0; 1];             % 2x1
C = [1 0];              % 1x2
Q = eye(2);             % 2x2
R = 1;                  % scalar

% LQR
K_lqr = lqr(A, B, Q, R);
% EXPECT: K_lqr = matrix[1 x 2]

[K2, S, e] = lqr(A, B, Q, R);
% EXPECT: K2 = matrix[1 x 2]
% EXPECT: S = matrix[2 x 2]
% EXPECT: e = matrix[2 x 1]

% DLQR (same shapes as lqr)
K_dlqr = dlqr(A, B, Q, R);
% EXPECT: K_dlqr = matrix[1 x 2]

% Place
K_place = place(A, B, [-1, -2]);
% EXPECT: K_place = matrix[1 x 2]

% Acker
K_acker = acker(A, B, [-1, -2]);
% EXPECT: K_acker = matrix[1 x 2]

% CARE
X_care = care(A, B, Q);
% EXPECT: X_care = matrix[2 x 2]

[X2, L, G] = care(A, B, Q);
% EXPECT: X2 = matrix[2 x 2]
% EXPECT: L = matrix[2 x 1]
% EXPECT: G = matrix[1 x 2]

% DARE
X_dare = dare(A, B, Q);
% EXPECT: X_dare = matrix[2 x 2]

% Lyapunov
X_lyap = lyap(A, Q);
% EXPECT: X_lyap = matrix[2 x 2]

X_dlyap = dlyap(A, Q);
% EXPECT: X_dlyap = matrix[2 x 2]

% Observability matrix
O = obsv(A, C);
% EXPECT: O = matrix[2 x 2]

% Controllability matrix
Co = ctrb(A, B);
% EXPECT: Co = matrix[2 x 2]

% Dot product
d = dot([1 2 3], [4 5 6]);
% EXPECT: d = scalar

% Recognized-only (no W_UNKNOWN_FUNCTION)
sys = ss(A, B, C, 0);
sys_tf = tf([1], [1 1]);
sys_d = c2d(sys, 0.01);
