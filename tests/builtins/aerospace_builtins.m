% Test: Aerospace Toolbox builtins shape inference
% EXPECT: warnings = 0

% DCM from Euler angles
C = angle2dcm(0.1, 0.2, 0.3);
% EXPECT: C = matrix[3 x 3]

% DCM from quaternion
q = [1 0 0 0];
C2 = quat2dcm(q);
% EXPECT: C2 = matrix[3 x 3]

% Quaternion from DCM
q2 = dcm2quat(C);
% EXPECT: q2 = matrix[1 x 4]

% Quaternion multiply
q3 = quatmultiply(q, q2);
% EXPECT: q3 = matrix[1 x 4]

% Quaternion operations (passthrough)
qc = quatconj(q);
% EXPECT: qc = matrix[1 x 4]

qi = quatinv(q);
% EXPECT: qi = matrix[1 x 4]

qn = quatnormalize(q);
% EXPECT: qn = matrix[1 x 4]

% Quaternion norm
n = quatnorm(q);
% EXPECT: n = scalar

% dcm2angle multi-return
[r1, r2, r3] = dcm2angle(C);
% EXPECT: r1 = scalar
% EXPECT: r2 = scalar
% EXPECT: r3 = scalar

% recognized-only
p = lla2ecef([37.7749, -122.4194, 0]);
rho = atmoscoesa(1000);
g = gravitywgs84(1000, 37.7749);
