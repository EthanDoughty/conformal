% Test: ODE solver builtins (ode45 family and odeset)
% Verifies shape rules for odeset (open struct) and ode45/ode23/ode15s/etc.
% EXPECT: warnings = 0

% odeset returns an open struct: any field access succeeds, no field warnings.
% EXPECT: opts = struct{...}
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);

% Single-output form returns the solution struct.
% EXPECT: sol = struct{...}
sol = ode45(@(t,y) -y, [0 1], 1);

% Multi-output with column-vector y0: cols(y) = rows(y0).
y0 = [1; 0; 0];
% EXPECT: t = matrix[None x 1]
% EXPECT: y = matrix[None x 3]
[t, y] = ode45(@(t,y) [y(2); -y(1); 0], [0 10], y0);

% Multi-output with scalar y0: y is column vector.
% EXPECT: t2 = matrix[None x 1]
% EXPECT: y2 = matrix[None x 1]
[t2, y2] = ode23(@(t,y) -y, [0 1], 1);

% Stiff solver variant uses the same shape rule.
% EXPECT: t3 = matrix[None x 1]
% EXPECT: y3 = matrix[None x 1]
[t3, y3] = ode15s(@(t,y) -y, [0 1], 1);
