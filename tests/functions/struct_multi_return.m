% Test multi-return assignment to struct fields

% Basic struct multi-return from user-defined function
function [a, b] = get_pair()
    a = [1 2 3];
    b = [4; 5; 6];
end

[s.x, s.y] = get_pair();
% EXPECT: s = struct{x: matrix[1 x 3], y: matrix[3 x 1]}

% Multi-return from builtin size() with struct targets
A = rand(3, 4);
[info.rows, info.cols] = size(A);
% EXPECT: info = struct{cols: scalar, rows: scalar}

% Mixed: plain target and struct target
[plain, t.field] = get_pair();
% EXPECT: plain = matrix[1 x 3]
% EXPECT: t = struct{field: matrix[3 x 1]}

% Struct targets preserve existing fields
existing = struct();
existing.z = 99;
[existing.x, existing.y] = get_pair();
% EXPECT: existing = struct{x: matrix[1 x 3], y: matrix[3 x 1], z: scalar}

% Struct target with ~ placeholder
[~, only.second] = get_pair();
% EXPECT: only = struct{second: matrix[3 x 1]}

% Struct targets from unknown function
[u.a, u.b] = unknown_func();
% EXPECT: u = struct{a: unknown, b: unknown}

% Reading struct fields after multi-return assignment
r1 = s.x;
r2 = s.y;
% EXPECT: r1 = matrix[1 x 3]
% EXPECT: r2 = matrix[3 x 1]

% EXPECT: warnings = 1
