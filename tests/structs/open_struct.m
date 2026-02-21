% Test open struct lattice behavior

% Open struct from unknown base (struct field assign on unknown variable)
s = some_func();
s.x = [1 2 3];
% EXPECT: s = struct{x: matrix[1 x 3], ...}

% Reading tracked field works normally
y = s.x;
% EXPECT: y = matrix[1 x 3]

% Reading untracked field returns unknown silently (no warning for field access)
z = s.untracked;
% EXPECT: z = unknown

% Multiple field assignments on unknown base (open flag is preserved)
t = another_func();
t.a = 5;
t.b = [1; 2; 3];
% EXPECT: t = struct{a: scalar, b: matrix[3 x 1], ...}

% Join: closed + open = open
% x is in closed only: join(scalar, unknown) = unknown (open side default is unknown)
% y is in open only: join(bottom, scalar) = scalar (closed side default is bottom)
if cond
    r = struct();
    r.x = 1;
else
    r = load_func();
    r.y = 2;
end
% EXPECT: r = struct{x: unknown, y: scalar, ...}

% Join: open + open = open
% x is in both branches with different concrete shapes: join = unknown
if cond
    p = load_a();
    p.x = [1 2];
else
    p = load_b();
    p.x = 99;
end
% EXPECT: p = struct{x: unknown, ...}

% 5 W_UNKNOWN_FUNCTION warnings from: some_func, another_func, load_func, load_a, load_b
% (strict-only codes but tests receive unfiltered warnings)
% EXPECT: warnings = 5
