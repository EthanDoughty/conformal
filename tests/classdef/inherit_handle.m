% Test: inheriting from handle (known builtin base) works
% EXPECT: warnings = 0

h = HandleChild(42);
v = h.value;
