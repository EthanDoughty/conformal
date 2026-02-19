% Test: A(i).field = val chained indexed struct assignment parses correctly (no recovery OpaqueStmt)
% EXPECT: warnings = 1
% EXPECT: s = unknown

s = struct();
s(1).name = 'hello';
s(1).value = 42;
s(2).name = 'world';
