% Test: chained struct-index-struct assignment (s.field(i).field = expr)
% These patterns produce OpaqueStmt without the parser fix.
% EXPECT: warnings = 0

% Basic: s.field(i).field = expr
hdr.wfs(1).num_sam = 100;
% EXPECT: hdr = struct{wfs: struct{num_sam: scalar}}

% Multiple prefix fields
data.params.list(1).name = 'test';
% EXPECT: data = struct{params: struct{list: struct{name: string}}}

% Multiple suffix fields
cfg.items(1).opts.verbose = 1;
% EXPECT: cfg = struct{items: struct{opts: struct{verbose: scalar}}}

% Curly variant: s.field{i}.field = expr
cs.cells{1}.value = 42;
% EXPECT: cs = struct{cells: struct{value: scalar}}

% Variable index
s.arr(k).x = 0;
% EXPECT: s = struct{arr: struct{x: scalar}}
