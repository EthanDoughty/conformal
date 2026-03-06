% Test: [] is type-neutral in concatenation (no W_CONCAT_TYPE_MISMATCH)
% MATLAB: [[], struct_value] returns a 1-element struct array

s.x = 1;
s.y = 2;
arr = [];
arr = [arr s];
% No warning expected: [] is type-neutral for concatenation
