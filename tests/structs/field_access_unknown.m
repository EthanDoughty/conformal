% Test: field access on unknown base should not warn
% Bug B3: data = load(...); data.field triggered W_FIELD_ACCESS_NON_STRUCT
% because unknown is lattice top (could be anything including struct)
% Also: x = []; x.field = val should not warn (empty matrix -> struct promotion)

% EXPECT: warnings = 0

function test()
    data = load('file.mat');
    x = data.values;
    y = data.timestamps;
    data.newfield = 42;

    % Empty matrix promotion to struct
    opts = [];
    opts.color = 'red';
    opts.size = 10;
end
