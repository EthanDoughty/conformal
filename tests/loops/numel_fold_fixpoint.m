% Test: the Bottom-to-shape anti-monotone step the fold introduces inside a
% loop (Env.get returns Bottom pre-assignment, mapping to Unknown, the top of
% the dim lattice), plus the empty-accumulator guard from decision 3.
% EXPECT: warnings = 0
% EXPECT: names = cell[1 x 3]
% EXPECT: v = matrix[1 x 3]
% EXPECT: K = cell[0 x 0]
% EXPECT: z = matrix[1 x None]

names = {'a', 'b', 'c'};
for i = 1:numel(names)
    v = zeros(1, numel(names));
end

% Empty-accumulator guard: K is (wrongly) reported as cell[0 x 0] today, and
% the numel fold must not amplify that into a confidently wrong z = matrix[1 x 0].
K = {};
for i = 1:3
    K{end+1} = i;
end
z = zeros(1, numel(K));
