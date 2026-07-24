% Test: soundness bails that must survive the inline numel/length/size fold.
% `control` is a positive fold (length is not shadowed here) proving the
% mechanism is active; everything else must stay at None. MATLAB local
% functions are visible file-wide, so the trailing `numel` definition also
% shadows the numel() calls above it -- that overlaps with, but does not
% weaken, the struct/Bottom bails, since each already independently bails.
% EXPECT: warnings = 0
% EXPECT: v = matrix[1 x 5]
% EXPECT: control = matrix[1 x 5]
% EXPECT: st = struct{a: scalar}
% EXPECT: zst = matrix[1 x None]
% EXPECT: A = matrix[3 x 4]
% EXPECT: zA3 = matrix[1 x None]
% EXPECT: E = matrix[3 x 0]
% EXPECT: zE = matrix[1 x None]
% EXPECT: zU = matrix[1 x None]
% EXPECT: zShadow = matrix[1 x None]

v = zeros(1, 5);
control = zeros(1, length(v));   % length is not shadowed: proves the mechanism is active

st = struct('a', 1);
zst = zeros(1, numel(st));       % struct arrays are not modeled

A = zeros(3, 4);
zA3 = zeros(1, size(A, 3));      % k >= 3 is left to the ndArraySlices path

E = zeros(3, 0);
zE = zeros(1, length(E));        % zero-extent bail: E is a pre-existing wrong shape

zU = zeros(1, numel(never_assigned));  % Bottom maps to Unknown

zShadow = zeros(1, numel(v));    % numel is shadowed below: the fold must defer to it

function n = numel(x)
    n = 99;
end
