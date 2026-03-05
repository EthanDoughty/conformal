% Test: function in subdir/ resolves when maxDepth >= 2 (CLI only; TestRunner uses maxDepth=1)
% SKIP_TEST
function y = ws_cross_dir(x)
    y = ws_subdir_helper(x);
end
