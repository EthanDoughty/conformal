% Test: Cache hit replays warnings at each call site with correct line numbers
% First call analyzes and caches warning; second call replays at different line
% EXPECT: warnings = 2
% EXPECT: A = unknown
% EXPECT: B = unknown

function y = inner_mismatch(x)
    y = x * x;
end

A = inner_mismatch(zeros(3, 4));  % Line 10: cache miss, emit warning
B = inner_mismatch(zeros(3, 4));  % Line 11: cache hit, replay warning
