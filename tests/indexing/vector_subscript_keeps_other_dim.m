% Vector-as-subscript extent: an unknown row extent must not erase a known
% column count. find() has unknown numel, so pos's row extent stays unknown,
% but the column extent (4) is exactly known and must survive.
% EXPECT: pos = matrix[None x 4]
% EXPECT: idx = matrix[1 x None]
% EXPECT: warnings = 1

data = randn(500, 4);
labels = data(:, 4);
idx = find(labels > 0);  % EXPECT_WARNING: W_SUSPICIOUS_COMPARISON
pos = data(idx, :);
