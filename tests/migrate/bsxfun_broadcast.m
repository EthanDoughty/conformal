A = ones(3, 5);
mu = mean(A, 2);
X = bsxfun(@minus, A, mu);
Y = bsxfun(@times, A, mu);
Z = bsxfun(@plus, A, mu);
W = bsxfun(@rdivide, A, mu);
cmp = bsxfun(@lt, A, mu);
