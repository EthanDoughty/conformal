function result = ws_builtin_chain(A)
    d = diag(sum(A));
    result = repmat(d, 1, 2);
end
