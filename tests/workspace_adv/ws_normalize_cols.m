function N = ws_normalize_cols(v)
    s = norm(v);
    N = v ./ s;
end
