function [r, g, t, s] = ws_fan_out(A)
    r = ws_reduce(A);
    g = ws_gram(A);
    [t, s] = ws_transform(A);
end
