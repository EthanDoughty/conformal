function [T, S] = ws_transform(A)
    T = A';
    S = A * A';
end
