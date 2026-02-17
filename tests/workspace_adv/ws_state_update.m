function x_next = ws_state_update(A, x, B, u)
    x_next = A * x + B * u;
end
