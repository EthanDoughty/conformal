function x_new = ws_gradient_step(x, alpha, grad)
    x_new = x - alpha * grad;
end
