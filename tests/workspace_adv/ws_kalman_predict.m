function [x_pred, P_pred] = ws_kalman_predict(F, x, P, Q)
    x_pred = F * x;
    P_pred = F * P * F' + Q;
end
