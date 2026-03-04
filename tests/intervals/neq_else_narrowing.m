% Test: else-branch narrowing for ~= on integer loop variable endpoint.
% When t in [1,16] and t == 1 takes the then-branch, else narrows to [2,16].
% t-1 in else should be [1,15], not [0,15] — no OOB.
% EXPECT: warnings = 0
T = 16;
A = eye(2);
B = zeros(2, 1);
update_mu = zeros(2, T);
predict_mu = zeros(2, T);
init_mu = zeros(2, 1);
for t = 1:T
    if t == 1
        predict_mu(:,1) = init_mu;
    else
        predict_mu(:,t) = A * update_mu(:,t-1) + B;
    end
end
