% Stress Test: Control Flow Edge Cases
% Nested constructs, early returns, branch interactions.
% EXPECT: warnings = 0

% ==========================================================================
% Break inside switch inside for loop
% ==========================================================================
A = zeros(3, 3);
for i = 1:5
    switch i
        case 3
            break;
        otherwise
            A = A + eye(3);
    end
end
% EXPECT: A = matrix[3 x 3]

% ==========================================================================
% Continue inside switch inside while loop
% ==========================================================================
B = ones(4, 4);
j = 0;
while j < 10
    j = j + 1;
    switch j
        case 5
            continue;
        otherwise
            B = B + ones(4, 4);
    end
end
% EXPECT: B = matrix[4 x 4]

% ==========================================================================
% Switch with no otherwise — join includes pre-switch env
% ==========================================================================
X = zeros(2, 2);
val = 1;
switch val
    case 1
        X = ones(2, 2);
    case 2
        X = eye(2);
end
% NOTE: X should be matrix[2 x 2] (all branches agree on shape)
% EXPECT: X = matrix[2 x 2]

% ==========================================================================
% Deep nesting: for → if → switch → try → while (5 levels)
% ==========================================================================
result = zeros(3, 3);
for k = 1:3
    if k > 0
        switch k
            case 1
                try
                    m = 0;
                    while m < 2
                        result = result + eye(3);
                        m = m + 1;
                    end
                catch
                    result = result;
                end
            otherwise
                result = result + ones(3, 3);
        end
    end
end
% EXPECT: result = matrix[3 x 3]

% ==========================================================================
% IfChain with mixed return/non-return branches (inside function)
% ==========================================================================
function out = mixed_returns(x)
    if x > 5
        out = zeros(3, 3);
        return;
    elseif x > 0
        out = ones(3, 3);
    else
        out = eye(3);
    end
end

mr1 = mixed_returns(10);
mr2 = mixed_returns(3);
% EXPECT: mr1 = matrix[3 x 3]
% EXPECT: mr2 = matrix[3 x 3]

% ==========================================================================
% Try/catch with return in try body (inside function)
% ==========================================================================
function res = try_return(flag)
    try
        if flag > 0
            res = zeros(4, 4);
            return;
        end
        res = ones(4, 4);
    catch
        res = eye(4);
    end
end

tr1 = try_return(1);
tr2 = try_return(0);
% EXPECT: tr1 = matrix[4 x 4]
% EXPECT: tr2 = matrix[4 x 4]

% ==========================================================================
% Post-loop environment correctness after break
% ==========================================================================
C = zeros(5, 5);
for ii = 1:10
    if ii > 3
        break;
    end
    C = C + ones(5, 5);
end
% C is still matrix[5 x 5] after the loop
% EXPECT: C = matrix[5 x 5]

% ==========================================================================
% For loop with constraint recording in body
% ==========================================================================
function z = loop_constraint(P, Q)
    z = zeros(3, 3);
    for idx = 1:3
        z = z + P * Q;
    end
end

P = zeros(3, 4);
Q = ones(4, 3);
lc = loop_constraint(P, Q);
% EXPECT: lc = matrix[3 x 3]

% ==========================================================================
% Nested for loops with shape-preserving ops
% ==========================================================================
D = zeros(3, 3);
for r = 1:3
    for cc = 1:3
        D = D + eye(3);
    end
end
% EXPECT: D = matrix[3 x 3]
