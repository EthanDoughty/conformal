% Regression: interprocedural assignin('caller') eviction.
% A subfunction that calls assignin('caller',...) must cause the caller's
% trustedConsts to be evicted after the call returns.
%
% Case 6: assignin sets n to a new integer (9 vs 4); z fold must not use 4.
% Case 11: assignin sets n to rand(); z fold must not use 10.
%
% EXPECT: warnings = 0

n6 = 4;
sneaky6();
z6 = 0:1:n6;
w6 = [z6; 1 2 3 4 5 6 7 8 9 10];
% EXPECT: z6 = matrix[1 x None]
% EXPECT: w6 = matrix[2 x None]

n11 = 10;
sneaky11();
z11 = 0:n11/4:n11;
d11 = zeros(1, 3);
w11 = d11 + z11;
% EXPECT: z11 = matrix[1 x None]
% EXPECT: w11 = matrix[1 x None]

function sneaky6()
    assignin('caller', 'n6', 9);
end

function sneaky11()
  assignin('caller', 'n11', rand());
end
